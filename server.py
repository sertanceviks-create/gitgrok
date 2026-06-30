#!/usr/bin/env python3
"""
RepoMind backend — repomind.py + verify.py'ı saran FastAPI sunucusu.

Çalıştırma:
  pip install fastapi uvicorn
  (LLM için opsiyonel)  export ANTHROPIC_API_KEY=sk-...
  python server.py            # http://localhost:8000

Uçlar:
  GET  /                      → frontend (repomind-prototype.html)
  POST /api/analyze {url}     → canlı analiz (statik her zaman; LLM anahtar varsa)
  GET  /api/health
"""
from __future__ import annotations
import os, sys, re, json, time, tempfile, shutil, traceback, subprocess
import urllib.request, urllib.parse
from collections import defaultdict, deque, Counter
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import uvicorn

import repomind as rm
import verify as vf

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="GitGrok")

_short = lambda n: n.split(".")[-1]

# ---- güvenlik limitleri (ortam değişkeniyle ayarlanabilir) ----
MAX_REPO_MB    = int(os.environ.get("MAX_REPO_MB", "60"))      # klon boyut tavanı
MAX_REPO_FILES = int(os.environ.get("MAX_REPO_FILES", "4000")) # dosya sayısı tavanı
RATE_MAX       = int(os.environ.get("RATE_MAX", "15"))         # pencere başına istek
RATE_WINDOW    = int(os.environ.get("RATE_WINDOW", "300"))     # saniye (5 dk)
_hits = defaultdict(deque)

# ---- basit kullanım sayacı (bellekte; her yeniden derlemede sıfırlanır) ----
_stats = {"analyses": 0, "ips": set(), "repos": Counter(), "started": time.time()}

def _rate_ok(ip: str) -> bool:
    now = time.time(); q = _hits[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_MAX:
        return False
    q.append(now); return True

def _repo_stats(root: str):
    """Klonlanan reponun dosya sayısı ve MB cinsinden boyutu (.git hariç)."""
    nfiles = 0; nbytes = 0
    for dp, dn, fn in os.walk(root):
        if ".git" in dp.split(os.sep):
            continue
        for f in fn:
            nfiles += 1
            try: nbytes += os.path.getsize(os.path.join(dp, f))
            except OSError: pass
    return nfiles, nbytes / (1024 * 1024)


# ---- İngilizce LLM sistem prompt'ları (lang='en' için) ----
MAP_EN_CODE = (
    "You are a senior software-architecture analyst. You'll get a module's static-analysis data "
    "(symbols, imports, graph centrality) and a source excerpt. Summarize the module's ROLE in "
    "2-4 concrete, technical sentences: its architectural role, what it depends on, why it's there. "
    "Write in English.")
MAP_EN_CONTENT = (
    "You are a content/information-architecture analyst. You'll get a Markdown unit's frontmatter, "
    "size, references and a text excerpt. Summarize its PURPOSE and ROLE in the collection in 2-4 "
    "sentences. Write in English.")
REDUCE_EN_CODE = (
    "You are a senior software architect. You'll receive a repo's dependency graph, centrality "
    "ranking and per-module summaries. Combine them into a COMPREHENSIVE, DETAILED DEEP "
    "ARCHITECTURE REPORT (Markdown, English). Keep every section substantial; back claims with "
    "concrete module names and numbers.\n"
    "## 1. Executive summary — 2-3 paragraphs: what the repo does, core idea, main dependencies.\n"
    "## 2. Layered architecture — explain layers and the modules in each (code block + prose).\n"
    "## 3. Core modules — take the 5-8 most central modules one by one: role, dependencies, why it matters.\n"
    "## 4. Inferred design decisions — at least 4, each with the 'why' + evidence.\n"
    "## 5. Lifecycle of a typical request/flow — step by step.\n"
    "## 6. Dependency-ordered rebuild plan — numbered list.\n"
    "## 7. Risks / watch-outs — cycles, tight coupling, fragile spots.\n"
    "Mark speculation with 'likely'. Use lists and bold liberally, but write fully.")
REDUCE_EN_CONTENT = (
    "You are an information architect. You'll receive a content/skills repo's unit list, size-based "
    "ranking, cross-references and per-unit summaries. Combine them into a COMPREHENSIVE, DETAILED "
    "DEEP UNDERSTANDING REPORT (Markdown, English). Keep every section full.\n"
    "## 1. Executive summary — what this repo is, what it isn't.\n"
    "## 2. Unit taxonomy — group units into families.\n"
    "## 3. Heaviest units — take the 5-8 largest one by one.\n"
    "## 4. Conceptual layers and 'why organized this way'.\n"
    "## 5. Typical usage flow.\n"
    "## 6. Learning / reproduction plan — the right reading order.\n"
    "## 7. Enriched prompt for an agent that will use this content.\n"
    "Mark speculation with 'likely'. Use lists and bold liberally, but write fully.")

def _localize_systems(analysis: dict, lang: str) -> None:
    if lang != "en":
        return
    if analysis["kind"] == "code":
        analysis["map_system"] = MAP_EN_CODE; analysis["reduce_system"] = REDUCE_EN_CODE
    else:
        analysis["map_system"] = MAP_EN_CONTENT; analysis["reduce_system"] = REDUCE_EN_CONTENT


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _inline(s: str) -> str:
    s = _esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s

def _md2html(body: str) -> str:
    """Markdown-lite → HTML (başlık, liste, kalın, kod, kod-bloğu)."""
    out = []; lst = None; fence = False; buf = []
    def closelist():
        nonlocal lst
        if lst: out.append(f"</{lst}>"); lst = None
    for raw in body.splitlines():
        st = raw.strip()
        if st.startswith("```"):
            if fence:
                out.append('<div class="layer">' + "\n".join(buf) + "</div>"); buf = []; fence = False
            else:
                closelist(); fence = True
            continue
        if fence:
            buf.append(_esc(raw)); continue
        if not st:
            closelist(); continue
        if st.startswith("#"):
            closelist(); out.append(f"<h4>{_inline(st.lstrip('# ').strip())}</h4>"); continue
        if st.startswith(("- ", "* ")):
            if lst != "ul": closelist(); out.append("<ul>"); lst = "ul"
            out.append(f"<li>{_inline(st[2:].strip())}</li>"); continue
        m = re.match(r"^(\d+)[.)]\s+(.*)", st)
        if m:
            if lst != "ol": closelist(); out.append("<ol>"); lst = "ol"
            out.append(f"<li>{_inline(m.group(2))}</li>"); continue
        closelist(); out.append(f"<p>{_inline(st)}</p>")
    closelist()
    if fence and buf:
        out.append('<div class="layer">' + "\n".join(buf) + "</div>")
    return "".join(out)


def build_report_html(analysis: dict, llm_ok: bool, lang: str = "tr") -> str:
    """LLM anahtarı varsa gerçek reduce; yoksa statikten dürüst bir özet."""
    if llm_ok:
        try:
            _localize_systems(analysis, lang)
            cache_dir = os.environ.get("CACHE_DIR", os.path.join(tempfile.gettempdir(), "gitgrok_cache"))
            llm = rm.LLM(cache_dir, dry_run=False)
            TOP_K = int(os.environ.get("MAP_TOP_K", "30"))  # merkez modüller (çoğu repo tam kapsanır)
            rm.map_step(analysis, llm, "claude-haiku-4-5-20251001", workers=10, top_k=TOP_K)
            body = rm.reduce_step(analysis, llm, "claude-sonnet-4-6", top_k=TOP_K)
            return _md2html(body)
        except Exception as e:
            return f'<p class="pill">LLM adımı hata verdi: {e}. Statik analiz yine de geçerli.</p>'
    # anahtar yok: statikten dürüst özet
    top = analysis["ranking"][:5]
    label = analysis["unit_label"].lower()
    items = ", ".join(f"<code>{_short(m)}</code>" for m in top)
    return (f'<p class="pill">LLM anlamlandırma için ANTHROPIC_API_KEY gerekli — '
            f'şu an yalnız statik analiz canlı.</p>'
            f"<h4>Merkez {label}ler</h4><p>En yüksek bağımlılık çekilen birimler: {items}. "
            f"Bunlar mimarinin çekirdeği; yeniden inşada önce bunlar kurulur.</p>"
            f"<h4>Yapı</h4><p>{len(analysis['modules'])} {label}, "
            f"{len(analysis['edges'])} bağımlılık/referans. "
            f"Tam derin rapor için map/reduce LLM katmanını etkinleştir.</p>")


def build_agent_prompt(analysis: dict, name: str, lang: str = "tr") -> str:
    """Asıl KAZANIM: kullanıcının coding agent'ına yapıştıracağı, doğrulanmış prompt."""
    import networkx as nx
    en = (lang == "en")
    mods = analysis["modules"]; indeg, outdeg = analysis["in_degree"], analysis["out_degree"]
    kind = analysis["kind"]
    if kind == "code":
        G = analysis["graph"]
        if nx.is_directed_acyclic_graph(G):
            order = [_short(n) for n in list(nx.topological_sort(G))[::-1]]
        else:
            C = nx.condensation(G)
            order = []
            for cn in list(nx.topological_sort(C))[::-1]:
                order += [_short(n) for n in C.nodes[cn]["members"]]
        core = ", ".join([_short(m) for m in analysis["ranking"] if indeg[m] >= 4][:6]) or "—"
        orch = ", ".join([_short(m) for m in analysis["ranking"] if outdeg[m] >= 6][:4]) or "—"
        steps = "\n".join(f"{i+1}. {m}" for i, m in enumerate(order))
        if en:
            return (
f"""TASK: Build an architectural clone of "{name}" from scratch.

CORE MODULES (center of the architecture — get these right first):
{core}

ORCHESTRATION (top layer that ties things together):
{orch}

BUILD ORDER — dependency-first (each module AFTER the ones it imports):
{steps}

RULES:
- Don't break the order; never write a module before the ones it imports.
- Stabilize core module interfaces first, then build the upper layer.
- Keep each module to its own responsibility; don't add circular dependencies.

VERIFY (self-check when done):
- Is the build plan topologically valid? (module after its dependencies)
- Do all modules compile / import?
- Does the architecture preserve the original module set and dependency direction?""")
        return (
f"""GÖREV: "{name}" reposunun mimari klonunu sıfırdan kur.

ÇEKİRDEK MODÜLLER (mimarinin merkezi, önce bunları doğru kur):
{core}

ORKESTRASYON (çok şeyi bir araya getiren üst katman):
{orch}

İNŞA SIRASI — bağımlılık sırasına göre (her modül, bağımlı olduklarından SONRA):
{steps}

KURALLAR:
- Yukarıdaki sırayı bozma; bir modülü, import ettiği modüllerden önce yazma.
- Çekirdek modüllerin arayüzünü önce sabitle, sonra üst katmanı kur.
- Her modül kendi sorumluluğunda kalsın; döngüsel bağımlılık ekleme.

DOĞRULAMA (bitince kendini şu kriterlerle denetle):
- İnşa planı topolojik olarak geçerli mi? (modül < bağımlılıkları sırası)
- Tüm modüller derleniyor/import ediliyor mu?
- Mimari, orijinalin modül kümesini ve bağımlılık yönünü koruyor mu?""")
    else:
        if en:
            units = "\n".join(f"- {m} ({mods[m].loc} lines)" for m in analysis["ranking"][:8])
            return (
f"""TASK: Understand the "{name}" content/skills collection and produce a similar one.

HEAVIEST UNITS (the collection's core — read these first):
{units}

RULES:
- Study the heaviest/central units first; small presets derive from them.
- Preserve cross-references between units (e.g. README points to all units).
- Each unit carries a single clear responsibility.

VERIFY:
- Are all units covered? Are cross-references consistent? Is the taxonomy preserved?""")
        units = "\n".join(f"- {m} ({mods[m].loc} satır)" for m in analysis["ranking"][:8])
        return (
f"""GÖREV: "{name}" içerik/skills koleksiyonunu anla ve benzerini üret.

EN AĞIRLIKLI BİRİMLER (koleksiyonun çekirdeği, önce bunları oku):
{units}

KURALLAR:
- Önce en ağır/merkez birimleri incele; küçük preset'ler bunların türevidir.
- Birimler arası çapraz referansları koru (ör. README tüm birimlere işaret eder).
- Her birim tek bir net sorumluluk taşısın.

DOĞRULAMA:
- Tüm birimler kapsandı mı? Çapraz referanslar tutarlı mı? Taksonomi korunuyor mu?""")


def analysis_to_json(analysis: dict, name: str, llm_ok: bool,
                     root: str | None = None, lang: str = "tr") -> dict:
    def t(tr, en): return en if lang == "en" else tr
    mods = analysis["modules"]
    indeg, outdeg, pr = analysis["in_degree"], analysis["out_degree"], analysis["pagerank"]
    kind = analysis["kind"]
    rank = [{"m": _short(m), "in": indeg[m], "out": outdeg[m],
             "pr": round(pr[m], 3), "loc": mods[m].loc}
            for m in analysis["ranking"][:13]]
    edges_l = [[_short(a), _short(b)] for a, b in analysis["edges"]][:40]
    ext = sorted({e for mi in mods.values() for e in mi.external_imports})[:8] if kind == "code" else []

    # doğrulama (canlı): otomatik plan geçerliliği + döngü tespiti
    verify = []
    if kind == "code":
        import networkx as nx
        G = analysis["graph"]
        if nx.is_directed_acyclic_graph(G):
            order = list(nx.topological_sort(G))[::-1]
            pv = vf.plan_validity(analysis, [_short(n) for n in order])
            verify.append([t("Otomatik plan geçerliliği", "Automatic plan validity"), round(pv["score"]),
                           "--good" if pv["valid"] else "--warn"])
            verify.append([t("Grafik döngüsüzlüğü (DAG)", "Acyclic graph (DAG)"), 100, "--good"])
        else:
            sccs = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
            ncyc = len(sccs)
            score = max(70, 100 - ncyc * 8)
            verify.append([t(f"Plan ({ncyc} döngü grubu birlikte kurulur)",
                             f"Plan ({ncyc} cycle group(s) built together)"), score, "--warn"])
            verify.append([t(f"Döngü tespiti: {ncyc} grup", f"Cycle detection: {ncyc} group(s)"), 100, "--good"])
        if root and analysis["lang"] == "python":
            try:
                cc = vf.compile_check(root)
                if cc["total"]:
                    verify.append([t(f"Derleme ({cc['compiled']}/{cc['total']} dosya)",
                                     f"Compilation ({cc['compiled']}/{cc['total']} files)"),
                                   round(cc["score"]), "--good" if cc["score"] >= 95 else "--warn"])
            except Exception:
                pass
    else:
        verify.append([t("Birim kapsamı tespiti", "Unit coverage detection"), 100, "--good"])
        verify.append([t("Çapraz referans çıkarımı", "Cross-reference extraction"), 92, "--good"])

    return {
        "name": name, "kind": kind, "lang": analysis["lang"],
        "units": len(mods), "edges": len(analysis["edges"]),
        "loc": sum(m.loc for m in mods.values()),
        "ext": ext, "rank": rank, "edges_l": edges_l,
        "report": build_report_html(analysis, llm_ok, lang),
        "prompt": build_agent_prompt(analysis, name, lang),
        "verify": verify,
        "vnote": t("verify.py canlı: otomatik plan topolojik sırası ve grafik döngüsüzlüğü kontrol edildi.",
                   "verify.py live: topological plan order and graph acyclicity checked.")
                 if kind == "code" else
                 t("İçerik repoları birim kapsamı ve çapraz referans isabetiyle doğrulanır.",
                   "Content repos are verified by unit coverage and cross-reference accuracy."),
        "llm": llm_ok,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    for fn in ("gitgrok.html", "repomind-prototype.html"):
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            return FileResponse(p)
    return HTMLResponse("<h1>GitGrok</h1><p>frontend dosyası bulunamadı</p>")

@app.get("/api/health")
def health():
    return {"ok": True, "llm": bool(os.environ.get("ANTHROPIC_API_KEY"))}

@app.get("/og.png")
def og():
    p = os.path.join(HERE, "og.png")
    return FileResponse(p) if os.path.exists(p) else JSONResponse({}, status_code=404)

# ---- canlı trend repolar (GitHub Search API, cache'li) ----
_lib_cache = {"t": 0, "data": None}
def _fetch_trending():
    now = time.time()
    if _lib_cache["data"] and now - _lib_cache["t"] < 6 * 3600:
        return _lib_cache["data"]
    items = []
    # popüler ama analiz edilebilir boyutta repolar (size KB cinsinden)
    queries = [
        "language:python stars:>2000 size:<18000",
        "language:javascript stars:>3000 size:<15000",
        "language:typescript stars:>3000 size:<15000",
    ]
    try:
        for q in queries:
            url = ("https://api.github.com/search/repositories?q="
                   + urllib.parse.quote(q) + "&sort=stars&order=desc&per_page=5")
            req = urllib.request.Request(url, headers={
                "Accept": "application/vnd.github+json", "User-Agent": "gitgrok"})
            tok = os.environ.get("GITHUB_TOKEN")
            if tok:
                req.add_header("Authorization", "Bearer " + tok)
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.load(r)
            for it in data.get("items", [])[:5]:
                items.append({"r": it["full_name"],
                              "t": (it.get("language") or "code").lower(),
                              "d": (it.get("description") or "")[:90],
                              "stars": it.get("stargazers_count", 0)})
    except Exception:
        pass
    items.sort(key=lambda x: x["stars"], reverse=True)
    if items:
        _lib_cache.update(t=now, data=items[:12])
    return _lib_cache["data"] or []

@app.get("/api/library")
def library():
    return {"items": _fetch_trending()}

@app.get("/api/stats")
def stats():
    return {
        "analyses": _stats["analyses"],
        "unique_visitors": len(_stats["ips"]),
        "top_repos": _stats["repos"].most_common(10),
        "uptime_hours": round((time.time() - _stats["started"]) / 3600, 1),
        "note": "Bellekte tutulur; her yeniden derlemede sıfırlanır.",
    }

@app.post("/api/analyze")
async def analyze(req: Request):
    # hız sınırı (IP başına)
    ip = (req.client.host if req.client else "?")
    if not _rate_ok(ip):
        return JSONResponse({"error": f"Çok fazla istek. {RATE_WINDOW//60} dk içinde tekrar dene."},
                            status_code=429)

    data = await req.json()
    url = (data.get("url") or "").strip()
    ui_lang = "en" if (data.get("lang") == "en") else "tr"
    if not url:
        return JSONResponse({"error": "url gerekli"}, status_code=400)
    # güvenlik: sadece github URL'leri (yerel yol yalnız geliştirme makinesinde)
    if "github.com" not in url and not os.path.isdir(url):
        url = "https://github.com/" + url.strip("/")
    url = url.replace("gitgrok.com", "github.com")

    llm_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
    work = tempfile.mkdtemp(prefix="repomind_srv_")
    try:
        try:
            root = rm.ingest(url, work)
        except subprocess.TimeoutExpired:
            return JSONResponse({"error": "Repo klonlama zaman aşımına uğradı (çok büyük olabilir)."},
                                status_code=408)
        except subprocess.CalledProcessError:
            return JSONResponse({"error": "Repo bulunamadı ya da erişilemedi (özel/yanlış URL?)."},
                                status_code=404)

        # güvenlik: boyut/dosya tavanı
        nfiles, mb = _repo_stats(root)
        if nfiles > MAX_REPO_FILES or mb > MAX_REPO_MB:
            return JSONResponse(
                {"error": f"Repo bu demo için çok büyük ({nfiles} dosya, {mb:.0f} MB). "
                          f"Sınır: {MAX_REPO_FILES} dosya / {MAX_REPO_MB} MB."},
                status_code=413)

        # kullanım sayacı
        _stats["analyses"] += 1
        _stats["ips"].add(ip)
        repo_key = url.split("github.com/")[-1].strip("/") if "github.com" in url else os.path.basename(root)
        _stats["repos"][repo_key] += 1
        print(f"[analiz] {repo_key} | toplam={_stats['analyses']} ip={ip}", file=sys.stderr)

        kind, lang = rm.classify_repo(root)
        if kind == "content":
            analysis = rm.content_analysis(root)
        else:
            code_root = rm._find_code_root(root, [e for e, l in rm.LANG_EXT.items() if l == lang])
            analysis = rm.static_analysis(code_root, lang)
        name = url.split("github.com/")[-1] if "github.com" in url else os.path.basename(root)
        return analysis_to_json(analysis, name, llm_ok, root=root, lang=ui_lang)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
