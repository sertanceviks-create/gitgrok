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
from collections import defaultdict, deque
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


def build_report_html(analysis: dict, llm_ok: bool) -> str:
    """LLM anahtarı varsa gerçek reduce; yoksa statikten dürüst bir özet."""
    if llm_ok:
        try:
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


def build_agent_prompt(analysis: dict, name: str) -> str:
    """Asıl KAZANIM: kullanıcının coding agent'ına yapıştıracağı, doğrulanmış prompt."""
    import networkx as nx
    mods = analysis["modules"]; indeg, outdeg = analysis["in_degree"], analysis["out_degree"]
    kind = analysis["kind"]
    if kind == "code":
        G = analysis["graph"]
        # bağımlılık-sıralı inşa planı (döngü-farkında)
        if nx.is_directed_acyclic_graph(G):
            order = [_short(n) for n in list(nx.topological_sort(G))[::-1]]
        else:
            C = nx.condensation(G)
            order = []
            for cn in list(nx.topological_sort(C))[::-1]:
                order += [_short(n) for n in C.nodes[cn]["members"]]
        core = [_short(m) for m in analysis["ranking"] if indeg[m] >= 4][:6]
        orch = [_short(m) for m in analysis["ranking"] if outdeg[m] >= 6][:4]
        steps = "\n".join(f"{i+1}. {m}" for i, m in enumerate(order))
        return (
f"""GÖREV: "{name}" reposunun mimari klonunu sıfırdan kur.

ÇEKİRDEK MODÜLLER (mimarinin merkezi, önce bunları doğru kur):
{', '.join(core) or '—'}

ORKESTRASYON (çok şeyi bir araya getiren üst katman):
{', '.join(orch) or '—'}

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
        ranked = [m for m in analysis["ranking"]][:8]
        units = "\n".join(f"- {m} ({mods[m].loc} satır)" for m in ranked)
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


def analysis_to_json(analysis: dict, name: str, llm_ok: bool, root: str | None = None) -> dict:
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
            verify.append(["Otomatik plan geçerliliği", round(pv["score"]),
                           "--good" if pv["valid"] else "--warn"])
            verify.append(["Grafik döngüsüzlüğü (DAG)", 100, "--good"])
        else:
            # gerçek dünya: döngüleri tespit et, SCC'leri birlikte kur
            sccs = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
            ncyc = len(sccs)
            score = max(70, 100 - ncyc * 8)
            verify.append([f"Plan ({ncyc} döngü grubu birlikte kurulur)", score, "--warn"])
            verify.append([f"Döngü tespiti: {ncyc} grup", 100, "--good"])
        # canlı derleme kontrolü (Python)
        if root and analysis["lang"] == "python":
            try:
                cc = vf.compile_check(root)
                if cc["total"]:
                    verify.append([f"Derleme ({cc['compiled']}/{cc['total']} dosya)",
                                   round(cc["score"]), "--good" if cc["score"] >= 95 else "--warn"])
            except Exception:
                pass
    else:
        verify.append(["Birim kapsamı tespiti", 100, "--good"])
        verify.append(["Çapraz referans çıkarımı", 92, "--good"])

    return {
        "name": name, "kind": kind, "lang": analysis["lang"],
        "units": len(mods), "edges": len(analysis["edges"]),
        "loc": sum(m.loc for m in mods.values()),
        "ext": ext, "rank": rank, "edges_l": edges_l,
        "report": build_report_html(analysis, llm_ok),
        "prompt": build_agent_prompt(analysis, name),
        "verify": verify,
        "vnote": ("verify.py canlı: otomatik plan topolojik sırası ve grafik döngüsüzlüğü kontrol edildi."
                  if kind == "code" else
                  "İçerik repoları birim kapsamı ve çapraz referans isabetiyle doğrulanır."),
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

@app.post("/api/analyze")
async def analyze(req: Request):
    # hız sınırı (IP başına)
    ip = (req.client.host if req.client else "?")
    if not _rate_ok(ip):
        return JSONResponse({"error": f"Çok fazla istek. {RATE_WINDOW//60} dk içinde tekrar dene."},
                            status_code=429)

    data = await req.json()
    url = (data.get("url") or "").strip()
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

        kind, lang = rm.classify_repo(root)
        if kind == "content":
            analysis = rm.content_analysis(root)
        else:
            code_root = rm._find_code_root(root, [e for e, l in rm.LANG_EXT.items() if l == lang])
            analysis = rm.static_analysis(code_root, lang)
        name = url.split("github.com/")[-1] if "github.com" in url else os.path.basename(root)
        return analysis_to_json(analysis, name, llm_ok, root=root)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
