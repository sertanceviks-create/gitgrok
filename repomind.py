#!/usr/bin/env python3
"""
RepoMind — Advanced GitReverse MVP (Faz 1)
==========================================
GitHub reposunu derin, doğrulanabilir bir mimari rapora çevirir.

Pipeline:
  1. ingest   : repoyu klonla (sığ)
  2. parse    : Tree-sitter ile AST -> sembol + import çıkarımı
  3. graph    : bağımlılık grafiği + merkezilik skoru (networkx)
  4. map      : her modül için paralel LLM özeti  (ucuz/hızlı model)
  5. reduce   : grafik + tüm özetlerden sistem sentezi (güçlü model)
  6. render   : Markdown derin rapor + agent prompt

Kullanım:
  export ANTHROPIC_API_KEY=sk-...
  python repomind.py https://github.com/psf/requests
  python repomind.py /yerel/yol --reduce-model claude-opus-4-8
  python repomind.py https://github.com/psf/requests --dry-run   # LLM'siz, sadece statik

Bağımlılıklar:
  pip install tree-sitter tree-sitter-python tree-sitter-javascript \
              networkx scipy anthropic
"""
from __future__ import annotations
import os, sys, json, glob, time, hashlib, argparse, subprocess, tempfile, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict

import networkx as nx

# ----------------------------------------------------------------------------
# Dil desteği (genişletilebilir). Şimdilik Python + JS/TS iskeleti.
# ----------------------------------------------------------------------------
LANG_EXT = {".py": "python", ".js": "javascript", ".jsx": "javascript",
            ".ts": "javascript", ".tsx": "javascript"}

def _load_language(lang: str):
    if lang == "python":
        import tree_sitter_python as ts
    elif lang == "javascript":
        try:
            import tree_sitter_javascript as ts
        except ImportError:
            return None
    else:
        return None
    from tree_sitter import Language
    return Language(ts.language())


# ----------------------------------------------------------------------------
# 1. INGEST
# ----------------------------------------------------------------------------
def ingest(target: str, workdir: str) -> str:
    """GitHub URL ise klonla, yerel yol ise olduğu gibi kullan. Repo KÖKÜNÜ döndür."""
    if target.startswith("http") or target.endswith(".git"):
        dest = os.path.join(workdir, "repo")
        subprocess.run(["git", "clone", "--depth=1", target, dest],
                       check=True, capture_output=True)
        return dest
    return os.path.abspath(target)


def _find_code_root(root: str, exts: list[str]) -> str:
    """Kod dosyalarının yoğunlaştığı alt dizini (src/lib) bul, yoksa kökü döndür."""
    for cand in ("src", "lib", "."):
        p = os.path.join(root, cand)
        if os.path.isdir(p):
            for ext in exts:
                if glob.glob(os.path.join(p, f"**/*{ext}"), recursive=True):
                    return p
    return root


# ----------------------------------------------------------------------------
# 1b. REPO-TİPİ TESPİTİ (yeni: doğru merceği seçer)
# ----------------------------------------------------------------------------
def _count_files(root: str, exts: tuple) -> int:
    n = 0
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in IGNORE_DIRS and not d.startswith(".git")]
        n += sum(1 for f in fn if os.path.splitext(f)[1] in exts)
    return n

def classify_repo(root: str) -> tuple[str, str | None]:
    """('code', lang) ya da ('content', None) döndür."""
    # Güçlü içerik sinyalleri
    if glob.glob(os.path.join(root, "skills", "*", "SKILL.md")) or \
       glob.glob(os.path.join(root, "**/SKILL.md"), recursive=True):
        return ("content", None)
    code_exts = (".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java")
    n_code = _count_files(root, code_exts)
    n_md = _count_files(root, (".md", ".mdx"))
    # Çoğunluk Markdown ve çok az kod -> içerik reposu
    if n_md >= 5 and n_code <= 4 and n_md > n_code:
        return ("content", None)
    # Kod: en yaygın dili seç
    n_py = _count_files(root, (".py",))
    n_js = _count_files(root, (".js", ".jsx", ".ts", ".tsx"))
    return ("code", "python" if n_py >= n_js else "javascript")


# ----------------------------------------------------------------------------
# 2-3. PARSE + GRAPH
# ----------------------------------------------------------------------------
@dataclass
class ModuleInfo:
    name: str
    path: str
    loc: int
    classes: list = field(default_factory=list)
    functions: list = field(default_factory=list)
    internal_imports: list = field(default_factory=list)
    type_imports: list = field(default_factory=list)      # TYPE_CHECKING (build-order DIŞI)
    deferred_imports: list = field(default_factory=list)  # fonksiyon içi (döngü kırıcı, DIŞI)
    external_imports: list = field(default_factory=list)
    docstring: str = ""
    summary: str = ""  # map adımında doldurulur

IGNORE_DIRS = {"node_modules", ".git", "dist", "build", "__pycache__",
               "venv", ".venv", "test", "tests", "__tests__"}

def _module_name(path: str, root: str) -> str:
    rel = os.path.relpath(path, root)
    rel = os.path.splitext(rel)[0].replace(os.sep, ".")
    if rel.endswith(".__init__"):
        rel = rel[:-9] or "__init__"
    return rel

def _txt(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")

def static_analysis(root: str, lang: str = "python") -> dict:
    """Tree-sitter ile sembol + import çıkar, bağımlılık grafiği kur."""
    LANGUAGE = _load_language(lang)
    if LANGUAGE is None:
        raise RuntimeError(f"'{lang}' için tree-sitter grammar yüklü değil.")
    from tree_sitter import Parser
    parser = Parser(LANGUAGE)

    exts = [e for e, l in LANG_EXT.items() if l == lang]
    files = []
    for ext in exts:
        for f in glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True):
            if not any(d in f.split(os.sep) for d in IGNORE_DIRS):
                files.append(f)
    files = sorted(set(files))

    modules: dict[str, ModuleInfo] = {}
    name_by_path = {f: _module_name(f, root) for f in files}
    modset = set(name_by_path.values())
    edges: list[tuple[str, str]] = []

    for path in files:
        name = name_by_path[path]
        src = open(path, "rb").read()
        tree = parser.parse(src)
        info = ModuleInfo(name=name, path=path, loc=src.count(b"\n") + 1)
        if lang == "python":
            _parse_python(tree.root_node, src, info, modset)
        else:
            _parse_js(tree.root_node, src, info, modset)
        runtime = set(info.internal_imports) - {name}
        info.internal_imports = sorted(runtime)
        # tip-only ve fonksiyon-içi importlar: build-order grafiğinden hariç
        info.type_imports = sorted(set(info.type_imports) - {name} - runtime)
        info.deferred_imports = sorted(set(info.deferred_imports) - {name} - runtime)
        info.external_imports = sorted({e for e in set(info.external_imports)
                                        if e and e[0].isalpha()})
        for tgt in info.internal_imports:
            edges.append((name, tgt))
        modules[name] = info

    G = nx.DiGraph()
    G.add_nodes_from(modules)
    G.add_edges_from(edges)
    pr = nx.pagerank(G) if G.number_of_edges() else {n: 0 for n in G}
    indeg, outdeg = dict(G.in_degree()), dict(G.out_degree())
    ranking = sorted(modules, key=lambda m: (indeg[m], pr[m]), reverse=True)

    return {
        "modules": modules, "graph": G, "edges": sorted(set(edges)),
        "pagerank": pr, "in_degree": indeg, "out_degree": outdeg,
        "ranking": ranking, "lang": lang, "kind": "code",
        "unit_label": "Modül", "weight_label": "loc",
        "map_system": MAP_SYSTEM_CODE, "reduce_system": REDUCE_SYSTEM_CODE,
    }

def _parse_python(root, src, info: ModuleInfo, modset):
    # Mevcut modülün paketi (göreli import çözümü için).
    # requests.sessions -> paket=requests ; requests (__init__) -> paket=requests
    parts = info.name.split(".")
    pkg = ".".join(parts[:-1]) if len(parts) > 1 else parts[0]

    def resolve_rel(raw: str, mod_after_dots: str, names: list[str], bucket: list):
        """Göreli importu mutlak modül adına çevirip modset'tekileri 'bucket'a ekle."""
        dots = len(raw) - len(raw.lstrip("."))
        base = pkg
        for _ in range(max(0, dots - 1)):  # her ekstra nokta bir üst paket
            base = base.rsplit(".", 1)[0] if "." in base else base
        # from .models import X  -> base.models
        if mod_after_dots:
            cand = f"{base}.{mod_after_dots}" if base else mod_after_dots
            if cand in modset:
                bucket.append(cand)
        # from . import sessions  -> base.sessions
        for nm in names:
            cand = f"{base}.{nm}" if base else nm
            if cand in modset:
                bucket.append(cand)

    # docstring (ilk string ifadesi)
    for c in root.children:
        if c.type == "expression_statement" and c.children and c.children[0].type == "string":
            doc = _txt(c.children[0], src).strip("\"' \n")
            info.docstring = doc.split("\n\n")[0].replace("\n", " ")[:240]
            break
    def walk(node, depth=0, type_ctx=False, in_func=False):
        for c in node.children:
            t = c.type
            # `if TYPE_CHECKING:` bloğu içindeki importlar tip-only kabul edilir
            child_type_ctx = type_ctx
            if t == "if_statement":
                cond = c.child_by_field_name("condition")
                if cond and "TYPE_CHECKING" in _txt(cond, src):
                    child_type_ctx = True
            child_in_func = in_func or (t == "function_definition")
            if t == "class_definition":
                n = c.child_by_field_name("name")
                if n: info.classes.append(_txt(n, src))
            elif t == "function_definition" and depth == 0:
                n = c.child_by_field_name("name")
                if n: info.functions.append(_txt(n, src))
            if t in ("import_statement", "import_from_statement"):
                bucket = (info.type_imports if type_ctx
                          else info.deferred_imports if in_func
                          else info.internal_imports)
                raw = _txt(c, src)
                if raw.lstrip().startswith("from ."):
                    fn = c.child_by_field_name("module_name")
                    mod_after = (_txt(fn, src).lstrip(".") if fn else "")
                    after = raw.split("from", 1)[1].lstrip()  # nokta sayısı için
                    names = []
                    for ch in c.children:
                        if ch.type in ("dotted_name", "aliased_import"):
                            names.append(_txt(ch, src).split(" as ")[0].strip())
                    resolve_rel(after, mod_after, names, bucket)
                else:
                    fn = c.child_by_field_name("module_name") if t == "import_from_statement" else None
                    if fn:
                        info.external_imports.append(_txt(fn, src).split(".")[0])
                    else:
                        for ch in c.children:
                            if ch.type in ("dotted_name", "aliased_import"):
                                info.external_imports.append(
                                    _txt(ch, src).split(" as ")[0].split(".")[0].strip())
            walk(c, depth + (1 if t in ("class_definition", "function_definition") else 0),
                 child_type_ctx, child_in_func)
    walk(root)

def _parse_js(root, src, info: ModuleInfo, modset):
    # Minimal JS/TS iskeleti: import ... from "..." ve export'lar
    def resolve(spec: str) -> str | None:
        if spec.startswith("."):
            base = os.path.normpath(spec).replace(os.sep, ".").lstrip(".")
            return base if base in modset else None
        return None
    def walk(node):
        for c in node.children:
            if c.type == "import_statement":
                strn = next((x for x in c.children if x.type == "string"), None)
                if strn:
                    spec = _txt(strn, src).strip("\"'`")
                    internal = resolve(spec)
                    (info.internal_imports if internal else info.external_imports).append(
                        internal or spec.split("/")[0])
            elif c.type in ("class_declaration",):
                n = c.child_by_field_name("name")
                if n: info.classes.append(_txt(n, src))
            elif c.type in ("function_declaration",):
                n = c.child_by_field_name("name")
                if n: info.functions.append(_txt(n, src))
            walk(c)
    walk(root)


# ----------------------------------------------------------------------------
# 2c. İÇERİK / SKILLS ANALİZİ (kod merceği yerine)
# ----------------------------------------------------------------------------
def _frontmatter(text: str) -> dict:
    """Basit YAML-benzeri frontmatter ayrıştırıcı (--- ... ---)."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    fm = {}
    for line in text[3:end].splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            fm[k.strip().lower()] = v.strip().strip("\"'")
    return fm

def content_analysis(root: str) -> dict:
    """Markdown/skills reposunu birim haritasına çevir.
    Birim = her SKILL.md veya önemli .md dosyası. Ağırlık = satır sayısı.
    Kenar = bir birimin gövdesinde başka bir birimin adını/yolunu anması."""
    md_files = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in IGNORE_DIRS and not d.startswith(".git")]
        for f in fn:
            if f.endswith((".md", ".mdx")):
                md_files.append(os.path.join(dp, f))
    # Birim adı: SKILL.md ise klasör adı; değilse repoya göreli yol
    units: dict[str, ModuleInfo] = {}
    for path in sorted(md_files):
        rel = os.path.relpath(path, root)
        if os.path.basename(path) == "SKILL.md":
            uid = os.path.basename(os.path.dirname(path))
            kind = "skill"
        else:
            uid = rel.replace(os.sep, "/")
            kind = "doc"
        text = open(path, encoding="utf-8", errors="replace").read()
        fm = _frontmatter(text)
        info = ModuleInfo(name=uid, path=path, loc=text.count("\n") + 1)
        info.classes = [kind] + ([fm.get("name")] if fm.get("name") else [])
        info.docstring = (fm.get("description") or text.lstrip("#").strip().split("\n")[0])[:240]
        units[uid] = info

    # install-name -> uid eşlemesi (çapraz referans için)
    install_names = {}
    for uid, info in units.items():
        for c in info.classes:
            if c and c not in ("skill", "doc"):
                install_names[c] = uid

    edges = []
    for uid, info in units.items():
        body = open(info.path, encoding="utf-8", errors="replace").read()
        refs = set()
        for nm, tgt in install_names.items():
            if tgt != uid and (nm in body):
                refs.add(tgt)
        info.internal_imports = sorted(refs)
        for tgt in refs:
            edges.append((uid, tgt))

    G = nx.DiGraph(); G.add_nodes_from(units); G.add_edges_from(edges)
    indeg, outdeg = dict(G.in_degree()), dict(G.out_degree())
    pr = nx.pagerank(G) if G.number_of_edges() else {n: 0 for n in G}
    # Ağırlık = satır sayısı (içerikte "merkezilik" boyuttan okunur) + referanslar
    ranking = sorted(units, key=lambda u: (units[u].loc, indeg[u]), reverse=True)

    return {
        "modules": units, "graph": G, "edges": sorted(set(edges)),
        "pagerank": pr, "in_degree": indeg, "out_degree": outdeg,
        "ranking": ranking, "lang": "markdown", "kind": "content",
        "unit_label": "Birim", "weight_label": "satır",
        "map_system": MAP_SYSTEM_CONTENT, "reduce_system": REDUCE_SYSTEM_CONTENT,
    }


# ----------------------------------------------------------------------------
# 4-5. LLM KATMANI (map + reduce) — gerçek Anthropic API çağrıları
# ----------------------------------------------------------------------------
MAP_SYSTEM_CODE = (
    "Sen bir kıdemli yazılım mimarisi analistisin. Sana bir modülün statik analiz "
    "verisi (semboller, importlar, grafikteki merkezilik) ve kaynak kod parçası "
    "verilecek. Modülün ROLÜNÜ 2-4 cümleyle, somut ve teknik biçimde özetle. "
    "Mimari rolüne, neye bağlı olduğuna ve neden orada olduğuna odaklan. Türkçe yaz."
)
REDUCE_SYSTEM_CODE = (
    "Sen bir kıdemli yazılım mimarısın. Sana bir reponun bağımlılık grafiği, "
    "merkezilik sıralaması ve modül modül özetler verilecek. Bunları birleştirip "
    "şu bölümleri içeren bir DERİN MİMARİ RAPORU üret (Markdown, Türkçe): "
    "1) Yönetici özeti, 2) Katmanlı mimari, 3) Çıkarımsal tasarım kararları ve 'neden', "
    "4) Bir tipik isteğin/akışın yaşam döngüsü, 5) Bağımlılık-sıralı yeniden inşa planı, "
    "6) Zenginleştirilmiş bir coding-agent prompt'u. Spekülasyonu 'muhtemelen' ile işaretle."
)
MAP_SYSTEM_CONTENT = (
    "Sen bir içerik/bilgi-mimarisi analistisin. Sana bir Markdown biriminin (bir Agent "
    "'skill' dosyası ya da doküman) frontmatter'ı, boyutu, diğer birimlere referansları "
    "ve metin parçası verilecek. Bu birimin AMACINI ve koleksiyondaki ROLÜNÜ 2-4 cümleyle "
    "özetle: ne işe yarar, hangi aileye aittir, neden önemlidir. Türkçe yaz."
)
REDUCE_SYSTEM_CONTENT = (
    "Sen bir bilgi-mimarisisin. Sana bir içerik/skills reposunun birim listesi, boyuta göre "
    "ağırlık sıralaması, çapraz referanslar ve birim birim özetler verilecek. Bunları "
    "birleştirip şu bölümleri içeren bir DERİN ANLAMA RAPORU üret (Markdown, Türkçe): "
    "1) Yönetici özeti (bu repo ne, ne değil), 2) Birim taksonomisi (aileler), "
    "3) Kavramsal katmanlar ve 'neden böyle düzenlenmiş', 4) Bir birimin tipik kullanım akışı, "
    "5) Öğrenme/yeniden-üretme planı (doğru okuma sırası), 6) İçeriği kullanacak bir ajan için "
    "zenginleştirilmiş prompt. Spekülasyonu 'muhtemelen' ile işaretle."
)

class LLM:
    def __init__(self, cache_dir: str, dry_run: bool = False):
        self.dry_run = dry_run
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.client = None
        self.usage = {"in": 0, "out": 0, "calls": 0, "cached": 0}
        if not dry_run:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise SystemExit("anthropic SDK gerekli: pip install anthropic")
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise SystemExit("ANTHROPIC_API_KEY tanımlı değil (veya --dry-run kullan).")
            self.client = Anthropic()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, hashlib.sha256(key.encode()).hexdigest()[:24] + ".txt")

    def call(self, system: str, prompt: str, model: str, max_tokens: int = 1024) -> str:
        cache_key = f"{model}\n{system}\n{prompt}"
        cp = self._cache_path(cache_key)
        if os.path.exists(cp):
            self.usage["cached"] += 1
            return open(cp).read()
        if self.dry_run:
            return f"[DRY-RUN özet — model={model}, prompt {len(prompt)} karakter]"
        for attempt in range(4):
            try:
                resp = self.client.messages.create(
                    model=model, max_tokens=max_tokens, system=system,
                    messages=[{"role": "user", "content": prompt}])
                text = "".join(b.text for b in resp.content if b.type == "text")
                self.usage["in"] += resp.usage.input_tokens
                self.usage["out"] += resp.usage.output_tokens
                self.usage["calls"] += 1
                open(cp, "w").write(text)
                return text
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

def _source_excerpt(path: str, max_chars: int = 3500) -> str:
    src = open(path, encoding="utf-8", errors="replace").read()
    return src if len(src) <= max_chars else src[:max_chars] + "\n# ... (kısaltıldı)"

def map_step(analysis: dict, llm: LLM, model: str, workers: int) -> None:
    mods = analysis["modules"]
    indeg, outdeg = analysis["in_degree"], analysis["out_degree"]
    label, system = analysis["unit_label"], analysis["map_system"]
    is_code = analysis["kind"] == "code"
    def build_prompt(info: ModuleInfo) -> str:
        head = (
            f"{label}: {info.name}\n"
            f"{analysis['weight_label']}: {info.loc} | bağımlı/referans (in): {indeg[info.name]} | "
            f"bağlı (out): {outdeg[info.name]}\n"
            f"Açıklama: {info.docstring or '(yok)'}\n"
        )
        if is_code:
            head += (
                f"Sınıflar: {', '.join(info.classes) or '(yok)'}\n"
                f"Fonksiyonlar: {', '.join(info.functions[:25]) or '(yok)'}\n"
                f"İç importlar: {', '.join(info.internal_imports) or '(yok)'}\n"
                f"Dış importlar: {', '.join(info.external_imports) or '(yok)'}\n"
            )
        else:
            head += (
                f"Tür/etiketler: {', '.join(info.classes) or '(yok)'}\n"
                f"Atıfta bulunduğu birimler: {', '.join(info.internal_imports) or '(yok)'}\n"
            )
        return head + f"\nİÇERİK (parça):\n```\n{_source_excerpt(info.path)}\n```"
    def run(info):
        info.summary = llm.call(system, build_prompt(info), model, max_tokens=400)
        return info.name
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run, m): m.name for m in mods.values()}
        for f in as_completed(futs):
            print(f"  [map] {f.result()} ✓", file=sys.stderr)

def reduce_step(analysis: dict, llm: LLM, model: str) -> str:
    mods, indeg, outdeg = analysis["modules"], analysis["in_degree"], analysis["out_degree"]
    pr, wl, label = analysis["pagerank"], analysis["weight_label"], analysis["unit_label"]
    rank_table = "\n".join(
        f"  {m} | in={indeg[m]} out={outdeg[m]} pr={pr[m]:.3f} {wl}={mods[m].loc}"
        for m in analysis["ranking"])
    edge_list = "\n".join(f"  {a} -> {b}" for a, b in analysis["edges"]) or "  (yok)"
    summaries = "\n\n".join(f"### {m}\n{mods[m].summary}" for m in analysis["ranking"])
    prompt = (
        f"REPO TÜRÜ: {analysis['kind']} ({analysis['lang']}) | "
        f"{label.upper()}: {len(mods)} | REFERANS/KENAR: {len(analysis['edges'])}\n\n"
        f"AĞIRLIK SIRALAMASI:\n{rank_table}\n\n"
        f"ÇAPRAZ REFERANSLAR:\n{edge_list}\n\n"
        f"{label.upper()} ÖZETLERİ:\n{summaries}\n\n"
        f"Yukarıdaki verilerle derin raporu üret."
    )
    return llm.call(analysis["reduce_system"], prompt, model, max_tokens=4096)


# ----------------------------------------------------------------------------
# 6. RENDER + CLI
# ----------------------------------------------------------------------------
def render(analysis: dict, report_body: str, target: str) -> str:
    mods, indeg = analysis["modules"], analysis["in_degree"]
    label, wl, kind = analysis["unit_label"], analysis["weight_label"], analysis["kind"]
    edge_word = "iç bağımlılık" if kind == "code" else "çapraz referans"
    header = [f"# Derin Repo Raporu — {target}", "",
              f"_Repo türü: **{kind}** ({analysis['lang']}) · {len(mods)} {label.lower()} · "
              f"{len(analysis['edges'])} {edge_word} · "
              f"{sum(m.loc for m in mods.values())} {wl}_", "",
              f"## Ağırlık / merkezilik (statik analiz)", "",
              f"| {label} | in | out | pagerank | {wl} |", "|---|---|---|---|---|"]
    for m in analysis["ranking"][:15]:
        header.append(f"| {m} | {indeg[m]} | {analysis['out_degree'][m]} | "
                      f"{analysis['pagerank'][m]:.3f} | {mods[m].loc} |")
    header += ["", "---", "", report_body]
    return "\n".join(header)

def main():
    ap = argparse.ArgumentParser(description="RepoMind — repo → derin mimari raporu")
    ap.add_argument("target", help="GitHub URL veya yerel yol")
    ap.add_argument("--type", default="auto", choices=["auto", "code", "content"],
                    help="Repo türü (varsayılan: otomatik tespit)")
    ap.add_argument("--lang", default="auto", choices=["auto", "python", "javascript"])
    ap.add_argument("--out", default="rapor.md")
    ap.add_argument("--map-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--reduce-model", default="claude-sonnet-4-6")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache", default=".repomind_cache")
    ap.add_argument("--dry-run", action="store_true", help="LLM çağrısı yapma; sadece statik + prompt")
    args = ap.parse_args()

    t0 = time.time()
    work = tempfile.mkdtemp(prefix="repomind_")
    try:
        print("→ ingest", file=sys.stderr)
        repo_root = ingest(args.target, work)

        # repo-tipi tespiti
        if args.type == "auto":
            kind, lang = classify_repo(repo_root)
        else:
            kind = args.type
            lang = (args.lang if args.lang != "auto" else "python")
        if args.lang != "auto":
            lang = args.lang
        print(f"→ repo-tipi: {kind}" + (f" ({lang})" if kind == "code" else ""), file=sys.stderr)

        print("→ statik analiz", file=sys.stderr)
        if kind == "content":
            analysis = content_analysis(repo_root)
        else:
            code_root = _find_code_root(repo_root, [e for e, l in LANG_EXT.items() if l == lang])
            analysis = static_analysis(code_root, lang)
        ul = analysis["unit_label"].lower()
        print(f"  {len(analysis['modules'])} {ul}, {len(analysis['edges'])} kenar/referans", file=sys.stderr)

        llm = LLM(args.cache, dry_run=args.dry_run)
        print(f"→ map ({len(analysis['modules'])} {ul}, model={args.map_model})", file=sys.stderr)
        map_step(analysis, llm, args.map_model, args.workers)
        print(f"→ reduce (model={args.reduce_model})", file=sys.stderr)
        body = reduce_step(analysis, llm, args.reduce_model)

        report = render(analysis, body, args.target)
        open(args.out, "w").write(report)

        # Statik veriyi de JSON dök
        dump = {m: {k: v for k, v in asdict(info).items() if k != "path"}
                for m, info in analysis["modules"].items()}
        json.dump(dump, open(os.path.splitext(args.out)[0] + ".json", "w"),
                  indent=2, ensure_ascii=False)

        dt = time.time() - t0
        u = llm.usage
        print(f"\n✓ {args.out} yazıldı ({dt:.1f}s)", file=sys.stderr)
        print(f"  LLM: {u['calls']} çağrı, {u['cached']} cache, "
              f"{u['in']} in-token, {u['out']} out-token", file=sys.stderr)
    finally:
        shutil.rmtree(work, ignore_errors=True)

if __name__ == "__main__":
    main()
