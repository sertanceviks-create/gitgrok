#!/usr/bin/env python3
"""
RepoMind Faz 3 — Doğrulama Motoru (deterministik, LLM'siz)
==========================================================
RepoMind'ın ürettiği analizi/planı GERÇEK kodla otomatik puanlar.
"Ürettiğimiz çıktı doğru mu?" sorusunu kanıta bağlayan asıl teknik hendek.

Üç bağımsız kontrol:
  1. plan_validity   : yeniden-inşa planı geçerli bir topolojik sıra mı?
                       (hiçbir modül, bağımlı olduğu modülden ÖNCE kurulmuyor mu)
  2. arch_fidelity   : bir aday repo, hedef mimariye ne kadar uyuyor?
                       (modül kapsamı recall + bağımlılık kenarı precision/recall -> F1)
  3. compile_check   : aday kod gerçekten derleniyor mu? (py_compile)

Toplam skor = ağırlıklı ortalama (0-100).

Kullanım:
  python verify.py --target /yol/requests --plan plan.txt          # sadece plan
  python verify.py --target /yol/requests --candidate /yol/aday     # mimari + derleme
  python verify.py --self /yol/requests                             # kendiyle (sağlık testi)
"""
from __future__ import annotations
import os, sys, argparse, py_compile, glob
import repomind as rm   # statik analizi yeniden kullan


# ----------------------------------------------------------------------------
def _analyze(root: str, lang: str = "python") -> dict:
    code_root = rm._find_code_root(root, [".py"])
    return rm.static_analysis(code_root, lang)

def _short(name: str) -> str:
    """requests.models -> models  (paket önekini at, son parçayı al)."""
    return name.split(".")[-1]


# 1. PLAN GEÇERLİLİĞİ ---------------------------------------------------------
def plan_validity(analysis: dict, plan: list[str]) -> dict:
    """Plan, gerçek bağımlılık grafiğinin geçerli bir topolojik sırası mı?
    Bir modül kurulduğunda, iç bağımlılıklarının hepsi ZATEN kurulmuş olmalı."""
    mods = analysis["modules"]
    # kısa-ad -> gerçek modül adı eşlemesi
    by_short = {}
    for full in mods:
        by_short.setdefault(_short(full), full)
    deps = {full: set(mods[full].internal_imports) for full in mods}

    installed, violations, unknown = set(), [], []
    for step in plan:
        full = by_short.get(_short(step))
        if full is None:
            unknown.append(step)
            continue
        missing = {d for d in deps[full] if d not in installed}
        if missing:
            violations.append((step, sorted(_short(m) for m in missing)))
        installed.add(full)

    covered = len([s for s in plan if by_short.get(_short(s)) in mods])
    coverage = covered / max(1, len(mods))
    valid = len(violations) == 0
    score = (1.0 if valid else max(0.0, 1 - len(violations) / max(1, len(plan)))) * \
            min(1.0, coverage) * 100
    return {"valid": valid, "violations": violations, "unknown": unknown,
            "coverage": round(coverage, 3), "score": round(score, 1)}


# 2. MİMARİ UYUMU -------------------------------------------------------------
def arch_fidelity(target: dict, cand: dict) -> dict:
    """Aday repo, hedef mimariye ne kadar sadık? Modül kapsamı + kenar F1."""
    t_mods = {_short(m) for m in target["modules"]}
    c_mods = {_short(m) for m in cand["modules"]}
    module_recall = len(t_mods & c_mods) / max(1, len(t_mods))

    t_edges = {(_short(a), _short(b)) for a, b in target["edges"]}
    c_edges = {(_short(a), _short(b)) for a, b in cand["edges"]}
    tp = len(t_edges & c_edges)
    precision = tp / max(1, len(c_edges))
    recall = tp / max(1, len(t_edges))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    score = (0.5 * module_recall + 0.5 * f1) * 100
    return {"module_recall": round(module_recall, 3),
            "edge_precision": round(precision, 3), "edge_recall": round(recall, 3),
            "edge_f1": round(f1, 3),
            "missing_modules": sorted(t_mods - c_mods),
            "extra_modules": sorted(c_mods - t_mods),
            "score": round(score, 1)}


# 3. DERLEME KONTROLÜ ---------------------------------------------------------
def compile_check(root: str) -> dict:
    code_root = rm._find_code_root(root, [".py"])
    files = [f for f in glob.glob(os.path.join(code_root, "**/*.py"), recursive=True)
             if "__pycache__" not in f]
    ok, fails = 0, []
    for f in files:
        try:
            py_compile.compile(f, doraise=True)
            ok += 1
        except py_compile.PyCompileError as e:
            fails.append(os.path.basename(f))
    score = ok / max(1, len(files)) * 100
    return {"compiled": ok, "total": len(files), "failures": fails,
            "score": round(score, 1)}


# AGGREGATE -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="RepoMind doğrulama motoru")
    ap.add_argument("--target", help="Hedef (gerçek) repo yolu")
    ap.add_argument("--candidate", help="Aday/yeniden-üretilmiş repo yolu")
    ap.add_argument("--plan", help="Yeniden-inşa planı dosyası (satır başına 1 modül)")
    ap.add_argument("--self", dest="self_root", help="Kendiyle karşılaştır (sağlık testi)")
    args = ap.parse_args()

    if args.self_root:
        args.target = args.candidate = args.self_root

    results, weights = {}, {}
    target = _analyze(args.target) if args.target else None

    if args.plan and target:
        plan = [l.strip() for l in open(args.plan) if l.strip() and not l.startswith("#")]
        results["plan_validity"] = plan_validity(target, plan)
        weights["plan_validity"] = 0.3

    if args.candidate and target:
        cand = _analyze(args.candidate)
        results["arch_fidelity"] = arch_fidelity(target, cand)
        weights["arch_fidelity"] = 0.5
        results["compile_check"] = compile_check(args.candidate)
        weights["compile_check"] = 0.2

    # rapor
    print("=" * 56)
    print("REPOMIND DOĞRULAMA RAPORU")
    print("=" * 56)
    total, wsum = 0.0, 0.0
    for k, v in results.items():
        print(f"\n[{k}]  skor: {v['score']}")
        for kk, vv in v.items():
            if kk == "score":
                continue
            if isinstance(vv, list) and len(vv) > 8:
                vv = vv[:8] + ["..."]
            print(f"    {kk}: {vv}")
        total += v["score"] * weights[k]
        wsum += weights[k]
    if wsum:
        print("\n" + "-" * 56)
        print(f"TOPLAM SKOR: {round(total / wsum, 1)} / 100")
        print("-" * 56)

if __name__ == "__main__":
    main()
