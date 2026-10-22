"""Generate the paper's figures and number macros from results/*.csv.

Every number in tilecdc.tex comes from a macro defined in numbers.tex, which
this script writes. Run after bench/run_all.sh.
"""
import csv
import math
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.dirname(__file__)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
GREY, BLUE, LIGHT = "#8a8f98", "#1f5fbf", "#c9ccd1"


def rows(name):
    with open(os.path.join(R, name)) as f:
        return [r for r in csv.reader(f) if r and not any(c.startswith("#") for c in r[:3])]


macros = {}


def m(name, value):
    macros[name] = value


def num(x, d=2):
    return f"{x:.{d}f}"


# ---------------- throughput: independent runs, median and range ----------------
runs, stats = {}, {}
for r in rows("compare_runs.csv"):
    ds, meth = r[1], r[2]
    runs.setdefault((ds, meth), []).append(float(r[4]))
    stats[(ds, meth)] = dict(chunks=int(r[5]), avg=float(r[6]), sd=float(r[7]), p5=float(r[8]), p95=float(r[9]), dedup=float(r[10]))
for r in rows("compare_rust_runs.csv"):
    ds = r[1]
    runs.setdefault((ds, "fastcdc-rs"), []).append(float(r[4]))
    stats[(ds, "fastcdc-rs")] = dict(chunks=int(r[5]), avg=float(r[6]), dedup=float(r[7]))
med = {k: statistics.median(v) for k, v in runs.items()}
lo = {k: min(v) for k, v in runs.items()}
hi = {k: max(v) for k, v in runs.items()}
m("nRuns", str(min(len(v) for v in runs.values())))

DS = {"SRC": "Src", "BIN": "Bin", "RAND": "Rand"}
for ds, tag in DS.items():
    for meth, mt in [("tilecdc-amx", "amx"), ("tilecdc-avx512", "avx"), ("fastcdc-c", "fc"), ("fastcdc-rs", "rs"), ("tilecdc-scalar", "scalar")]:
        m(f"{mt}{tag}", num(med[(ds, meth)]))
        m(f"{mt}{tag}Lo", num(lo[(ds, meth)])); m(f"{mt}{tag}Hi", num(hi[(ds, meth)]))
    best = max(med[(ds, "fastcdc-c")], med[(ds, "fastcdc-rs")])
    m(f"spAmx{tag}", num(med[(ds, "tilecdc-amx")] / best, 1))
    m(f"spAvx{tag}", num(med[(ds, "tilecdc-avx512")] / best, 2))
amx_all = [med[(d, "tilecdc-amx")] for d in DS]
fc_all = [med[(d, k)] for d in DS for k in ("fastcdc-c", "fastcdc-rs")]
avx_all = [med[(d, "tilecdc-avx512")] for d in DS]
sp = [med[(d, "tilecdc-amx")] / max(med[(d, "fastcdc-c")], med[(d, "fastcdc-rs")]) for d in DS]
spa = [med[(d, "tilecdc-avx512")] / max(med[(d, "fastcdc-c")], med[(d, "fastcdc-rs")]) for d in DS]
m("amxMin", num(min(amx_all), 1)); m("amxMax", num(max(amx_all), 1))
m("fcMin", num(min(fc_all), 1)); m("fcMax", num(max(fc_all), 1))
m("avxMin", num(min(avx_all), 1)); m("avxMax", num(max(avx_all), 1))
m("spMin", num(min(sp), 1)); m("spMax", num(max(sp), 1))
m("spAvxMin", num(min(spa), 2)); m("spAvxMax", num(max(spa), 2))
spread = [100 * (hi[k] / lo[k] - 1) for k in runs if k[1] != "tilecdc-scalar"]
m("runSpreadMax", f"{max(spread):.0f}")
# stage-time reduction for chunk + fingerprint (1 and 3 GB/s fingerprints)
fcb, amxb = min(fc_all), min(amx_all)
red = [100 * (1 - (1 / amxb + 1 / f) / (1 / fcb + 1 / f)) for f in (1.0, 3.0)]
m("stageRedLo", f"{min(red):.0f}"); m("stageRedHi", f"{max(red):.0f}")

fig, ax = plt.subplots(figsize=(3.4, 2.2))
methods = [("FastCDC (C)", "fastcdc-c", LIGHT, None), ("fastcdc-rs", "fastcdc-rs", LIGHT, "//"),
           ("TileCDC AVX-512", "tilecdc-avx512", GREY, None), ("TileCDC AMX", "tilecdc-amx", BLUE, None)]
w = 0.2
for i, (lab, key, col, hatch) in enumerate(methods):
    xs = [j + (i - 1.5) * w for j in range(3)]
    vals = [med[(d, key)] for d in DS]
    errs = [[med[(d, key)] - lo[(d, key)] for d in DS], [hi[(d, key)] - med[(d, key)] for d in DS]]
    ax.bar(xs, vals, w, label=lab, color=col, edgecolor="white", hatch=hatch, yerr=errs, error_kw=dict(lw=0.6, capsize=1.2, ecolor="#444"))
    for x, v, e in zip(xs, vals, errs[1]):
        ax.text(x, v + e + 0.08, f"{v:.1f}", ha="center", fontsize=6)
ax.set_xticks(range(3)); ax.set_xticklabels(["SRC", "BIN", "RAND"])
ax.set_ylabel("GB/s, one core")
ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.24))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_throughput.pdf")); plt.close(fig)

# ---------------- dedup vs mean chunk size ----------------
sw, amx_sw = {}, []
for r in rows("sweep.csv"):
    sw.setdefault((r[0], r[2]), []).append((float(r[4]), float(r[6])))
    if r[2] == "tilecdc-amx":
        amx_sw.append(float(r[3]))
swr = {}
for r in rows("sweep_rust.csv"):
    swr.setdefault(r[0], []).append((float(r[6]), float(r[7])))


def interp(curve, x):
    c = sorted(curve)
    for (x0, y0), (x1, y1) in zip(c, c[1:]):
        if x0 <= x <= x1:
            t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + t * (y1 - y0)
    return None


fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2))
diffs, xs_used = [], []
for ax, ds in zip(axes, ["SRC", "BIN"]):
    for lab, curve, col, mk in [("FastCDC (C)", sw[(ds, "fastcdc-c")], GREY, "s"), ("fastcdc-rs", swr[ds], LIGHT, "^"), ("TileCDC", sw[(ds, "tilecdc-amx")], BLUE, "o")]:
        c = sorted(curve)
        ax.plot([x for x, _ in c], [y for _, y in c], marker=mk, ms=3.5, lw=1.4, color=col, label=lab)
    for x, y in sw[(ds, "tilecdc-amx")]:
        for ref in (sw[(ds, "fastcdc-c")], swr[ds]):
            v = interp(ref, x)
            if v:
                diffs.append(100 * (y / v - 1)); xs_used.append(x)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("mean chunk size (KiB)"); ax.set_title(ds, fontsize=9)
    ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels(["4", "8", "16", "32", "64"])
axes[0].set_ylabel("dedup ratio")
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_dedup.pdf")); plt.close(fig)
m("dedupDiffMin", f"{min(diffs):+.1f}"); m("dedupDiffMax", f"{max(diffs):+.1f}")
m("dedupDiffMean", f"{sum(diffs) / len(diffs):+.2f}"); m("dedupPairs", str(len(diffs)))
m("dedupNeg", str(sum(1 for d in diffs if d < 0)))
m("dedupXMin", f"{min(xs_used):.1f}"); m("dedupXMax", f"{max(xs_used):.1f}")
m("amxSweepMin", num(min(amx_sw), 1)); m("amxSweepMax", num(max(amx_sw), 1))

ns = list(csv.DictReader(open(os.path.join(R, "normsweep.csv"))))
d2 = [100 * (float(r["dedup"]) / interp(sw[(r["dataset"], "fastcdc-c")], float(r["avg_kib"])) - 1)
      for r in ns if r["norm"] == "2" and interp(sw[(r["dataset"], "fastcdc-c")], float(r["avg_kib"]))]
m("normTwoWorst", f"{min(d2):.1f}")

for ds, tag in [("SRC", "Src"), ("BIN", "Bin")]:
    for meth, mt in [("fastcdc-c", "Fc"), ("tilecdc-amx", "Tc")]:
        c = stats[(ds, meth)]
        for k, kn in (("avg", "avg"), ("sd", "sd"), ("p5", "pFive"), ("p95", "pNinetyFive")):
            m(f"{kn}{mt}{tag}", num(c[k]))
        m(f"dedup{mt}{tag}", num(c["dedup"], 3))
    m(f"avgRs{tag}", num(stats[(ds, "fastcdc-rs")]["avg"])); m(f"dedupRs{tag}", num(stats[(ds, "fastcdc-rs")]["dedup"], 3))

# ---------------- edits ----------------
erows, means = [], {}
for line in open(os.path.join(R, "edits.csv")):
    file, _, rest = line.strip().partition(",")
    if rest.startswith("# mean chunk"):
        means[file] = re.findall(r"([0-9]+(?:\.[0-9]+)?)", rest)
    elif rest and not rest.startswith("#"):
        k, n, a, b, c = rest.split(",")
        erows.append((file, k, int(n), float(a), float(b), float(c)))
label = {"numpy-src": "NumPy src", "scipy-bin": "SciPy bin"}
m("editrows", "\n".join(f"{label[f]} & {k} & {n} & {a:.2f} & {b:.2f} & {c:.2f} \\\\" for f, k, n, a, b, c in erows))
gm = [c - a for f, k, n, a, b, c in erows]
m("editMatchedMin", f"{min(gm):+.2f}"); m("editMatchedMax", f"{max(gm):+.2f}")
gd = [b - a for f, k, n, a, b, c in erows]
m("editDefaultMin", f"{min(gd):+.2f}"); m("editDefaultMax", f"{max(gd):+.2f}")
for f, tag in [("numpy-src", "Np"), ("scipy-bin", "Sp")]:
    v = means[f]   # fastcdc mean, tilecdc default mean, avg_size, tilecdc matched mean
    m(f"editMeanFc{tag}", v[0]); m(f"editMeanTd{tag}", v[1]); m(f"editAvgTm{tag}", v[2]); m(f"editMeanTm{tag}", v[3])

# ---------------- breakdown and threads ----------------
br = {r[0]: (float(r[1]), float(r[2])) for r in rows("breakdown.csv")}
m("brSbox", f"{br['sbox_avx512'][0]:.3f}"); m("brSboxG", f"{br['sbox_avx512'][1]:.0f}")
m("brAmx", f"{br['hash+mix+test_amx'][0]:.3f}"); m("brAmxG", f"{br['hash+mix+test_amx'][1]:.1f}")
m("brAvx", f"{br['hash+mix+test_avx512'][0]:.3f}"); m("brAvxG", f"{br['hash+mix+test_avx512'][1]:.1f}")
m("brSc", f"{br['hash+mix+test_scalar'][0]:.2f}"); m("brScG", f"{br['hash+mix+test_scalar'][1]:.2f}")
tm = br["amx_loads+tdpbusd+1store"][0]; ts = br["amx_loads+tdpbusd+stores"][0]; ta = br["hash+mix+test_amx"][0]
m("brTmul", f"{tm:.3f}"); m("brStore", f"{ts - tm:.3f}"); m("brMix", f"{ta - ts:.3f}"); m("brMixPct", f"{100 * (ta - ts) / ta:.0f}")
ks = ta + br["sbox_avx512"][0]
m("brKernelPlusSbox", f"{ks:.3f}"); m("brKernelPlusSboxG", f"{1 / ks:.1f}")
m("brAmxCyc", f"{ta * 2.1:.2f}"); m("brAvxCyc", f"{br['hash+mix+test_avx512'][0] * 2.1:.2f}")
th = {(r[0], r[1]): float(r[2]) for r in rows("threads.csv")}
m("thFcOne", num(th[("fastcdc-c", "1")])); m("thFcTwo", num(th[("fastcdc-c", "2")]))
m("thAmxOne", num(th[("tilecdc-amx", "1")])); m("thAmxTwo", num(th[("tilecdc-amx", "2")]))

# ---------------- seed variance ----------------
sv = {}
for r in csv.DictReader(open(os.path.join(R, "seedvar.csv"))):
    sv.setdefault(int(r["avg_target_kib"]), []).append(float(r["mean_kib"]))
for k, tag in [(8, "Eight"), (16, "Sixteen"), (32, "ThirtyTwo"), (64, "SixtyFour")]:
    m(f"seed{tag}Lo", f"{min(sv[k]):.1f}"); m(f"seed{tag}Hi", f"{max(sv[k]):.1f}")

# ---------------- machine and tests ----------------
mt = open(os.path.join(R, "machine.txt")).read()
mem = re.search(r"Mem:\s+(\d+)", mt)
m("memGB", mem.group(1) if mem else "?")
last = [l for l in open(os.path.join(R, "tests.txt")) if "ALL TESTS PASSED" in l][0]
m("testComparisons", f"{int(last.split('(')[1].split()[0]):,}")

with open(os.path.join(OUT, "numbers.tex"), "w") as fo:
    fo.write("% generated by gen.py from results/*.csv; do not edit\n")
    for k, v in sorted(macros.items()):
        fo.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"{len(macros)} macros; speedup {macros['spMin']}-{macros['spMax']}x; dedup {macros['dedupDiffMin']}..{macros['dedupDiffMax']}% "
      f"mean {macros['dedupDiffMean']}% ({macros['dedupNeg']}/{macros['dedupPairs']} below); stage reduction {macros['stageRedLo']}-{macros['stageRedHi']}%")
