"""Generate the paper figures and the web SVGs from the recorded numbers.

Most numbers are transcribed from results/summary.md and the per-run logs under
results/runs/; the injection panel reads results/control_causal_3b.json.
Keeping them here in one place makes it easy to audit the figures against
those files.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "figures"
WEB = ROOT / "docs" / "assets"

NAVY = "#1d3f66"
TEAL = "#0f766e"
ORANGE = "#c2410c"
GREY = "#6b7280"
LIGHT = "#e9edf2"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#4b5563",
    "axes.linewidth": 0.8,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def save(fig, name):
    PAPER.mkdir(parents=True, exist_ok=True)
    WEB.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER / f"{name}.pdf")
    fig.savefig(WEB / f"{name}.svg")
    fig.savefig(WEB / f"{name}.png", dpi=160)
    plt.close(fig)


def fig_signals():
    conds = ["no hint", "correct hint", "conflicting hint"]
    x = [0, 1, 2]
    frozen = [0.995, 0.995, 0.957]
    refit = [0.995, 1.000, 0.997]
    punk = [0.993, 0.969, 0.834]
    maxp = [0.615, 0.916, 0.463]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.axhline(0.5, color=GREY, lw=0.7, ls=(0, (2, 3)))
    ax.plot(x, refit, ":", color=NAVY, lw=1.3, label="probe, refit on condition")
    ax.plot(x, frozen, "-o", color=NAVY, lw=1.8, ms=4.5,
            label="probe, frozen from no hint")
    ax.plot(x, punk, "-o", color=TEAL, lw=1.8, ms=4.5, label="p(unknown)")
    ax.plot(x, maxp, "-o", color=ORANGE, lw=1.8, ms=4.5, label="max probability")
    for yy, color in ((frozen, NAVY), (punk, TEAL), (maxp, ORANGE)):
        ax.annotate(f"{yy[-1]:.3f}", (2, yy[-1]), xytext=(6, 0),
                    textcoords="offset points", color=color, fontsize=8, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_xlim(-0.15, 2.45)
    ax.set_ylim(0.4, 1.04)
    ax.set_ylabel("AUROC: evidence present vs. absent")
    ax.legend(loc="lower left", frameon=False, handlelength=1.6)
    save(fig, "fig_signals")


def fig_depth():
    import json

    cc = json.load(open(ROOT / "results" / "control_causal_3b.json"))
    st = cc["steer"]["diff_vs_base"]
    models = [
        ("Qwen2.5-VL-3B", [66.7, 88.9], [22, 80], NAVY, "o"),
        ("Qwen2.5-VL-7B", [64.3, 85.7, 92.9], [0, 69, 90], TEAL, "s"),
        ("LLaVA-OneVision-7B", [64.3, 85.7, 92.9], [36, 80, 96], ORANGE, "^"),
        ("Gemma-3-4B", [23.5, 47.1, 70.6, 88.2], [1, 12, 90, 101], "#7c3aed", "D"),
        ("Phi-3.5-vision", [43.8, 68.8, 87.5, 93.8], [31, 97, 102, 101], "#a16207", "v"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    ax = axes[0]
    for name, xs, ys, color, marker in models:
        ax.plot(xs, ys, "-", color=color, lw=1.6, marker=marker, ms=4.5, label=name)
    ax.set_xlim(18, 101)
    ax.set_ylim(-6, 106)
    ax.set_xlabel("relative depth of the patched layer (%)")
    ax.set_ylabel("refusal recovered (%)")
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.legend(loc="upper left", frameon=False, handlelength=1.3, fontsize=7.2,
              borderaxespad=0.2, labelspacing=0.3)
    ax.grid(axis="y", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title("(a) no-hint state pasted back", loc="left", fontsize=8.5)

    ax = axes[1]
    depths = [8 / 36 * 100, 16 / 36 * 100, 24 / 36 * 100, 32 / 36 * 100, 100]
    series = [
        ("$-0.25\\,w$ (close gate)", [f"L{L}-0.25" for L in (8, 16, 24, 32)] + ["w-0.25"],
         NAVY, "o"),
        ("$+0.25\\,w$ (open gate)", [f"L{L}+0.25" for L in (8, 16, 24, 32)] + ["w+0.25"],
         ORANGE, "s"),
    ]
    for name, keys, color, marker in series:
        ys = [st[k]["absent_refusal"]["diff"] for k in keys]
        cis = [st[k]["absent_refusal"]["ci"] for k in keys]
        err = [[y - c[0] for y, c in zip(ys, cis)], [c[1] - y for y, c in zip(ys, cis)]]
        ax.errorbar(depths, ys, yerr=err, color=color, lw=1.6, marker=marker,
                    ms=4.5, capsize=2, elinewidth=0.8, label=name)
    ax.axhline(0, color=GREY, lw=0.7, ls=(0, (2, 3)))
    ax.set_xlim(15, 106)
    ax.set_ylim(-0.56, 0.42)
    ax.set_xticks([22, 44, 67, 89, 100])
    ax.set_xticklabels(["22", "44", "67", "89", "head"])
    ax.set_xlabel("relative depth of the injected layer (%)")
    ax.set_ylabel("change in absent-object refusal")
    ax.legend(loc="lower left", frameon=False, handlelength=1.6)
    ax.grid(axis="y", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title("(b) probe direction injected", loc="left", fontsize=8.5)

    fig.tight_layout(w_pad=1.4)
    save(fig, "fig_depth")


def fig_risk():
    cov = [0.9, 0.7, 0.5]
    series = [
        ("probe, frozen", [0.801, 0.398, 0.069], NAVY, "o", "-"),
        ("probe, refit", [0.801, 0.398, 0.011], NAVY, "o", ":"),
        ("p(unknown)", [0.801, 0.456, 0.249], TEAL, "s", "-"),
        ("max probability", [0.985, 0.762, 0.510], ORANGE, "^", "-"),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for name, ys, color, marker, ls in series:
        ax.plot(cov, ys, ls, color=color, lw=1.6, marker=marker, ms=4.5, label=name)
    ax.set_ylim(-0.02, 1.02)
    for name, ys, color, marker, ls in series:
        ax.annotate(f"{ys[-1]:.3f}", (0.5, ys[-1]), xytext=(6, 0),
                    textcoords="offset points", color=color, fontsize=7.4, va="center")
    ax.set_xlim(0.95, 0.40)
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("fabrication rate")
    ax.set_xticks([0.9, 0.7, 0.5])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, handlelength=1.6)
    save(fig, "fig_risk")


def fig_design():
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    cells = [
        (3.1, "no hint", "0.011", "0.904", "good", "refusal", "refusal"),
        (5.9, "correct", "0.004", "0.521", "good", "refusal", "hint a"),
        (8.7, "conflicting", "0.088", "0.563", "bad", "refusal", "hint b"),
    ]
    for cx, title, present, absent, tone, ptag, atag in cells:
        color = "#1f7a4d" if tone == "good" else "#b42318"
        ax.text(cx, 5.55, title, ha="center", va="center", fontsize=7.2, color="#4b5563")
        for yy, val, tag in ((4.45, present, ptag), (2.45, absent, atag)):
            ax.add_patch(FancyBboxPatch((cx - 0.98, yy - 0.55), 1.96, 1.05,
                                        boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc="#f7f9fb", ec="#d7dde5", lw=0.6))
            ax.text(cx, yy + 0.10, val, ha="center", va="center", fontsize=10.5,
                    color=color, family="serif")
            ax.text(cx, yy - 0.32, tag, ha="center", va="center", fontsize=5.9,
                    color="#6b7280")
    ax.text(0.95, 4.45, "object\npresent", ha="center", va="center", fontsize=7.0,
            color="#4b5563", linespacing=1.3)
    ax.text(0.95, 2.45, "object\nabsent", ha="center", va="center", fontsize=7.0,
            color="#4b5563", linespacing=1.3)
    ax.text(5.6, 0.75, "one image · six prompts · no pixel edited",
            ha="center", va="center", fontsize=7.5, style="italic", color="#4b5563")
    save(fig, "fig_design")


if __name__ == "__main__":
    fig_design()
    fig_signals()
    fig_depth()
    fig_risk()
    print("figures ->", PAPER, "and", WEB)
