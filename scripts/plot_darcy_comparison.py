"""
Plot PyTorch vs JAX training comparison for Darcy Flow FNO (300 epochs).
Metrics: train loss, train error, eval 16_h1/l2, eval 32_h1/l2, epoch time.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── Parse logs ────────────────────────────────────────────────────────────────

def parse_log(path):
    epochs = {}
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r'\[(\d+)\] time=([\d.]+), avg_loss=([\d.]+), train_err=([\d.]+)', lines[i])
        if m:
            ep = int(m.group(1))
            d = {
                "time": float(m.group(2)),
                "loss": float(m.group(3)),
                "err":  float(m.group(4)),
            }
            if i + 1 < len(lines):
                e = re.match(
                    r'Eval: 16_h1=([\d.]+), 16_l2=([\d.]+), 32_h1=([\d.]+), 32_l2=([\d.]+)',
                    lines[i + 1],
                )
                if e:
                    d["16_h1"] = float(e.group(1))
                    d["16_l2"] = float(e.group(2))
                    d["32_h1"] = float(e.group(3))
                    d["32_l2"] = float(e.group(4))
                    i += 1
            epochs[ep] = d
        i += 1
    return epochs

HERE = Path(__file__).parent
pt_raw  = parse_log(HERE.parent / "training_log_darcy.txt")
jax_raw = parse_log("/tmp/training_log_darcy_jax_300.txt")

def arrays(log, key):
    items = sorted((ep, v[key]) for ep, v in log.items() if key in v)
    return np.array([x[0] for x in items]), np.array([x[1] for x in items])

def smooth(y, w=9):
    """Centered rolling mean with edge padding."""
    pad = w // 2
    y_pad = np.pad(y, pad, mode="edge")
    return np.convolve(y_pad, np.ones(w) / w, mode="valid")[:len(y)]

# ── Styling ───────────────────────────────────────────────────────────────────

PT_COLOR  = "#EE4C2C"   # PyTorch torch-orange
JAX_COLOR = "#1A73E8"   # JAX Google-blue
PT_LIGHT  = "#FBBCAA"
JAX_LIGHT = "#A8CEFC"

LR_DECAY_EPOCHS = [60, 120, 180, 240]
LR_COLOR        = "#9CA3AF"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.30,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.55,
    "axes.labelsize":    10,
    "axes.titlesize":    11.5,
    "axes.titleweight":  "bold",
    "axes.titlepad":     8,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   9,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#DDDDDD",
    "figure.facecolor":  "#F7F8FC",
    "axes.facecolor":    "#FFFFFF",
    "axes.edgecolor":    "#CCCCCC",
    "axes.linewidth":    0.8,
})

# ── Layout ────────────────────────────────────────────────────────────────────
#   Row 0 (tall):   Train Loss │ Train Error │ Epoch Time
#   Row 1 (medium): 16_h1 eval │ 16_l2 eval  │ 32_h1 eval │ 32_l2 eval

fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor("#FAFAFA")

gs = gridspec.GridSpec(
    2, 4,
    figure=fig,
    hspace=0.42,
    wspace=0.32,
    left=0.055, right=0.97,
    top=0.90,   bottom=0.08,
)

ax_loss = fig.add_subplot(gs[0, 0])
ax_err  = fig.add_subplot(gs[0, 1])
ax_time = fig.add_subplot(gs[0, 2])
ax_spd  = fig.add_subplot(gs[0, 3])  # speed-up bar

ax_16h1 = fig.add_subplot(gs[1, 0])
ax_16l2 = fig.add_subplot(gs[1, 1])
ax_32h1 = fig.add_subplot(gs[1, 2])
ax_32l2 = fig.add_subplot(gs[1, 3])


def add_lr_lines(ax):
    for i, ep in enumerate(LR_DECAY_EPOCHS):
        ax.axvline(ep, color=LR_COLOR, lw=1.1, ls=(0, (4, 3)), zorder=0, alpha=0.7)


def plot_metric(ax, key, title, ylabel, log_y=False, ylim=None, smooth_w=13):
    ep_pt,  y_pt  = arrays(pt_raw,  key)
    ep_jax, y_jax = arrays(jax_raw, key)

    # raw (very transparent background trace)
    ax.plot(ep_pt,  y_pt,  color=PT_COLOR,  alpha=0.15, lw=0.9)
    ax.plot(ep_jax, y_jax, color=JAX_COLOR, alpha=0.15, lw=0.9)

    # smoothed foreground
    sm_pt  = smooth(y_pt,  smooth_w)
    sm_jax = smooth(y_jax, smooth_w)
    ax.plot(ep_pt,  sm_pt,  color=PT_COLOR,  lw=2.2, label="PyTorch", zorder=3)
    ax.plot(ep_jax, sm_jax, color=JAX_COLOR, lw=2.2, label="JAX",     zorder=3)

    # shaded band ±1 raw std (rolling)
    def rolling_std(y, w=smooth_w):
        pad = w // 2
        yp = np.pad(y, pad, mode="edge")
        return np.array([yp[i:i+w].std() for i in range(len(y))])

    std_pt  = rolling_std(y_pt)
    std_jax = rolling_std(y_jax)
    ax.fill_between(ep_pt,  sm_pt  - std_pt,  sm_pt  + std_pt,  color=PT_COLOR,  alpha=0.10, zorder=1)
    ax.fill_between(ep_jax, sm_jax - std_jax, sm_jax + std_jax, color=JAX_COLOR, alpha=0.10, zorder=1)

    add_lr_lines(ax)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(0, 299)
    ax.legend(loc="upper right", frameon=True)


# ── 1. Train Loss ─────────────────────────────────────────────────────────────
plot_metric(ax_loss, "loss", "Train Loss (H1)", "H1 Loss", log_y=True)

# ── 2. Train Error ────────────────────────────────────────────────────────────
plot_metric(ax_err, "err", "Train Error", "Relative Error", log_y=True)

# ── 3. Epoch Time ─────────────────────────────────────────────────────────────
ep_pt_t,  t_pt  = arrays(pt_raw,  "time")
ep_jax_t, t_jax = arrays(jax_raw, "time")

# exclude epoch 0 (JIT warmup)
mask_pt  = ep_pt_t  > 0
mask_jax = ep_jax_t > 0

ax_time.plot(ep_pt_t[mask_pt],   t_pt[mask_pt],   color=PT_COLOR,  alpha=0.15, lw=0.8)
ax_time.plot(ep_jax_t[mask_jax], t_jax[mask_jax], color=JAX_COLOR, alpha=0.15, lw=0.8)
ax_time.plot(ep_pt_t[mask_pt],   smooth(t_pt[mask_pt],  13), color=PT_COLOR,  lw=2.2,
             label=f"PyTorch  avg {t_pt[mask_pt].mean():.2f}s")
ax_time.plot(ep_jax_t[mask_jax], smooth(t_jax[mask_jax], 13), color=JAX_COLOR, lw=2.2,
             label=f"JAX  avg {t_jax[mask_jax].mean():.2f}s")

# Annotate JAX LR-decay recompile spikes
for ep in LR_DECAY_EPOCHS:
    if ep in jax_raw:
        t = jax_raw[ep]["time"]
        if t > 3.0:
            ax_time.scatter([ep], [t], color=JAX_COLOR, s=40, zorder=5)
            ax_time.annotate(
                f"  LR÷2 ({t:.1f}s)",
                xy=(ep, t), xytext=(ep + 5, t - 2.5),
                fontsize=7.5, color=JAX_COLOR,
                arrowprops=dict(arrowstyle="-", color=JAX_COLOR, lw=0.8),
                va="top",
            )

add_lr_lines(ax_time)
ax_time.set_title("Epoch Wall Time  (excl. ep 0 JIT warmup)")
ax_time.set_xlabel("Epoch")
ax_time.set_ylabel("Seconds / Epoch")
ax_time.set_xlim(0, 299)
ax_time.set_ylim(0, None)
ax_time.legend(loc="upper left", frameon=True)

# ── 4. Speed-up bar chart ─────────────────────────────────────────────────────
pt_avg  = t_pt[mask_pt].mean()
jax_avg = t_jax[mask_jax].mean()
speedup = pt_avg / jax_avg

bars = ax_spd.bar(
    ["PyTorch", "JAX"],
    [pt_avg, jax_avg],
    color=[PT_COLOR, JAX_COLOR],
    width=0.5,
    edgecolor="white",
    linewidth=1.5,
    zorder=3,
)
# value labels on top of bars
for bar, val in zip(bars, [pt_avg, jax_avg]):
    ax_spd.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.03,
        f"{val:.2f}s",
        ha="center", va="bottom", fontsize=10.5, fontweight="bold",
        color=bar.get_facecolor(),
    )

# bracket + speedup annotation
y_bracket = pt_avg * 1.10
ax_spd.annotate(
    "",
    xy=(1, y_bracket), xytext=(0, y_bracket),
    arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.8),
)
ax_spd.text(
    0.5, y_bracket + 0.05,
    f"{speedup:.2f}× faster",
    ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1A1A2E",
)

ax_spd.set_title("Avg Epoch Time\n(after JIT warmup)")
ax_spd.set_ylabel("Seconds / Epoch")
ax_spd.set_ylim(0, pt_avg * 1.40)
ax_spd.grid(axis="x", alpha=0)
ax_spd.spines["left"].set_visible(True)
ax_spd.spines["bottom"].set_visible(True)

# ── 5–8. Eval metrics ────────────────────────────────────────────────────────
plot_metric(ax_16h1, "16_h1", "Eval 16×16 — H1 Error",  "H1 Error", ylim=(0.17, 0.32))
plot_metric(ax_16l2, "16_l2", "Eval 16×16 — L2 Error",  "L2 Error", ylim=(0.10, 0.27))
plot_metric(ax_32h1, "32_h1", "Eval 32×32 — H1 Error",  "H1 Error", ylim=(0.45, 0.70))
plot_metric(ax_32l2, "32_l2", "Eval 32×32 — L2 Error",  "L2 Error", ylim=(0.14, 0.31))

# ── LR decay tiny labels (axis-fraction coords so they never clip) ────────────
for ax in [ax_loss, ax_err, ax_16h1, ax_16l2, ax_32h1, ax_32l2]:
    for ep in LR_DECAY_EPOCHS:
        ax.text(ep, 1.01, "÷2", fontsize=6.5, color=LR_COLOR,
                va="bottom", ha="center",
                transform=ax.get_xaxis_transform(), alpha=0.75, clip_on=False)

# ── Figure title & legend ─────────────────────────────────────────────────────
fig.suptitle(
    "PyTorch vs JAX · FNO Darcy Flow · 300 Epochs · 1000 Training Samples",
    fontsize=14, fontweight="bold", y=0.96, color="#1A1A2E",
)

custom_lines = [
    Line2D([0], [0], color=PT_COLOR,  lw=2.5, label="PyTorch"),
    Line2D([0], [0], color=JAX_COLOR, lw=2.5, label="JAX"),
    Line2D([0], [0], color=LR_COLOR,  lw=1.2, ls=":", label="LR decay (÷2)"),
]
fig.legend(
    handles=custom_lines,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.935),
    frameon=True,
    fontsize=10,
    framealpha=0.95,
    edgecolor="#CCCCCC",
)

# ── Footer: final-epoch stats ─────────────────────────────────────────────────
bp  = min((v["16_h1"], k) for k, v in pt_raw.items()  if "16_h1" in v)
bj  = min((v["16_h1"], k) for k, v in jax_raw.items() if "16_h1" in v)
p299 = pt_raw[299]; j299 = jax_raw[299]
footer = (
    f"Final epoch [299]  ▸  "
    f"Train loss: PT {p299['loss']:.4f} | JAX {j299['loss']:.4f}  ·  "
    f"16_h1: PT {p299['16_h1']:.4f} | JAX {j299['16_h1']:.4f}  ·  "
    f"16_l2: PT {p299['16_l2']:.4f} | JAX {j299['16_l2']:.4f}  ·  "
    f"32_h1: PT {p299['32_h1']:.4f} | JAX {j299['32_h1']:.4f}  ·  "
    f"32_l2: PT {p299['32_l2']:.4f} | JAX {j299['32_l2']:.4f}  ·  "
    f"Best 16_h1: PT {bp[0]:.4f} @ep{bp[1]} | JAX {bj[0]:.4f} @ep{bj[1]}"
)
fig.text(
    0.5, 0.01, footer,
    ha="center", va="bottom", fontsize=8, color="#555566",
    style="italic",
)

# ── Save ─────────────────────────────────────────────────────────────────────
out = HERE / "darcy_pt_jax_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
