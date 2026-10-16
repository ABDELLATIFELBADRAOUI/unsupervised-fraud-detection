# Figure 1 -- data access by pipeline stage (ARRAY-D-26-04414R1).
# Rows: pipeline stages. Columns: the chronological training, validation and test blocks.
# A shaded cell is a block the stage accesses; the symbols say how (key at the bottom).
# Usage: python scripts/figure1_data_access.py [output_folder]
#        -> figure1_data_access.pdf, .png and .tif (600 dpi)
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyArrowPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
NAME = "figure1_data_access"
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                            "pdf.fonttype": 42, "svg.fonttype": "none"})

MM = 1 / 25.4                      # inches per mm
W = 90.0                           # single-column width (mm); the height follows from the rows
FS_LAB, FS_HEAD, FS_SMALL = 7.5, 7.5, 7.0   # no type below 7 pt

FILL, OPEN, LABEL = "fill", "open", "label"
BLUE, VERM, INK, GRID = "#1F3B73", "#D55E00", "#222222", "#B8B8B8"
COLS = [("Training", "60%", "#C9DCF2", "#E8F0FA"),
        ("Validation", "20%", "#F6D9A3", "#FCF0DA"),
        ("Test", "20%", "#E6C6C6", "#F6E9E9")]
# (stage label, [training, validation, test]); each cell: None or a list of marks,
# a mark being FILL, OPEN or (LABEL, regimes) with regimes "" when labels are read in every regime.
ROWS = [
    ("Scalers",                 [[FILL], [OPEN], [OPEN]]),
    ("Five detectors",          [[FILL, (LABEL, "N, O")], [OPEN], [OPEN]]),
    ("ECDF maps, ensembles",    [None, [FILL], [OPEN]]),
    ("Threshold $\\tau_\\rho$", [[(LABEL, "O")], [FILL], [OPEN]]),
    ("Supervised combinations", [None, [FILL, (LABEL, "")], [OPEN]]),
    ("Hyperparameter selection", [None, [FILL, (LABEL, "")], [OPEN]]),
    ("Metrics and intervals",   [None, None, [OPEN, (LABEL, "")]]),
]

X0, XC = 0.4, 37.0                 # label x, first data column x
CW = (W - 0.4 - XC) / 3            # data column width
RH = 5.6                           # row height
KEY_GAP, BOTTOM = 3.9, 2.3         # body-to-key distance, key-to-bottom margin
H = 15.2 + len(ROWS) * RH + KEY_GAP + BOTTOM
Y_ARROW, Y_HEAD_TOP, Y_HEAD_BOT = H - 4.2, H - 7.2, H - 15.2
Y_BODY_TOP = Y_HEAD_BOT
R_CIRC, R_DIAM, GAP = 1.25, 1.2, 0.45

fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

def circle(x, y, filled):
    ax.add_patch(Circle((x, y), R_CIRC, facecolor=BLUE if filled else "white", edgecolor=BLUE, lw=0.9, zorder=3))

def diamond(x, y):
    ax.add_patch(Polygon([(x, y + R_DIAM), (x + R_DIAM, y), (x, y - R_DIAM), (x - R_DIAM, y)],
                         closed=True, facecolor=VERM, edgecolor=VERM, lw=0.6, zorder=3))

def text_width_mm(s, fs):
    return TextPath((0, 0), s, size=fs, prop=FontProperties(family="DejaVu Sans")).get_extents().width * 25.4 / 72

def draw_marks(xc, yc, marks):
    widths = []
    for m in marks:
        if m in (FILL, OPEN): widths.append(2 * R_CIRC)
        else: widths.append(2 * R_DIAM + (text_width_mm(m[1], FS_SMALL) + 0.5 if m[1] else 0))
    total = sum(widths) + GAP * (len(marks) - 1)
    x = xc - total / 2
    for m, w in zip(marks, widths):
        if m in (FILL, OPEN):
            circle(x + R_CIRC, yc, m == FILL)
        else:
            diamond(x + R_DIAM, yc)
            if m[1]:
                ax.text(x + 2 * R_DIAM + 0.5, yc, m[1], fontsize=FS_SMALL, color=VERM,
                        ha="left", va="center_baseline", zorder=4)
        x += w + GAP

# time arrow spanning the three blocks
ax.add_patch(FancyArrowPatch((XC + 0.3, Y_ARROW), (W - 0.6, Y_ARROW), arrowstyle="-|>", mutation_scale=6,
                             lw=0.8, color=INK))
ax.text((XC + W) / 2, Y_ARROW + 0.9, "chronological order", fontsize=FS_SMALL, style="italic",
        ha="center", va="bottom", color=INK)

# column headers
for j, (name, share, dark, light) in enumerate(COLS):
    x = XC + j * CW
    ax.add_patch(Rectangle((x + 0.25, Y_HEAD_BOT + 0.25), CW - 0.5, Y_HEAD_TOP - Y_HEAD_BOT - 0.5,
                           facecolor=dark, edgecolor="none", zorder=1))
    ax.text(x + CW / 2, Y_HEAD_BOT + 4.85, name, fontsize=FS_HEAD, weight="bold", ha="center", va="center", color=INK)
    ax.text(x + CW / 2, Y_HEAD_BOT + 1.95, share, fontsize=FS_SMALL, ha="center", va="center", color=INK)
ax.text(X0, Y_HEAD_BOT + 3.4, "Pipeline stage", fontsize=FS_HEAD, weight="bold", ha="left", va="center", color=INK)

# body
for i, (label, cells) in enumerate(ROWS):
    y_top = Y_BODY_TOP - i * RH; yc = y_top - RH / 2
    ax.text(X0, yc, label, fontsize=FS_LAB, ha="left", va="center", color=INK)
    for j, marks in enumerate(cells):
        x = XC + j * CW
        if marks:
            ax.add_patch(Rectangle((x + 0.25, y_top - RH + 0.25), CW - 0.5, RH - 0.5,
                                   facecolor=COLS[j][3], edgecolor="none", zorder=1))
            draw_marks(x + CW / 2, yc, marks)
    ax.plot([0, W], [y_top - RH, y_top - RH], color=GRID, lw=0.35, zorder=0)
ax.plot([0, W], [Y_BODY_TOP, Y_BODY_TOP], color=INK, lw=0.6, zorder=2)

# key
y_key = Y_BODY_TOP - len(ROWS) * RH - KEY_GAP
items = [("fill", "fitted on this block"), ("open", "applied unchanged"), ("label", "labels read")]
x = 2.0
for kind, txt in items:
    if kind == "fill": circle(x + R_CIRC, y_key, True)
    elif kind == "open": circle(x + R_CIRC, y_key, False)
    else: diamond(x + R_DIAM, y_key)
    t = ax.text(x + 2 * R_CIRC + 1.0, y_key, txt, fontsize=FS_SMALL, ha="left", va="center", color=INK)
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(ax.transData.inverted())
    x = bb.x1 + 4.0

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, NAME + ".pdf"))
fig.savefig(os.path.join(OUT, NAME + ".png"), dpi=600)
from PIL import Image                       # TIFF: RGB, no alpha channel, 600 dpi, LZW
Image.open(os.path.join(OUT, NAME + ".png")).convert("RGB").save(
    os.path.join(OUT, NAME + ".tif"), dpi=(600, 600), compression="tiff_lzw")
print("size (mm):", W, "x", round(H, 1), "| key ends at x =", round(x - 4.0, 1))
