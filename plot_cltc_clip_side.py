"""
Plots for the CLTC clipping fix in main.glsl: where the fit gives d > 0, why clipping the +x lune is
wrong there, and the samples that fall below the horizon before the fix.

Self-contained: ports ltc_coeffs() and cltc_sample() from main.glsl. Needs numpy and matplotlib.
Writes cltc_clip_side.png.
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

PI = np.pi


def ltc_coeffs(mu, r):
    a = 1.0 + r * (0.303392 + (-0.518982 + 0.111709 * mu) * mu + (-0.276266 + 0.335918 * mu) * r)
    b = r * (-1.16407 + 1.15859 * mu + (0.150815 - 0.150105 * mu) * r) / (mu * mu * mu - 1.43545)
    c = 1.0 + r * (0.20013 + (-0.506373 + 0.261777 * mu) * mu)
    d = r * (0.540852 + (-1.01625 + 0.475392 * mu) * mu) / (-1.0743 + (0.0725628 + mu) * mu)
    return a, b, c, d


def cltc_sample_z(mu, r, u1, u2, sign_aware):
    """z of the sampled wi in the local frame (the frame rotation about z does not change it)."""
    a, b, c, d = ltc_coeffs(mu, r)
    R = np.sqrt(u1)
    phi = 2 * PI * u2
    x, y = R * np.cos(phi), R * np.sin(phi)
    vz = 1 / np.sqrt(d * d + 1)
    s = 0.5 * (1 + vz)
    side = (1.0 if d >= 0 else -1.0) if sign_aware else -1.0
    x = side * ((1 - s) * np.sqrt(1 - y * y) + s * x)
    whz = np.sqrt(np.maximum(1 - (x * x + y * y), 0))
    w = np.stack([a * x + b * whz, c * y, d * x + whz], axis=-1)
    return w[:, 2] / np.linalg.norm(w, axis=-1)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=130)

# (a) d near normal incidence
ax = axes[0]
mu = np.linspace(0.9990, 1.0, 400)
for r, col in ((0.25, "#86b6ef"), (0.5, "#3987e5"), (1.0, "#0d366b")):
    ax.plot(mu, ltc_coeffs(mu, r)[3], color=col, lw=2, label=f"r = {r:g}")
ax.axhline(0, color="k", lw=0.8)
mu0 = (1.01625 - np.sqrt(1.01625**2 - 4 * 0.475392 * 0.540852)) / (2 * 0.475392)
ax.axvspan(mu0, 1.0, color="#f4b6b0", alpha=0.6, lw=0)
ax.annotate(f"d > 0 for mu > {mu0:.5f},\nwithin {np.degrees(np.arccos(mu0)):.2f} deg of the normal",
            (0.99996, 0.0012), xytext=(0.99905, 0.0022), textcoords="data", fontsize=8, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b"))
ax.set_title("(a) d from ltc_coeffs() near normal incidence", loc="left", fontsize=10)
ax.set_xlabel("mu = cos(theta_o)")
ax.set_ylabel("d")
ax.legend(frameon=False, fontsize=8, loc="lower right")

# (b) the projected sampling region in w_h space, with d exaggerated so that the lunes are visible
ax = axes[1]
d_ill = 0.6
vz = 1 / np.sqrt(1 + d_ill**2)
t = np.linspace(-PI / 2, PI / 2, 200)
circle_pos = np.c_[np.cos(t), np.sin(t)]           # half disk, x > 0
circle_neg = np.c_[-np.cos(t), np.sin(t)]          # half disk, x < 0
ellipse_pos = np.c_[vz * np.cos(t), np.sin(t)]
ellipse_neg = np.c_[-vz * np.cos(t), np.sin(t)]
# valid region for d > 0: d x + z >= 0, which projects to the half disk (x > 0) and the half ellipse (x < 0)
ax.add_patch(Polygon(np.r_[circle_pos, ellipse_neg[::-1]], closed=True, fc="#d6e6f8", ec="none",
                     label="valid: maps to wi.z > 0"))
ax.add_patch(Polygon(np.r_[ellipse_neg, circle_neg[::-1]], closed=True, fc="#f4b6b0", ec="none",
                     label="sampled by Listing 4 but wi.z < 0"))
ax.add_patch(Polygon(np.r_[circle_pos, ellipse_pos[::-1]], closed=True, fc="#f9d59b", ec="none",
                     label="valid but never sampled by Listing 4"))
ax.plot(*np.r_[circle_neg, ellipse_pos[::-1], circle_neg[:1]].T, color="#c0392b", lw=1.6,
        label="region sampled by Listing 4")
ax.plot(*np.r_[circle_pos, ellipse_neg[::-1], circle_pos[:1]].T, color="#1f6fd1", lw=1.6, ls="--",
        label="region sampled with the fix")
ax.set_aspect("equal")
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.1, 2.3)
ax.set_xlabel("x of w_h")
ax.set_ylabel("y of w_h")
ax.set_title(f"(b) Projected w_h region for d > 0 (d = {d_ill} to show it)", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=1)

# (c) samples below the horizon at mu = 1, r = 1
ax = axes[2]
rng = np.random.default_rng(11)
n, chunks = 20_000_000, 3
bins = np.linspace(-0.003, 0.003, 25)
for sign_aware, col, label in ((False, "#c0392b", "Listing 4"), (True, "#1f6fd1", "clip side from the sign of d")):
    counts = np.zeros(len(bins) - 1)
    below = 0
    for _ in range(chunks):
        z = cltc_sample_z(1.0, 1.0, rng.random(n), rng.random(n), sign_aware)
        counts += np.histogram(z, bins=bins)[0]
        below += np.count_nonzero(z < 0)
    ax.stairs(counts, bins, color=col, lw=2,
              label=f"{label}: {below} of {n * chunks // 10**6}M samples below the horizon")
ax.axvline(0, color="k", lw=0.8)
ax.set_ylim(0, None)
ax.set_xlabel("wi.z of CLTC samples")
ax.set_ylabel("samples per bin")
ax.set_title("(c) CLTC samples near the horizon, mu = 1, r = 1", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper left")
for side in ("top", "right"):
    for a_ in axes:
        a_.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig("cltc_clip_side.png")
print("wrote cltc_clip_side.png")
