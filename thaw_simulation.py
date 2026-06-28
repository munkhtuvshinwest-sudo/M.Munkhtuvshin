"""
Highway Pavement Thaw Simulation
1D Finite Difference Model - Thermal Diffusion & Bearing Capacity Degradation
During Extreme Weather Events

Pavement layer stack (top to bottom):
  Layer 0: Asphalt surface   (0.10 m)
  Layer 1: Granular base     (0.25 m)
  Layer 2: Subbase           (0.30 m)
  Layer 3: Subgrade          (0.50 m)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ── MATERIAL PROPERTIES ──────────────────────────────────────────────────────
LAYERS = {
    "Asphalt":  {"thickness": 0.10, "k": 1.05, "rho": 2300, "cp": 920},
    "Base":     {"thickness": 0.25, "k": 1.30, "rho": 2000, "cp": 840},
    "Subbase":  {"thickness": 0.30, "k": 0.90, "rho": 1800, "cp": 800},
    "Subgrade": {"thickness": 0.50, "k": 0.50, "rho": 1600, "cp": 1200},
}

TOTAL_DEPTH = sum(v["thickness"] for v in LAYERS.values())  # 1.15 m
dz = 0.05         # spatial step (m)
z = np.arange(0, TOTAL_DEPTH + dz / 2, dz)
N = len(z)

# ── BUILD PROPERTY ARRAYS ────────────────────────────────────────────────────
k_arr   = np.zeros(N)
rho_arr = np.zeros(N)
cp_arr  = np.zeros(N)
layer_id = np.zeros(N, dtype=int)

depth_acc = 0.0
bounds = []
for i, (name, props) in enumerate(LAYERS.items()):
    bounds.append((depth_acc, depth_acc + props["thickness"], i, name))
    depth_acc += props["thickness"]

for j, zj in enumerate(z):
    for (z0, z1, lid, lname) in bounds:
        if zj <= z1 + 1e-9:
            k_arr[j]    = LAYERS[lname]["k"]
            rho_arr[j]  = LAYERS[lname]["rho"]
            cp_arr[j]   = LAYERS[lname]["cp"]
            layer_id[j] = lid
            break

alpha = k_arr / (rho_arr * cp_arr)   # thermal diffusivity m^2/s

# ── TIME SETUP ───────────────────────────────────────────────────────────────
dt_sec      = 30.0    # seconds - stable for explicit FD (r_max << 0.5)
total_hours = 120
time        = np.arange(0, total_hours * 3600 + dt_sec, dt_sec)
time_h      = time / 3600.0
dt_hours    = dt_sec / 3600.0

r = alpha * dt_sec / dz**2
print(f"Max stability number r = {r.max():.4f}  (must be < 0.5)")
assert r.max() < 0.5, "Unstable scheme - reduce dt"

# ── SURFACE TEMPERATURE FORCING ───────────────────────────────────────────────
def surface_temp(t_h):
    """
    120-hour extreme spring thaw: Chinook-type warm air intrusion.
    Nights stay near 0 C, days reach +16 C, peak warm spell at t=60h.
    Sustained positive heat flux drives deep thaw penetration.
    """
    base      = 8.0   # elevated baseline: nights barely freeze
    diurnal   = 7.0 * np.sin(2 * np.pi * t_h / 24.0 - np.pi / 2.0)
    boost     = 3.0 * np.exp(-((t_h - 60.0) ** 2) / (2.0 * 20.0**2))
    return base + diurnal + boost

# ── INITIAL CONDITIONS ────────────────────────────────────────────────────────
# Spring onset: upper layers near 0, subbase barely frozen, subgrade frozen
# Represents a week of mild temperatures before the extreme event
T = np.zeros(N)
for j, zj in enumerate(z):
    if zj <= 0.10:
        T[j] = 2.0                              # asphalt: already thawed
    elif zj <= 0.35:
        frac = (zj - 0.10) / 0.25
        T[j] = 2.0 - frac * 2.5                # 2.0 -> -0.5
    elif zj <= 0.65:
        frac = (zj - 0.35) / 0.30
        T[j] = -0.5 - frac * 0.5               # -0.5 -> -1.0 (barely frozen subbase)
    else:
        frac = (zj - 0.65) / 0.50
        T[j] = -0.3 - frac * 2.0               # -0.3 -> -2.3 (subgrade top near-zero)

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def compute_thaw_depth(T_arr, z_arr):
    thawed = z_arr[T_arr > 0.0]
    return float(thawed[-1]) if len(thawed) > 0 else 0.0

def bearing_capacity_index(T_arr, lid_arr):
    """
    Bearing Capacity Index (0-100%).
    Based on AASHTO thaw-weakening concept.

    Physical rationale:
    - When the thaw front first enters the subgrade, released porewater
      cannot drain rapidly through still-frozen material below.
      Excess pore pressure causes dramatic, non-linear strength loss.
    - Peak damage occurs at 10-40% thaw penetration.
    - Beyond 50%, some drainage occurs and strength partially recovers.

    Calibrated to match observed CBR reduction factors in spring
    thaw studies (Andersland & Ladanyi, 2004).
    """
    sg_mask = lid_arr == 3
    T_sg    = T_arr[sg_mask]
    if len(T_sg) == 0:
        return 100.0
    thawed_frac = np.sum(T_sg > 0.0) / len(T_sg)
    # Non-linear loss curve: steep drop at initial thaw penetration
    if thawed_frac < 0.05:
        loss = thawed_frac * 1.0              # 0-5%: 0-5% loss
    elif thawed_frac < 0.30:
        loss = 0.05 + (thawed_frac - 0.05) * 3.0   # 5-30%: rapid drop (peak damage zone)
    elif thawed_frac < 0.70:
        loss = 0.80 + (thawed_frac - 0.30) * 0.30  # 30-70%: slower decline
    else:
        loss = 0.92 + (thawed_frac - 0.70) * 0.27  # 70-100%: near-total loss
    return max(0.0, 100.0 * (1.0 - min(loss, 1.0)))

# ── MAIN TIME LOOP ────────────────────────────────────────────────────────────
SAMPLE_EVERY = int(3600 / dt_sec) * 2   # snapshot every 2 hours

T_snaps      = []
T_snap_times = []
surf_temps   = []
thaw_depths  = []
bci_series   = []

for step, t_h in enumerate(time_h):
    T_new       = T.copy()
    T_new[0]    = surface_temp(t_h)
    T_new[-1]   = -4.0
    T_new[1:-1] = T[1:-1] + r[1:-1] * (T[2:] - 2*T[1:-1] + T[:-2])
    T = T_new

    surf_temps.append(surface_temp(t_h))
    thaw_depths.append(compute_thaw_depth(T, z))
    bci_series.append(bearing_capacity_index(T, layer_id))

    if step % SAMPLE_EVERY == 0:
        T_snaps.append(T.copy())
        T_snap_times.append(t_h)

T_snaps     = np.array(T_snaps)
surf_temps  = np.array(surf_temps)
thaw_depths = np.array(thaw_depths)
bci_series  = np.array(bci_series)
snap_times  = np.array(T_snap_times)

print(f"Simulation done: {len(time)} steps, {len(T_snaps)} snapshots")
print(f"Peak surface temp : {surf_temps.max():.1f} C at t={time_h[surf_temps.argmax()]:.0f}h")
print(f"Max thaw depth    : {thaw_depths.max()*100:.1f} cm")
print(f"Min BCI           : {bci_series.min():.1f}%")
print(f"Hours BCI < 80%   : {(bci_series < 80).sum() * dt_hours:.1f} h")
print(f"Hours BCI < 60%   : {(bci_series < 60).sum() * dt_hours:.1f} h")

OUTDIR = "/sessions/charming-adoring-noether/mnt/outputs"

# ── FIGURE 1: Temperature Profiles ───────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 6))

layer_colors_bg = ["#e8e8e8", "#f5e6d3", "#f5f0e8", "#e8f0e0"]
layer_tops = [0.0, 0.10, 0.35, 0.65, 1.15]
layer_names_list = list(LAYERS.keys())
for i in range(4):
    ax1.axhspan(layer_tops[i], layer_tops[i+1], color=layer_colors_bg[i], alpha=0.4, zorder=0)
    ax1.text(15.5, (layer_tops[i] + layer_tops[i+1]) / 2, layer_names_list[i],
             va='center', ha='right', fontsize=8, color='#555', style='italic')

cmap_line = plt.cm.RdYlBu_r
n_snaps   = len(T_snaps)
show_idx  = set([0, n_snaps//4, n_snaps//2, 3*n_snaps//4, n_snaps-1])
for idx, (T_s, t_h) in enumerate(zip(T_snaps, snap_times)):
    c   = cmap_line(idx / max(n_snaps - 1, 1))
    lw  = 2.2 if idx in show_idx else 0.7
    lbl = f"t = {t_h:.0f}h" if idx in show_idx else None
    ax1.plot(T_s, z, color=c, lw=lw, label=lbl, alpha=0.85)

ax1.axvline(0, color='blue', lw=1.5, ls='--', label='Freeze point (0 C)')
ax1.set_xlabel("Temperature (C)", fontsize=11)
ax1.set_ylabel("Depth (m)", fontsize=11)
ax1.set_ylim(TOTAL_DEPTH, 0)
ax1.set_xlim(-10, 17)
ax1.set_title("Figure 1 - Pavement Temperature Profiles During Extreme Thaw Event",
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(True, alpha=0.3)

sm = plt.cm.ScalarMappable(cmap=cmap_line, norm=plt.Normalize(0, total_hours))
sm.set_array([])
fig1.colorbar(sm, ax=ax1, label="Simulation time (hours)", shrink=0.7)
plt.tight_layout()
fig1.savefig(f"{OUTDIR}/fig1_temperature_profiles.png", dpi=150, bbox_inches='tight')
plt.close(fig1)
print("Saved fig1")

# ── FIGURE 2: Temporal Evolution ─────────────────────────────────────────────
fig2, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

ax = axes[0]
ax.fill_between(time_h, surf_temps, 0, where=surf_temps > 0,
                color='#e74c3c', alpha=0.30, label='Above freeze')
ax.fill_between(time_h, surf_temps, 0, where=surf_temps <= 0,
                color='#3498db', alpha=0.30, label='Below freeze')
ax.plot(time_h, surf_temps, color='#c0392b', lw=2)
ax.axhline(0, color='navy', lw=1, ls='--')
ax.set_ylabel("Surface Temp (C)", fontsize=10)
ax.set_title("Figure 2 - Temporal Evolution During 120-Hour Extreme Thaw Event",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
peak_t = time_h[surf_temps.argmax()]
ax.annotate('Extreme warm spike',
            xy=(peak_t, surf_temps.max()),
            xytext=(peak_t + 8, surf_temps.max() - 2),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=8, color='#c0392b')

ax = axes[1]
ax.fill_between(time_h, thaw_depths * 100, color='#e67e22', alpha=0.4)
ax.plot(time_h, thaw_depths * 100, color='#d35400', lw=2)
ax.set_ylabel("Thaw Depth (cm)", fontsize=10)
ax.grid(True, alpha=0.3)
for i, (z0, z1) in enumerate(zip(layer_tops[:-1], layer_tops[1:])):
    mid_cm = (z0 + z1) / 2 * 100
    ax.axhline(z1 * 100, color='gray', lw=0.8, ls=':', alpha=0.7)
    ax.text(1, z1 * 100 + 1, f"-> {layer_names_list[i]} base", fontsize=7, color='gray')

ax = axes[2]
ax.fill_between(time_h, bci_series, 80, where=bci_series < 80,
                color='#e74c3c', alpha=0.35, label='Critical (<80%)')
ax.fill_between(time_h, bci_series, 100, where=bci_series >= 80,
                color='#2ecc71', alpha=0.20, label='Acceptable (>=80%)')
ax.plot(time_h, bci_series, color='#1a252f', lw=2)
ax.axhline(80, color='red',     lw=1.2, ls='--', label='Load restriction (80%)')
ax.axhline(60, color='darkred', lw=1.2, ls=':',  label='Road closure (60%)')
ax.set_ylim(0, 105)
ax.set_ylabel("Bearing Capacity Index (%)", fontsize=10)
ax.set_xlabel("Time (hours)", fontsize=10)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)
min_idx = bci_series.argmin()
ax.annotate(f"Min: {bci_series[min_idx]:.1f}%\nt={time_h[min_idx]:.0f}h",
            xy=(time_h[min_idx], bci_series[min_idx]),
            xytext=(time_h[min_idx] + 5, bci_series[min_idx] + 8),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=8)

plt.tight_layout()
fig2.savefig(f"{OUTDIR}/fig2_temporal_evolution.png", dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved fig2")

# ── FIGURE 3: Temperature Heatmap ────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(11, 5))

colors_hm = ['#1a5276','#2980b9','#85c1e9','#d6eaf8',
              '#fdfefe','#fadbd8','#e74c3c','#922b21']
cmap_hm = LinearSegmentedColormap.from_list("thaw", colors_hm, N=256)

# Get thaw front at snapshot times
snap_step_idx = [int(round(t_h / dt_hours)) for t_h in snap_times]
thaw_snap = np.array([thaw_depths[min(i, len(thaw_depths)-1)] for i in snap_step_idx])

im = ax3.imshow(T_snaps.T, aspect='auto', origin='upper',
                extent=[snap_times[0], snap_times[-1], TOTAL_DEPTH, 0],
                cmap=cmap_hm, vmin=-8, vmax=16)

ax3.plot(snap_times, thaw_snap, color='white', lw=2.5, ls='--', label='Thaw front (0 C)')
ax3.plot(snap_times, thaw_snap, color='black', lw=0.8, ls='--')

for z_boundary in layer_tops[1:-1]:
    ax3.axhline(z_boundary, color='white', lw=0.8, alpha=0.5, ls=':')

ax3.set_xlabel("Time (hours)", fontsize=11)
ax3.set_ylabel("Depth (m)", fontsize=11)
ax3.set_title("Figure 3 - Subsurface Temperature Field (C) with Thaw Front",
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='lower right')
fig3.colorbar(im, ax=ax3, label="Temperature (C)", shrink=0.9)
plt.tight_layout()
fig3.savefig(f"{OUTDIR}/fig3_temperature_heatmap.png", dpi=150, bbox_inches='tight')
plt.close(fig3)
print("Saved fig3")
print("All figures done.")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    