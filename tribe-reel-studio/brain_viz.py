"""Visualization and summary helpers for TRIBE v2 predictions on fsaverage5.

Everything here is pure numpy / matplotlib / plotly, so it can be tested
without a GPU or the TRIBE weights.
"""

from __future__ import annotations

import functools
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

logger = logging.getLogger(__name__)

N_VERTS_HEMI = 10242  # fsaverage5
VIEWS = {
    # name: (hemisphere, camera side along x: -1 = looking from the left)
    "Left lateral": ("left", -1),
    "Right lateral": ("right", 1),
    "Left medial": ("left", 1),
    "Right medial": ("right", -1),
}

# Destrieux (aparc.a2009s) parcels grouped into broad functional systems.
# These are anatomical approximations meant for a readable summary,
# not a substitute for functional localizers.
NETWORKS: dict[str, list[str]] = {
    "Visual": [
        "G_cuneus", "G_occipital_middle", "G_occipital_sup", "G_and_S_occipital_inf",
        "Pole_occipital", "S_calcarine", "G_oc-temp_med-Lingual",
        "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal", "S_occipital_ant",
        "S_parieto_occipital",
    ],
    "Faces & objects": [
        "G_oc-temp_lat-fusifor", "S_oc-temp_lat", "S_oc-temp_med_and_Lingual",
        "G_temporal_inf", "S_temporal_inf", "S_collat_transv_post",
    ],
    "Auditory": [
        "G_temp_sup-G_T_transv", "S_temporal_transverse", "G_temp_sup-Plan_tempo",
        "G_temp_sup-Plan_polar", "Lat_Fis-post",
    ],
    "Language": [
        "G_temp_sup-Lateral", "S_temporal_sup", "G_temporal_middle",
        "G_front_inf-Opercular", "G_front_inf-Triangul", "G_front_inf-Orbital",
        "Pole_temporal", "G_pariet_inf-Angular",
    ],
    "Attention": [
        "G_parietal_sup", "S_intrapariet_and_P_trans", "G_pariet_inf-Supramar",
        "S_precentral-sup-part",
    ],
    "Motor & touch": [
        "G_precentral", "G_postcentral", "S_central", "G_and_S_paracentral",
        "S_postcentral", "G_and_S_subcentral", "S_precentral-inf-part",
    ],
    "Prefrontal": [
        "G_front_sup", "G_front_middle", "S_front_sup", "S_front_middle",
        "S_front_inf", "G_and_S_frontomargin", "G_and_S_transv_frontopol",
    ],
    "Emotion & reward": [
        "G_orbital", "G_rectus", "S_orbital-H_Shaped", "S_orbital_lateral",
        "S_orbital_med-olfact", "S_suborbital", "G_subcallosal",
        "G_and_S_cingul-Ant", "G_insular_short", "G_Ins_lg_and_S_cent_ins",
        "S_circular_insula_ant", "S_circular_insula_inf", "S_circular_insula_sup",
    ],
    "Default mode (self & memory)": [
        "G_precuneus", "G_cingul-Post-dorsal", "G_cingul-Post-ventral",
        "S_subparietal", "G_oc-temp_med-Parahip",
    ],
}
NETWORK_COLORS = {
    "Overall": "#111111",
    "Visual": "#1f77b4",
    "Faces & objects": "#17becf",
    "Auditory": "#ff7f0e",
    "Language": "#d62728",
    "Attention": "#9467bd",
    "Motor & touch": "#8c564b",
    "Prefrontal": "#7f7f7f",
    "Emotion & reward": "#e377c2",
    "Default mode (self & memory)": "#2ca02c",
}


# ----------------------------------------------------------------------
# Mesh and atlas
# ----------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def load_mesh() -> dict:
    """Half-inflated fsaverage5 mesh (as in the TRIBE plotting code) + sulcal depth."""
    from nilearn import datasets, surface

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    out = {}
    for hemi in ("left", "right"):
        pial = surface.load_surf_mesh(fs[f"pial_{hemi}"])
        infl = surface.load_surf_mesh(fs[f"infl_{hemi}"])
        coords = 0.5 * np.asarray(pial.coordinates) + 0.5 * np.asarray(infl.coordinates)
        # pull the hemispheres apart so they never overlap in the 3D view
        if hemi == "left":
            coords[:, 0] -= coords[:, 0].max() + 5
        else:
            coords[:, 0] -= coords[:, 0].min() - 5
        sulc = np.asarray(surface.load_surf_data(fs[f"sulc_{hemi}"]), dtype=float)
        out[hemi] = dict(coords=coords, faces=np.asarray(infl.faces), sulc=sulc)
    return out


@functools.lru_cache(maxsize=1)
def load_network_masks() -> dict[str, np.ndarray] | None:
    """Boolean vertex masks (length 2*N_VERTS_HEMI) for each network in NETWORKS.

    Returns None if the Destrieux atlas cannot be downloaded.
    """
    from nilearn import datasets

    try:
        atlas = datasets.fetch_atlas_surf_destrieux()
    except Exception as e:  # network-restricted environments
        logger.warning("Could not fetch Destrieux atlas (%s); networks disabled", e)
        return None
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in atlas["labels"]]
    labels = np.r_[np.asarray(atlas["map_left"]), np.asarray(atlas["map_right"])]
    name_to_idx = {n: i for i, n in enumerate(names)}
    masks = {}
    for net, parcels in NETWORKS.items():
        idx = [name_to_idx[p] for p in parcels if p in name_to_idx]
        masks[net] = np.isin(labels, idx)
    cortex = ~np.isin(labels, [name_to_idx.get("Medial_wall", -1), name_to_idx.get("Unknown", 0)])
    masks["_cortex"] = cortex
    return masks


# ----------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------


# Same lines and axis on every reel's charts, so different reels can be compared.
COMPARE_NETWORKS = ["Visual", "Auditory", "Language", "Faces & objects", "Attention"]
CHART_YLIM = (-0.15, 0.35)


def chart_ylim(ts: dict) -> tuple[float, float]:
    """Fixed y-range, widened only if a reel goes beyond it."""
    vals = np.concatenate([s[~np.isnan(s)] for s in ts.values()] or [np.zeros(1)])
    lo, hi = CHART_YLIM
    if vals.size:
        lo, hi = min(lo, float(vals.min()) - 0.02), max(hi, float(vals.max()) + 0.02)
    return lo, hi


def densify(preds: np.ndarray, starts: list[float], tr: float, n_steps: int | None = None) -> np.ndarray:
    """Place per-segment predictions on a regular 0, TR, 2TR... grid (NaN where missing)."""
    idx = np.round(np.asarray(starts) / tr).astype(int)
    n = max(int(idx.max()) + 1, n_steps or 0)
    dense = np.full((n, preds.shape[1]), np.nan, dtype=np.float32)
    dense[idx] = preds
    return dense


def network_timeseries(dense: np.ndarray, masks: dict | None) -> dict[str, np.ndarray]:
    cortex = masks["_cortex"] if masks else np.ones(dense.shape[1], bool)
    out = {"Overall": np.nanmean(dense[:, cortex], axis=1) if len(dense) else np.array([])}
    if masks:
        for net in NETWORKS:
            out[net] = np.nanmean(dense[:, masks[net]], axis=1)
    return out


def find_peaks(series: np.ndarray, k: int = 3, min_gap: int = 3) -> list[int]:
    order = np.argsort(np.nan_to_num(series, nan=-np.inf))[::-1]
    chosen: list[int] = []
    for i in order:
        if np.isnan(series[i]):
            continue
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(int(i))
        if len(chosen) == k:
            break
    return chosen


def summary_table(ts: dict[str, np.ndarray], tr: float) -> list[list]:
    nets = [n for n in ts if n != "Overall"]
    winners = None
    if nets:
        stack = np.vstack([ts[n] for n in nets])
        valid = ~np.isnan(stack).any(0)
        winners = np.full(stack.shape[1], -1)
        winners[valid] = np.argmax(stack[:, valid], axis=0)
    rows = []
    for j, (name, s) in enumerate(ts.items()):
        if np.all(np.isnan(s)):
            continue
        peak_i = int(np.nanargmax(s))
        lead = "" if name == "Overall" or winners is None else (
            f"{100 * np.mean(winners[winners >= 0] == nets.index(name)):.0f}%"
        )
        rows.append([name, round(float(np.nanmean(s)), 4), round(float(np.nanmax(s)), 4),
                     round(peak_i * tr, 1), lead])
    return rows


def vertex_limits(dense: np.ndarray, percentile: float = 99.0, threshold_pct: float = 60.0):
    """Shared color limits across all timesteps so frames are comparable."""
    vals = np.abs(dense[~np.isnan(dense)])
    vmax = float(np.percentile(vals, percentile)) if vals.size else 1.0
    thr = vmax * threshold_pct / 100.0
    return thr, max(vmax, 1e-6)


# ----------------------------------------------------------------------
# Fast 2D rendering (matplotlib polygons with back-face culling)
# ----------------------------------------------------------------------


class BrainRenderer:
    """Renders four standard views quickly by projecting the mesh once and
    only updating face colours per frame."""

    def __init__(self, views=tuple(VIEWS)):
        self.mesh = load_mesh()
        self.views = list(views)
        self._geom = {v: self._project(*VIEWS[v]) for v in self.views}

    def _project(self, hemi: str, side: int):
        m = self.mesh[hemi]
        tri = m["coords"][m["faces"]]  # (F, 3, 3)
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        facing = normals[:, 0] * side
        keep = facing > 0
        depth = (tri[:, :, 0].mean(1) * side)[keep]
        order = np.argsort(depth)  # far first (painter's algorithm)
        faces_idx = np.flatnonzero(keep)[order]
        xy = np.stack([-side * tri[faces_idx, :, 1], tri[faces_idx, :, 2]], axis=-1)
        shade = 0.45 + 0.55 * facing[faces_idx]
        sulc = m["sulc"][m["faces"][faces_idx]].mean(1)
        base = np.where(sulc > 0, 0.55, 0.78)
        offset = 0 if hemi == "left" else N_VERTS_HEMI
        return dict(xy=xy, faces=m["faces"][faces_idx] + offset, shade=shade, base=base)

    def face_colors(self, view: str, values: np.ndarray, thr: float, vmax: float, diverging: bool):
        g = self._geom[view]
        rgb = np.repeat(g["base"][:, None], 3, axis=1)
        if values is not None and not np.all(np.isnan(values)):
            fv = np.nan_to_num(values[g["faces"]].mean(1))
            rgb = overlay(rgb, fv, thr, vmax, diverging)
        return np.clip(rgb * g["shade"][:, None], 0, 1)

    def draw(self, ax, view: str, values, thr, vmax, diverging=False, title=True):
        g = self._geom[view]
        pc = PolyCollection(g["xy"], facecolors=self.face_colors(view, values, thr, vmax, diverging),
                            edgecolors="face", linewidths=0.15, antialiaseds=False)
        ax.add_collection(pc)
        ax.set_xlim(g["xy"][..., 0].min() - 2, g["xy"][..., 0].max() + 2)
        ax.set_ylim(g["xy"][..., 1].min() - 2, g["xy"][..., 1].max() + 2)
        ax.set_aspect("equal")
        ax.axis("off")
        if title:
            ax.set_title(view, fontsize=9, color="#444")
        return pc


def _cmap(diverging: bool):
    return plt.get_cmap("RdBu_r" if diverging else "hot")


def overlay(rgb: np.ndarray, v: np.ndarray, thr: float, vmax: float, diverging: bool) -> np.ndarray:
    rgb = rgb.copy()
    if diverging:
        mask = np.abs(v) > thr
        x = np.clip(0.5 + 0.5 * v[mask] / vmax, 0, 1)
    else:
        mask = v > thr
        # map thr..vmax to the bright part of "hot" (dark red -> yellow/white)
        x = 0.25 + 0.75 * np.clip((v[mask] - thr) / max(vmax - thr, 1e-9), 0, 1)
    rgb[mask] = _cmap(diverging)(x)[:, :3]
    return rgb


def colorbar_mappable(thr, vmax, diverging):
    if diverging:
        norm = matplotlib.colors.Normalize(-vmax, vmax)
    else:
        norm = matplotlib.colors.Normalize(thr - (vmax - thr) / 3, vmax)
    return matplotlib.cm.ScalarMappable(norm=norm, cmap=_cmap(diverging))


def render_static(renderer: BrainRenderer, values, thr, vmax, diverging, title: str, path: str):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.2), dpi=110)
    for ax, view in zip(axes.flat, renderer.views):
        renderer.draw(ax, view, values, thr, vmax, diverging)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    cax = fig.add_axes([0.25, 0.04, 0.5, 0.02])
    fig.colorbar(colorbar_mappable(thr, vmax, diverging), cax=cax, orientation="horizontal",
                 label="predicted response (a.u.)")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1, wspace=0.02, hspace=0.12)
    fig.savefig(path)
    plt.close(fig)
    return path


class PanelAnimator:
    """Per-timestep panel: 4 brain views + intensity timeline with a cursor."""

    def __init__(self, renderer: BrainRenderer, ts: dict, tr: float, thr, vmax, diverging,
                 width_px=640, height_px=720, unreliable_last=False):
        self.r, self.thr, self.vmax, self.div = renderer, thr, vmax, diverging
        dpi = 100
        self.fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor="white")
        gs = self.fig.add_gridspec(2, 2, hspace=0.12, wspace=0.02,
                                   left=0.03, right=0.97, top=0.9, bottom=0.33)
        self.pcs = {}
        for k, view in enumerate(renderer.views):
            ax = self.fig.add_subplot(gs[k // 2, k % 2])
            self.pcs[view] = renderer.draw(ax, view, None, thr, vmax, diverging)
        ax = self.fig.add_axes([0.1, 0.07, 0.86, 0.22])
        t = np.arange(len(ts["Overall"])) * tr
        for name in ["Overall"] + [n for n in COMPARE_NETWORKS if n in ts]:
            ax.plot(t, ts[name], color=NETWORK_COLORS.get(name, None),
                    lw=2.2 if name == "Overall" else 1.2, label=name)
        ax.set_ylim(*chart_ylim(ts))
        if unreliable_last and len(t) > 1:
            ax.axvspan(t[-1] - tr / 2, t[-1] + tr / 2, color="#999", alpha=0.25, lw=0)
        ax.legend(fontsize=7, ncol=3, loc="upper left", frameon=False)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        self.cursor = ax.axvline(0, color="#e4572e", lw=2)
        self.title = self.fig.text(0.5, 0.955, "", ha="center", va="center", fontsize=13,
                                   fontweight="bold")
        self.ts, self.tr = ts, tr

    def frame(self, values: np.ndarray, i: int) -> np.ndarray:
        for view, pc in self.pcs.items():
            pc.set_facecolor(self.r.face_colors(view, values, self.thr, self.vmax, self.div))
        self.cursor.set_xdata([i * self.tr, i * self.tr])
        nets = {n: s[i] for n, s in self.ts.items() if n != "Overall" and not np.isnan(s[i])}
        top = ", ".join(sorted(nets, key=nets.get, reverse=True)[:2])
        label = f"t = {i * self.tr:.0f}s" + (f"  ·  top: {top}" if top else "")
        if np.all(np.isnan(values)):
            label = f"t = {i * self.tr:.0f}s  ·  (no prediction)"
        self.title.set_text(label)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        plt.close(self.fig)


# ----------------------------------------------------------------------
# Plotly figures
# ----------------------------------------------------------------------


def _plotly_scale(values: np.ndarray, base: np.ndarray, thr: float, vmax: float, diverging: bool):
    """Encode background (sulcal gray) and activation into one 0..1 intensity
    so a single colorscale can show both: [0, .2) = anatomy, [.2, 1] = activation."""
    out = 0.2 * base
    v = np.nan_to_num(values)
    if diverging:
        m = np.abs(v) > thr
        out[m] = 0.2 + 0.8 * np.clip(0.5 + 0.5 * v[m] / vmax, 0, 0.999)
    else:
        m = v > thr
        out[m] = 0.2 + 0.8 * np.clip((v[m] - thr) / max(vmax - thr, 1e-9), 0, 1)
    return out.astype(np.float32)


def _plotly_colorscale(diverging: bool):
    cm = _cmap(diverging)
    scale = [[0.0, "rgb(120,120,120)"], [0.1999, "rgb(205,205,205)"]]
    for x in np.linspace(0, 1, 9):
        r, g, b, _ = cm(x if diverging else 0.25 + 0.75 * x)
        scale.append([0.2 + 0.8 * x, f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"])
    return scale


CAMERAS = {
    "Left": dict(x=-1.7, y=0, z=0.2),
    "Right": dict(x=1.7, y=0, z=0.2),
    "Top": dict(x=0, y=-0.01, z=1.9),
    "Front": dict(x=0, y=1.9, z=0.2),
    "Back": dict(x=0, y=-1.9, z=0.2),
}


def brain_3d_figure(dense: np.ndarray, tr: float, thr: float, vmax: float, diverging: bool,
                    mean_map: np.ndarray | None = None):
    import plotly.graph_objects as go

    mesh = load_mesh()
    coords = np.r_[mesh["left"]["coords"], mesh["right"]["coords"]]
    faces = np.r_[mesh["left"]["faces"], mesh["right"]["faces"] + N_VERTS_HEMI]
    sulc = np.r_[mesh["left"]["sulc"], mesh["right"]["sulc"]]
    base = np.where(sulc > 0, 0.25, 0.95)

    frames_vals = []
    if mean_map is not None:
        frames_vals.append(("average", "Average", mean_map))
    for i in range(len(dense)):
        frames_vals.append((str(i), f"{i * tr:.0f}s", dense[i]))

    first = _plotly_scale(frames_vals[0][2], base, thr, vmax, diverging)
    trace = go.Mesh3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=first, cmin=0, cmax=1, colorscale=_plotly_colorscale(diverging),
        showscale=False, hoverinfo="skip", flatshading=False,
        lighting=dict(ambient=0.55, diffuse=0.7, specular=0.05, roughness=0.9),
        lightposition=dict(x=0, y=0, z=1e5),
    )
    frames = [go.Frame(name=name, data=[go.Mesh3d(intensity=_plotly_scale(v, base, thr, vmax, diverging))],
                       traces=[0]) for name, _, v in frames_vals]
    steps = [dict(method="animate", label=label,
                  args=[[name], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                     transition=dict(duration=0))])
             for name, label, _ in frames_vals]
    play_frames = [name for name, _, _ in frames_vals if name != "average"]
    fig = go.Figure(data=[trace], frames=frames)
    fig.update_layout(
        height=660, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="white",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   aspectmode="data", camera=dict(eye=dict(x=-1.6, y=0, z=0.3))),
        sliders=[dict(active=0, steps=steps, x=0.05, len=0.9, y=0.02,
                      currentvalue=dict(prefix="Showing: "))],
        updatemenus=[dict(type="buttons", direction="left", x=0.05, y=1.0, showactive=False,
                          buttons=[dict(label=lbl, method="relayout", args=[{"scene.camera.eye": eye}])
                                   for lbl, eye in CAMERAS.items()]),
                     dict(type="buttons", direction="left", x=0.05, y=0.12, showactive=False,
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[play_frames, dict(frame=dict(duration=int(tr * 1000), redraw=True),
                                                           transition=dict(duration=0), fromcurrent=True,
                                                           mode="immediate")]),
                              dict(label="❚❚ Pause", method="animate",
                                   args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))]),
                          ])],
    )
    return fig


def intensity_figure(ts: dict[str, np.ndarray], tr: float, peaks: list[int],
                     unreliable_last: bool = False):
    import plotly.graph_objects as go

    fig = go.Figure()
    t = np.arange(len(ts["Overall"])) * tr
    for name, s in ts.items():
        fig.add_trace(go.Scatter(
            x=t, y=s, name=name, mode="lines",
            line=dict(color=NETWORK_COLORS.get(name), width=4 if name == "Overall" else 2),
            visible=True if name in ["Overall"] + COMPARE_NETWORKS else "legendonly",
        ))
    if unreliable_last and len(t) > 1:
        fig.add_vrect(x0=t[-1] - tr / 2, x1=t[-1] + tr / 2, fillcolor="#999", opacity=0.25,
                      line_width=0, annotation_text="end of video: less reliable",
                      annotation_position="bottom right")
    for rank, i in enumerate(peaks, 1):
        fig.add_vline(x=i * tr, line=dict(color="#e4572e", dash="dot"),
                      annotation_text=f"peak #{rank}", annotation_position="top")
    fig.update_layout(
        height=480, margin=dict(l=50, r=20, t=40, b=40), hovermode="x unified",
        xaxis_title="time in reel (s)", yaxis_title="predicted response (a.u.)",
        legend=dict(orientation="h", y=-0.2), template="plotly_white",
        yaxis=dict(range=list(chart_ylim(ts))),
        title=dict(text="Brain response intensity over time (click legend items to toggle)",
                   font=dict(size=14)),
    )
    return fig
