from __future__ import annotations

from pathlib import Path

import numpy as np


def split_hemispheres(prediction: np.ndarray, vertices_per_hemisphere: int) -> tuple[np.ndarray, np.ndarray]:
    if prediction.ndim != 1:
        raise ValueError("Expected a 1D vertex vector for visualization.")

    if prediction.shape[0] == 2 * vertices_per_hemisphere:
        split = vertices_per_hemisphere
    else:
        split = prediction.shape[0] // 2

    return prediction[:split], prediction[split:]


def save_brain_snapshot_html(
    prediction: np.ndarray,
    output_path: Path,
    *,
    vertices_per_hemisphere: int = 10242,
    title: str = "TRIBE v2 prediction",
) -> Path:
    from nilearn import datasets as nl_datasets
    from nilearn.plotting import view_surf

    fsavg = nl_datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    left, right = split_hemispheres(prediction, vertices_per_hemisphere)
    vmax = max(float(np.percentile(np.abs(prediction), 99)), 1e-6)

    left_view = view_surf(
        surf_mesh=fsavg["infl_left"],
        surf_map=left,
        bg_map=fsavg["sulc_left"],
        hemi="left",
        threshold="20%",
        cmap="hot",
        black_bg=True,
        vmax=vmax,
        bg_on_data=True,
        colorbar=True,
        title=f"{title} [Left]",
    )
    right_view = view_surf(
        surf_mesh=fsavg["infl_right"],
        surf_map=right,
        bg_map=fsavg["sulc_right"],
        hemi="right",
        threshold="20%",
        cmap="hot",
        black_bg=True,
        vmax=vmax,
        bg_on_data=True,
        colorbar=True,
        title=f"{title} [Right]",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = (
        "<html><body style='margin:0;background:#000;'>"
        "<div style='display:flex;gap:10px;'>"
        f"{left_view.get_iframe(width='48%', height='500px')}"
        f"{right_view.get_iframe(width='48%', height='500px')}"
        "</div></body></html>"
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path
