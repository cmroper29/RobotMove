"""Quick-look plot of a trajectory (requires matplotlib)."""
from __future__ import annotations

import numpy as np

from .transforms import parse_axis


def plot_trajectory(result, path: str, title: str = "", axis: str = "z", arrows: int = 40) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, pos = result.t, result.position
    speed = np.linalg.norm(np.gradient(pos, t, axis=0), axis=1) if len(t) > 1 else np.zeros(len(t))
    rows = 3 if result.joints is not None else 2
    fig = plt.figure(figsize=(12, 4 * rows))
    ax = fig.add_subplot(rows, 2, (1, 3), projection="3d")
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=speed, s=2, cmap="viridis")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="TCP speed (mm/s)")
    idx, sign = parse_axis(axis)
    step = max(1, len(t) // arrows)
    d = sign * result.rotation[::step, :, idx]
    span = float(np.ptp(pos, axis=0).max()) if len(pos) else 1.0
    ax.quiver(pos[::step, 0], pos[::step, 1], pos[::step, 2], d[:, 0], d[:, 1], d[:, 2],
              length=0.06 * max(span, 1.0), color="tab:red", linewidth=0.6)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(title)
    try:
        ax.set_box_aspect(np.maximum(np.ptp(pos, axis=0), 1e-3))
    except Exception:
        pass

    ax2 = fig.add_subplot(rows, 2, 2)
    ax2.plot(t, speed, lw=1)
    ax2.set_ylabel("TCP speed (mm/s)")
    ax2.grid(alpha=0.3)
    ax3 = fig.add_subplot(rows, 2, 4)
    for i, lab in enumerate("xyz"):
        ax3.plot(t, pos[:, i], lw=1, label=lab)
    ax3.set_ylabel("position (mm)")
    ax3.legend(loc="upper right")
    ax3.grid(alpha=0.3)
    if result.joints is not None:
        ax4 = fig.add_subplot(rows, 2, 5)
        ax5 = fig.add_subplot(rows, 2, 6)
        qd = np.gradient(result.joints, t, axis=0) if len(t) > 1 else np.zeros_like(result.joints)
        for i, name in enumerate(result.joint_names):
            ax4.plot(t, result.joints[:, i], lw=1, label=name)
            ax5.plot(t, qd[:, i], lw=1, label=name)
        ax4.set_ylabel("joints (deg)")
        ax5.set_ylabel("joint speed (deg/s)")
        ax4.legend(loc="upper right", fontsize=7)
        for a in (ax4, ax5):
            a.set_xlabel("time (s)")
            a.grid(alpha=0.3)
    else:
        ax2.set_xlabel("time (s)")
        ax3.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
