"""Render README figures from a demo_eval run.

  python -m lang_nav.plot_demo docs/lang_nav/run1 --map /opt/ros/humble/share/nav2_bringup/maps/turtlebot3_world.yaml

Writes: map_landmarks.png, demo_trajectories.png, visual_target.png, results.md
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from lang_nav.grounding import LandmarkMap  # noqa: E402

DEFAULT_MAP = '/opt/ros/humble/share/nav2_bringup/maps/turtlebot3_world.yaml'
CROP = (-3.0, 3.0, -2.8, 2.8)


def load_map(path):
    with open(path) as f:
        meta = yaml.safe_load(f)
    img_path = os.path.join(os.path.dirname(path), meta['image'])
    img = plt.imread(img_path)
    if img.ndim == 3:
        img = img[..., 0]
    h, w = img.shape
    res = meta['resolution']
    ox, oy = meta['origin'][:2]
    extent = (ox, ox + w * res, oy, oy + h * res)
    return img, extent


def draw_map(ax, img, extent):
    ax.imshow(img, cmap='gray', extent=extent, origin='upper', vmin=0, vmax=1 if img.max() <= 1 else 255)
    ax.set_xlim(CROP[0], CROP[1])
    ax.set_ylim(CROP[2], CROP[3])
    ax.set_aspect('equal')
    ax.set_xlabel('x [m] (map)')
    ax.set_ylabel('y [m] (map)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--map', default=DEFAULT_MAP)
    ap.add_argument('--landmarks', default=os.path.join(os.path.dirname(__file__), '..', 'config',
                                                        'landmarks_tb3_world.yaml'))
    args = ap.parse_args()
    out = args.run_dir
    img, extent = load_map(args.map)
    lmap = LandmarkMap.from_yaml(args.landmarks)
    with open(os.path.join(out, 'run.json')) as f:
        runs = json.load(f)

    # 1. landmark map ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    draw_map(ax, img, extent)
    for lm in lmap.landmarks:
        ax.plot(lm.x, lm.y, marker='*', ms=16, color='tab:orange', mec='k')
        ax.annotate('', xy=(lm.x + 0.35 * math.cos(lm.yaw), lm.y + 0.35 * math.sin(lm.yaw)), xytext=(lm.x, lm.y),
                    arrowprops=dict(arrowstyle='->', color='tab:orange', lw=1.5))
        ax.text(lm.x + 0.08, lm.y + 0.12, lm.name, fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85, lw=0))
    ax.set_title('Labeled landmarks (config/landmarks_tb3_world.yaml)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'map_landmarks.png'))
    plt.close(fig)

    # 2. trajectories -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7.5), dpi=110)
    draw_map(ax, img, extent)
    cmap = plt.get_cmap('tab10')
    for r in runs:
        tr = np.array([p for p in r['traj'] if p])
        c = cmap(r['idx'] % 10)
        label = f"{r['idx']}: {r['command']}  [{r['kind']} → {r['outcome']}]"
        if len(tr) > 1:
            ax.plot(tr[:, 0], tr[:, 1], '-', color=c, lw=2.2, label=label)
            ax.plot(tr[0, 0], tr[0, 1], 'o', color=c, ms=6, mec='k')
        else:
            ax.plot([], [], '-', color=c, lw=2.2, label=label)
        if r.get('goal'):
            ax.plot(*r['goal'], marker='X', ms=12, color=c, mec='k')
            ax.text(r['goal'][0] + 0.1, r['goal'][1] - 0.2, str(r['idx']), fontsize=10, weight='bold', color=c)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fontsize=8, ncol=1, frameon=False)
    ax.set_title('Command sequence executed by Nav2 (o = start, X = grounded goal)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'demo_trajectories.png'), bbox_inches='tight')
    plt.close(fig)

    # 3. visual target -------------------------------------------------------------------
    vis = next((r for r in runs if r.get('pixel')), None)
    frame_path = os.path.join(out, 'frames', f"{vis['idx']:02d}.jpg") if vis else None
    if vis and os.path.exists(frame_path):
        fr = plt.imread(frame_path)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=110, gridspec_kw={'width_ratios': [1.1, 1]})
        a1.imshow(fr)
        u, v = vis['pixel']
        a1.plot(u, v, 'o', ms=14, mfc='none', mec='lime', mew=3)
        a1.plot(u, v, '+', ms=14, color='lime', mew=2)
        a1.set_title(f'Camera view at command time - target pixel ({u}, {v})')
        a1.axis('off')
        draw_map(a2, img, extent)
        if vis.get('start'):
            sx, sy, syaw = vis['start']
            a2.plot(sx, sy, 'o', ms=10, color='tab:blue', mec='k', label='robot')
            a2.annotate('', xy=(sx + 0.4 * math.cos(syaw), sy + 0.4 * math.sin(syaw)), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle='->', color='tab:blue', lw=2))
            for s in (-1, 1):  # depth camera horizontal FOV: Gazebo default 1.047 rad (60 deg)
                a = syaw + s * math.radians(30)
                a2.plot([sx, sx + 2.5 * math.cos(a)], [sy, sy + 2.5 * math.sin(a)], '--', color='tab:blue', lw=1)
        if vis.get('goal'):
            a2.plot(*vis['goal'], marker='X', ms=14, color='lime', mec='k', label='projected goal')
        tr = np.array([p for p in vis['traj'] if p])
        if len(tr) > 1:
            a2.plot(tr[:, 0], tr[:, 1], '-', color='tab:green', lw=2, label='path')
        a2.legend(loc='lower right', fontsize=8)
        a2.set_title('Pixel + depth → camera ray → TF → map goal (0.45 m stand-off)')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'visual_target.png'))
        plt.close(fig)

    # 4. results table -------------------------------------------------------------------
    lines = ['| # | Command | Grounded as | Target | Nav2 result | Final error (m) | Time (s) |',
             '|---|---|---|---|---|---|---|']
    for r in runs:
        tgt = r['landmark'] or (f"pixel {tuple(r['pixel'])}" if r.get('pixel') else '')
        if r['kind'] == 'relative':
            tgt = 'relative move'
        if r['kind'] == 'none':
            tgt = f"_{r['reason']}_"
        err = '-'
        if r.get('goal') and r.get('end'):
            err = f"{math.hypot(r['end'][0] - r['goal'][0], r['end'][1] - r['goal'][1]):.2f}"
        lines.append(f"| {r['idx']} | {r['command']} | {r['kind']} | {tgt} | {r['outcome']} | {err} | {r['nav_time_s']} |")
    with open(os.path.join(out, 'results.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'figures written to {out}/')


if __name__ == '__main__':
    main()
