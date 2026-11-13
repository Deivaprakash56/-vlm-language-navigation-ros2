"""Scripted end-to-end demo / evaluation of lang_nav in Gazebo.

Sends a list of commands to /lang_nav/command one after another, waits for each to finish,
and records: status events, robot trajectory (TF map->base_footprint), the camera frame seen
at command time and the Nav2 goal. Everything is written to <out>/run.json + <out>/frames/.
Then `python -m lang_nav.plot_demo <out>` renders the README figures.

  ros2 run lang_nav demo_eval --out docs/lang_nav/run1
  ros2 run lang_nav demo_eval --out docs/lang_nav/run1 --commands "go home" "turn around"

The special command "@visual" picks the obstacle closest to the image centre (within 2.5 m) and
sends "@pixel u v" (what a VLM would answer for "go to the pillar in front of you") - this tests the
depth back-projection path without an API key.
"""
import argparse
import json
import math
import os
import threading
import time

import numpy as np

DEFAULT_COMMANDS = [
    '@visual',
    'go to the north end',
    'drive to the north east corner',
    'move forward 0.5 m then turn around',
    'head to the south west corner',
    'go to the south end',
    'take me back home',
    'make me a sandwich',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='docs/lang_nav/run1')
    ap.add_argument('--commands', nargs='+', default=DEFAULT_COMMANDS)
    ap.add_argument('--timeout', type=float, default=90.0)
    ap.add_argument('--vlm', action='store_true',
                    help='send "@visual" as real text so the VLM picks the pixel itself')
    args = ap.parse_args()

    import cv2
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.duration import Duration
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener
    from visualization_msgs.msg import Marker

    os.makedirs(os.path.join(args.out, 'frames'), exist_ok=True)
    rclpy.init()
    node = rclpy.create_node('lang_nav_demo', parameter_overrides=[
        rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
    bridge = CvBridge()
    tfb = Buffer()
    TransformListener(tfb, node)
    lock = threading.Lock()
    st = {'rgb': None, 'depth': None, 'events': [], 'marker': None}

    def on_rgb(m):
        with lock:
            st['rgb'] = m

    def on_depth(m):
        with lock:
            st['depth'] = m

    def on_status(m):
        with lock:
            st['events'].append(json.loads(m.data))

    def on_marker(m):
        with lock:
            st['marker'] = (m.pose.position.x, m.pose.position.y)

    node.create_subscription(Image, '/intel_realsense_r200_depth/image_raw', on_rgb, qos_profile_sensor_data)
    node.create_subscription(Image, '/intel_realsense_r200_depth/depth/image_raw', on_depth, qos_profile_sensor_data)
    node.create_subscription(String, '/lang_nav/status', on_status, 50)
    node.create_subscription(Marker, '/lang_nav/goal_marker', on_marker, 10)
    pub = node.create_publisher(String, '/lang_nav/command', 10)
    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def pose():
        try:
            tf = tfb.lookup_transform('map', 'base_footprint', Time(), timeout=Duration(seconds=0.3))
            q = tf.transform.rotation
            return (tf.transform.translation.x, tf.transform.translation.y,
                    math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2)))
        except Exception:
            return None

    time.sleep(3.0)
    while pub.get_subscription_count() == 0 or st['rgb'] is None or st['depth'] is None or pose() is None:
        print('waiting for lang_nav node, camera and localisation...', flush=True)
        time.sleep(1.0)

    results = []
    for i, cmd in enumerate(args.commands):
        with lock:
            rgb, depth = st['rgb'], st['depth']
            st['events'].clear()
            st['marker'] = None
        frame = bridge.imgmsg_to_cv2(rgb, 'bgr8') if rgb is not None else None
        sent = cmd
        pixel = None
        if cmd == '@visual' and args.vlm:
            sent = 'go to the pillar in front of you'
        elif cmd == '@visual':
            d = bridge.imgmsg_to_cv2(depth, 'passthrough').astype(np.float32)
            h, w = d.shape
            # rows just above the optical centre only see obstacles (camera is level, ~0.12 m high),
            # so the nearest return there is the nearest obstacle, never the floor
            band = d[h // 2 - 12:h // 2 - 2, :]
            col_depth = np.where(np.isfinite(band) & (band > 0.2), band, np.inf).min(axis=0)
            # obstacles (not far walls) within 2.5 m; take the one closest to the image centre,
            # i.e. what a VLM would pick for "go to the pillar in front of you"
            cand = np.where(col_depth < 2.5)[0]
            if len(cand) == 0:
                cand = np.array([int(np.argmin(col_depth))])
            col = int(cand[np.argmin(np.abs(cand - w / 2))])
            pixel = (col, h // 2 - 7)
            sent = f'@pixel {pixel[0]} {pixel[1]}'
        if frame is not None:
            cv2.imwrite(os.path.join(args.out, 'frames', f'{i:02d}.jpg'), frame)

        start = pose()
        traj = [start] if start else []
        t0 = time.monotonic()
        pub.publish(String(data=sent))
        print(f'[{i}] -> {sent}')
        outcome = 'timeout'
        while time.monotonic() - t0 < args.timeout:
            p = pose()
            if p:
                traj.append(p)
            with lock:
                evs = list(st['events'])
            names = [e['event'] for e in evs]
            if any(n in ('goal_succeeded', 'goal_aborted', 'goal_canceled', 'goal_rejected', 'error') for n in names):
                outcome = next(n for n in names if n.startswith('goal_') and n != 'goal_sent' or n == 'error')
                break
            if 'grounded' in names and next(e for e in evs if e['event'] == 'grounded')['kind'] == 'none':
                outcome = 'no_target'
                break
            time.sleep(0.2)
        time.sleep(0.5)
        with lock:
            evs = list(st['events'])
            goal = st['marker']
        grounded = next((e for e in evs if e['event'] == 'grounded'), {})
        res = dict(idx=i, command=('go to the pillar in front of you *' if cmd == '@visual' else cmd), sent=sent, kind=grounded.get('kind'), landmark=grounded.get('landmark'),
                   reason=grounded.get('reason'), grounding_latency_s=grounded.get('latency_s'),
                   outcome=outcome, nav_time_s=round(time.monotonic() - t0, 1), goal=goal,
                   start=start, end=pose(), pixel=pixel, traj=traj[::3], events=evs)
        results.append(res)
        print(f'    kind={res["kind"]} outcome={outcome} time={res["nav_time_s"]}s goal={goal}')
        with open(os.path.join(args.out, 'run.json'), 'w') as f:
            json.dump(results, f, indent=1)
        time.sleep(1.0)
    executor.shutdown()
    spin_thread.join(timeout=2.0)
    node.destroy_node()
    rclpy.try_shutdown()
    print(f'wrote {args.out}/run.json')


if __name__ == '__main__':
    main()
