"""Command-line helper.

  ros2 run lang_nav lang_nav_cli                     # interactive: type commands, see status
  ros2 run lang_nav lang_nav_cli --offline "go to the north end" [--image img.jpg] [--backend keyword]
"""
import argparse
import os
import sys


def offline(args):
    from lang_nav.grounding import LandmarkMap, make_grounder
    lm_file = args.landmarks
    if lm_file is None:
        from ament_index_python.packages import get_package_share_directory
        lm_file = os.path.join(get_package_share_directory('lang_nav'), 'config', 'landmarks_tb3_world.yaml')
    lmap = LandmarkMap.from_yaml(lm_file)
    image = None
    if args.image:
        import cv2
        image = cv2.imread(args.image)
        if image is None:
            sys.exit(f'cannot read image {args.image}')
    g = make_grounder(args.backend, lmap)
    r = g.ground(args.offline, image)
    wh = (image.shape[1], image.shape[0]) if image is not None else None
    print(r.validate(lmap, wh))


def interactive():
    import rclpy
    from std_msgs.msg import String
    rclpy.init()
    node = rclpy.create_node('lang_nav_cli')
    pub = node.create_publisher(String, '/lang_nav/command', 10)
    node.create_subscription(String, '/lang_nav/status', lambda m: print(f'  <- {m.data}'), 10)
    import threading
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print('Type a command ("stop" cancels, Ctrl-D quits).')
    try:
        for line in sys.stdin:
            if line.strip():
                pub.publish(String(data=line.strip()))
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--offline', metavar='COMMAND', help='ground one command without ROS and print it')
    ap.add_argument('--image', help='camera image for --offline')
    ap.add_argument('--backend', default='claude', choices=['claude', 'keyword'])
    ap.add_argument('--landmarks', help='landmarks YAML (default: package config)')
    args = ap.parse_args()
    offline(args) if args.offline else interactive()


if __name__ == '__main__':
    main()
