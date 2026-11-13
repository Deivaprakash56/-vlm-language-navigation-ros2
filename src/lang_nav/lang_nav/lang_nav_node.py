"""ROS 2 node: text command -> grounded target -> Nav2 NavigateToPose goal.

Topics
  in : ~/command (std_msgs/String)              e.g. "go to the north end"
       <rgb_topic>, <depth_topic>, <camera_info_topic>
  out: ~/status (std_msgs/String)               JSON status events
       ~/goal_marker (visualization_msgs/Marker)
Action client: navigate_to_pose (nav2_msgs/NavigateToPose)
"""
import json
import math
import os
import threading
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform support)
from tf2_ros import Buffer, TransformListener

from lang_nav.grounding import GroundingResult, LandmarkMap, make_grounder, yaw_to_quat


def _yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class LangNavNode(Node):
    def __init__(self):
        super().__init__('lang_nav')
        default_lm = os.path.join(get_package_share_directory('lang_nav'), 'config', 'landmarks_tb3_world.yaml')
        p = self.declare_parameter
        p('landmarks_file', default_lm)
        p('backend', 'claude')                  # 'claude' | 'keyword'
        p('model', 'claude-opus-5-5')
        p('effort', 'low')                      # low keeps latency down for interactive commands
        p('use_camera', True)
        p('rgb_topic', '/intel_realsense_r200_depth/image_raw')
        p('depth_topic', '/intel_realsense_r200_depth/depth/image_raw')
        p('camera_info_topic', '/intel_realsense_r200_depth/camera_info')
        p('global_frame', 'map')
        p('robot_frame', 'base_footprint')
        p('standoff_m', 0.45)                   # stop this far short of image targets
        p('depth_scale', 1.0)                   # 0.001 for 16UC1 millimetre depth (real RealSense)
        p('max_image_age_s', 2.0)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.lmap = LandmarkMap.from_yaml(gp('landmarks_file'))
        backend = gp('backend')
        kwargs = {'model': gp('model'), 'effort': gp('effort')} if backend == 'claude' else {}
        self.grounder = make_grounder(backend, self.lmap, **kwargs)
        self.global_frame = gp('global_frame')
        self.robot_frame = gp('robot_frame')
        self.standoff = float(gp('standoff_m'))
        self.depth_scale = float(gp('depth_scale'))
        self.max_image_age = float(gp('max_image_age_s'))

        self.cb = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._lock = threading.Lock()
        self._rgb = self._depth = self._info = None
        self._busy = False
        self._goal_handle = None

        if gp('use_camera'):
            self.create_subscription(Image, gp('rgb_topic'), self._on_rgb, qos_profile_sensor_data)
            self.create_subscription(Image, gp('depth_topic'), self._on_depth, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, gp('camera_info_topic'), self._on_info, qos_profile_sensor_data)
        self.create_subscription(String, '~/command', self._on_command, 10, callback_group=self.cb)
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.marker_pub = self.create_publisher(Marker, '~/goal_marker', 1)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose', callback_group=self.cb)

        self.get_logger().info(
            f'lang_nav ready: backend={backend}, {len(self.lmap.landmarks)} landmarks, '
            f'send commands on {self.resolve_topic_name("~/command")}')

    # ------------------------------------------------------------------ sensors
    def _on_rgb(self, msg):
        with self._lock:
            self._rgb = msg

    def _on_depth(self, msg):
        with self._lock:
            self._depth = msg

    def _on_info(self, msg):
        with self._lock:
            self._info = msg

    def _fresh(self, msg) -> bool:
        if msg is None:
            return False
        age = (self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9
        return age < self.max_image_age

    # ------------------------------------------------------------------ status
    def _status(self, event: str, **kw):
        kw['event'] = event
        self.status_pub.publish(String(data=json.dumps(kw)))
        self.get_logger().info(f'[{event}] ' + ', '.join(f'{k}={v}' for k, v in kw.items() if k != 'event'))

    # ------------------------------------------------------------------ command
    def _on_command(self, msg: String):
        text = msg.data.strip()
        if not text:
            return
        if text.lower() in ('stop', 'cancel', 'halt'):
            self._cancel_active()
            self._status('cancelled', command=text)
            return
        if self._busy:
            self._status('rejected', command=text, reason='still grounding previous command')
            return
        self._busy = True
        threading.Thread(target=self._handle_command, args=(text,), daemon=True).start()

    def _handle_command(self, text: str):
        try:
            with self._lock:
                rgb, depth, info = self._rgb, self._depth, self._info
            image = None
            if self._fresh(rgb):
                image = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
            self._status('grounding', command=text, has_image=image is not None)
            t0 = time.monotonic()
            result = self.grounder.ground(text, image)
            wh = (image.shape[1], image.shape[0]) if image is not None else None
            result = result.validate(self.lmap, wh)
            self._status('grounded', command=text, kind=result.kind, landmark=result.landmark,
                         pixel=[result.pixel_u, result.pixel_v], reason=result.reason,
                         latency_s=round(time.monotonic() - t0, 2))
            if result.kind == 'none':
                return
            goal = self._result_to_pose(result, depth, info, rgb)
            if goal is None:
                return
            self._send_goal(goal, text)
        except Exception as e:  # keep the node alive on API/network errors
            self._status('error', command=text, error=f'{type(e).__name__}: {e}')
        finally:
            self._busy = False

    # ------------------------------------------------------------------ geometry
    def _robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.global_frame, self.robot_frame, Time(),
                                                 timeout=Duration(seconds=1.0))
        except Exception as e:
            self._status('error', reason=f'TF {self.global_frame}->{self.robot_frame} unavailable: {e}')
            return None
        t = tf.transform.translation
        return t.x, t.y, _yaw_from_quat(tf.transform.rotation)

    def _pose(self, x, y, yaw) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = self.global_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        qx, qy, qz, qw = yaw_to_quat(yaw)
        ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        return ps

    def _result_to_pose(self, r: GroundingResult, depth, info, rgb):
        if r.kind == 'landmark':
            lm = self.lmap.get(r.landmark)
            return self._pose(lm.x, lm.y, lm.yaw)

        if r.kind == 'relative':
            rp = self._robot_pose()
            if rp is None:
                return None
            x, y, yaw = rp
            gx = x + r.forward_m * math.cos(yaw) - r.left_m * math.sin(yaw)
            gy = y + r.forward_m * math.sin(yaw) + r.left_m * math.cos(yaw)
            return self._pose(gx, gy, yaw + math.radians(r.turn_deg))

        if r.kind == 'image_point':
            return self._pixel_to_pose(r.pixel_u, r.pixel_v, depth, info, rgb)
        return None

    def _pixel_to_pose(self, u, v, depth_msg, info, rgb_msg):
        if depth_msg is None or info is None:
            self._status('error', reason='image_point target but no depth/camera_info received')
            return None
        d = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)
        d = d * self.depth_scale
        # RGB and depth may differ in resolution; rescale the pixel.
        if rgb_msg is not None and (rgb_msg.width, rgb_msg.height) != (d.shape[1], d.shape[0]):
            u = int(u * d.shape[1] / rgb_msg.width)
            v = int(v * d.shape[0] / rgb_msg.height)
        r = 4
        win = d[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
        valid = win[np.isfinite(win) & (win > 0.05)]
        if valid.size == 0:
            self._status('error', reason=f'no valid depth around pixel ({u},{v})')
            return None
        z = float(np.median(valid))
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        pt = PointStamped()
        pt.header = depth_msg.header  # optical frame: x right, y down, z forward
        pt.point.x = (u - cx) * z / fx
        pt.point.y = (v - cy) * z / fy
        pt.point.z = z
        try:
            p_map = self.tf_buffer.transform(pt, self.global_frame, timeout=Duration(seconds=1.0))
        except Exception as e:
            self._status('error', reason=f'cannot transform camera point to {self.global_frame}: {e}')
            return None
        rp = self._robot_pose()
        if rp is None:
            return None
        rx, ry, _ = rp
        tx, ty = p_map.point.x, p_map.point.y
        self._status('projected', u=u, v=v, depth_m=round(z, 3), target_x=round(tx, 3), target_y=round(ty, 3),
                     target_z=round(p_map.point.z, 3))
        heading = math.atan2(ty - ry, tx - rx)
        dist = math.hypot(tx - rx, ty - ry)
        back = min(self.standoff, max(dist - 0.1, 0.0))
        return self._pose(tx - back * math.cos(heading), ty - back * math.sin(heading), heading)

    # ------------------------------------------------------------------ nav2
    def _publish_marker(self, goal: PoseStamped):
        m = Marker()
        m.header = goal.header
        m.ns, m.id, m.type, m.action = 'lang_nav', 0, Marker.ARROW, Marker.ADD
        m.pose = goal.pose
        m.scale.x, m.scale.y, m.scale.z = 0.5, 0.08, 0.08
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.8, 0.2, 1.0
        self.marker_pub.publish(m)

    def _cancel_active(self):
        gh = self._goal_handle
        if gh is not None:
            gh.cancel_goal_async()
            self._goal_handle = None

    def _send_goal(self, goal: PoseStamped, text: str):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self._status('error', reason='navigate_to_pose action server not available')
            return
        self._cancel_active()
        self._publish_marker(goal)
        req = NavigateToPose.Goal()
        req.pose = goal
        self._status('goal_sent', command=text, x=round(goal.pose.position.x, 2),
                     y=round(goal.pose.position.y, 2))
        fut = self.nav_client.send_goal_async(req)
        fut.add_done_callback(lambda f: self._on_goal_response(f, text))

    def _on_goal_response(self, fut, text):
        gh = fut.result()
        if not gh.accepted:
            self._status('goal_rejected', command=text)
            return
        self._goal_handle = gh
        gh.get_result_async().add_done_callback(lambda f: self._on_result(f, text, gh))

    def _on_result(self, fut, text, gh):
        status = fut.result().status
        names = {GoalStatus.STATUS_SUCCEEDED: 'succeeded', GoalStatus.STATUS_ABORTED: 'aborted',
                 GoalStatus.STATUS_CANCELED: 'canceled'}
        self._status('goal_' + names.get(status, f'status_{status}'), command=text)
        if self._goal_handle is gh:
            self._goal_handle = None


def main():
    rclpy.init()
    node = LangNavNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
