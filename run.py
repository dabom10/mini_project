#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading
from collections import deque

import cv2
import numpy as np
import rclpy

from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage
from tf2_ros import Buffer, TransformListener
from turtlebot4_navigation.turtlebot4_navigator import (
    TurtleBot4Directions,
    TurtleBot4Navigator,
)
from ultralytics import YOLO


# ================================
# 설정 상수
# ================================
DEPTH_HEADER_BYTES = 12

MIN_VALID_DEPTH_M = 0.2
MAX_VALID_DEPTH_M = 5.0
REMEASURE_MAX_DISTANCE_M = 3.0

GUI_WINDOW_NAME = "RGB Detection (left) | Depth (right)"
GUI_WIDTH = 1280
GUI_HEIGHT = 480

NAV_START_DELAY_SEC = 1.0
TF_START_DELAY_SEC = 5.0
DISPLAY_PERIOD_SEC = 0.05

STOP_DISTANCE_M = 0.4
MIN_MOVE_DISTANCE_M = 0.05

MODEL_PATH = "/home/kyb/rokey_ws/src/rokey_pjt/rokey_pjt/robotv8n(e:100,b:32,p:20).pt"
TARGET_CLASS_NAME = "car"
CONF_THRESHOLD = 0.25
MEASURE_CONF_THRESHOLD = 0.80
DEVICE = "cpu"   # GPU 사용 시 "0"

DETECTION_START_DELAY_SEC = 2.0
YOLO_PERIOD_SEC = 0.12

DEPTH_SAMPLE_COUNT = 10
DEPTH_KERNEL_SIZE = 5

DEPTH_BUFFER_SIZE = 30
MAX_SYNC_DIFF_SEC = 0.20

OUTLIER_MIN_KEEP_COUNT = 4
OUTLIER_ABS_THRESH_M = 0.25
OUTLIER_MAD_SCALE = 2.5
OUTLIER_FALLBACK_THRESH_M = 0.08

DISTANCE_EMA_ALPHA = 0.6
# ================================


class DepthToMap(Node):
    def __init__(self):
        super().__init__("depth_to_map_node")

        self.lock = threading.Lock()

        self.K = None

        self.rgb_image = None
        self.rgb_stamp_sec = None

        self.depth_image = None
        self.depth_frame_id = None
        self.depth_stamp_sec = None

        self.depth_buffer = deque(maxlen=DEPTH_BUFFER_SIZE)

        self.display_image = None

        self.last_detection = None
        self.goal_in_progress = False

        self.nav_ready = False
        self.nav_init_started = False
        self.detection_enabled = False
        self.detection_enable_timer = None

        self.logged_intrinsics = False
        self.logged_rgb_shape = False
        self.logged_depth_shape = False
        self.logged_depth_dtype = False

        self.measurement_state = None
        self.ready_target = None
        self.filtered_distance_ema = None

        ns = self.get_namespace()

        self.depth_topic = f"{ns}/oakd/stereo/image_raw/compressedDepth"
        self.rgb_topic = f"{ns}/oakd/rgb/image_raw/compressed"
        self.info_topic = f"{ns}/oakd/rgb/camera_info"

        self.base_frame = "base_link"

        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"CameraInfo topic: {self.info_topic}")
        self.get_logger().info(f"Base frame: {self.base_frame}")
        self.get_logger().info(f"YOLO model: {MODEL_PATH}")
        self.get_logger().info(f"Target class: {TARGET_CLASS_NAME}")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.model = YOLO(MODEL_PATH)
        self.get_logger().info("YOLO model loaded successfully.")

        self.create_subscription(
            CameraInfo,
            self.info_topic,
            self.camera_info_callback,
            1,
        )
        self.create_subscription(
            CompressedImage,
            self.depth_topic,
            self.depth_callback,
            10,
        )
        self.create_subscription(
            CompressedImage,
            self.rgb_topic,
            self.rgb_callback,
            10,
        )

        self.gui_thread_stop = threading.Event()
        self.gui_thread = threading.Thread(target=self.gui_loop, daemon=True)
        self.gui_thread.start()

        self.navigator = TurtleBot4Navigator()

        self.get_logger().info("카메라/GUI 먼저 시작합니다.")
        self.nav_start_timer = self.create_timer(
            NAV_START_DELAY_SEC,
            self.init_navigation_once,
        )

        self.get_logger().info("TF Tree 안정화 시작. 잠시 후 자동 타겟 처리 활성화합니다.")
        self.tf_start_timer = self.create_timer(
            TF_START_DELAY_SEC,
            self.start_transform,
        )

    def stamp_to_sec(self, stamp_msg):
        return float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9

    def init_navigation_once(self):
        if self.nav_init_started:
            return

        self.nav_init_started = True
        self.nav_start_timer.cancel()

        try:
            self.get_logger().info("Checking dock status...")
            if not self.navigator.getDockedStatus():
                self.get_logger().info("Robot is undocked. Docking now...")
                self.navigator.dock()
                self.get_logger().info("Dock complete.")
            else:
                self.get_logger().info("Robot is already docked.")

            initial_pose = self.navigator.getPoseStamped(
                [0.0, 0.0],
                TurtleBot4Directions.NORTH,
            )

            self.get_logger().info("Setting initial pose...")
            self.navigator.setInitialPose(initial_pose)

            self.get_logger().info("Waiting for Nav2 to become active...")
            self.navigator.waitUntilNav2Active()
            self.get_logger().info("Nav2 is active.")

            self.get_logger().info("Undocking...")
            self.navigator.undock()
            self.get_logger().info("Undock complete.")

            self.nav_ready = True

            self.get_logger().info(
                f"Object detection will start in {DETECTION_START_DELAY_SEC:.1f} seconds..."
            )
            self.detection_enable_timer = self.create_timer(
                DETECTION_START_DELAY_SEC,
                self.enable_detection,
            )

        except Exception as e:
            self.get_logger().error(f"Navigation init failed: {e}")
            self.nav_ready = False

    def enable_detection(self):
        if self.detection_enabled:
            return

        self.detection_enabled = True
        self.get_logger().info("YOLO detection enabled.")

        if self.detection_enable_timer is not None:
            self.detection_enable_timer.cancel()

    def start_transform(self):
        self.get_logger().info("TF Tree 안정화 완료. 자동 타겟 처리 시작합니다.")
        self.display_timer = self.create_timer(DISPLAY_PERIOD_SEC, self.display_images)
        self.yolo_timer = self.create_timer(YOLO_PERIOD_SEC, self.run_detection_cycle)
        self.auto_goal_timer = self.create_timer(0.1, self.process_detection_goal)
        self.tf_start_timer.cancel()

    def camera_info_callback(self, msg: CameraInfo):
        with self.lock:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

        if not self.logged_intrinsics:
            self.get_logger().info(
                f"Camera intrinsics received: "
                f"fx={self.K[0, 0]:.2f}, fy={self.K[1, 1]:.2f}, "
                f"cx={self.K[0, 2]:.2f}, cy={self.K[1, 2]:.2f}"
            )
            self.logged_intrinsics = True

    def decode_compressed_depth(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)

        depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if depth is not None and depth.size > 0:
            return depth

        if len(np_arr) > DEPTH_HEADER_BYTES:
            depth = cv2.imdecode(np_arr[DEPTH_HEADER_BYTES:], cv2.IMREAD_UNCHANGED)
            if depth is not None and depth.size > 0:
                return depth

        return None

    def depth_callback(self, msg: CompressedImage):
        try:
            depth = self.decode_compressed_depth(msg)

            if depth is None:
                self.get_logger().error("Compressed depth decode failed: imdecode returned None")
                return

            depth_stamp_sec = self.stamp_to_sec(msg.header.stamp)

            if not self.logged_depth_shape:
                self.get_logger().info(f"Depth image received: {depth.shape}")
                self.logged_depth_shape = True

            if not self.logged_depth_dtype:
                self.get_logger().info(f"Depth dtype: {depth.dtype}")
                self.logged_depth_dtype = True

            with self.lock:
                self.depth_image = depth.copy()
                self.depth_frame_id = msg.header.frame_id
                self.depth_stamp_sec = depth_stamp_sec
                self.depth_buffer.append(
                    {
                        "stamp_sec": depth_stamp_sec,
                        "frame_id": msg.header.frame_id,
                        "depth": depth.copy(),
                    }
                )

            self.update_measurement_with_new_depth(depth, depth_stamp_sec)

        except Exception as e:
            self.get_logger().error(f"Compressed depth decode failed: {e}")

    def rgb_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if rgb is None or rgb.size == 0:
                self.get_logger().error("Compressed RGB decode failed: imdecode returned None")
                return

            rgb_stamp_sec = self.stamp_to_sec(msg.header.stamp)

            if not self.logged_rgb_shape:
                self.get_logger().info(f"RGB image decoded: {rgb.shape}")
                self.logged_rgb_shape = True

            with self.lock:
                self.rgb_image = rgb.copy()
                self.rgb_stamp_sec = rgb_stamp_sec

        except Exception as e:
            self.get_logger().error(f"Compressed RGB decode failed: {e}")

    def run_detection_cycle(self):
        if not self.detection_enabled:
            return

        with self.lock:
            frame = self.rgb_image.copy() if self.rgb_image is not None else None
            rgb_stamp_sec = self.rgb_stamp_sec
            goal_in_progress = self.goal_in_progress
            measurement_active = self.measurement_state is not None

        if frame is None or rgb_stamp_sec is None:
            return

        det = self.run_yolo_on_frame(frame)

        with self.lock:
            self.last_detection = det

        if det is None:
            return

        if det["conf"] < MEASURE_CONF_THRESHOLD:
            return

        if goal_in_progress or measurement_active:
            return

        self.start_depth_measurement(det, frame.shape[:2], rgb_stamp_sec)

    def run_yolo_on_frame(self, frame):
        best_det = None

        results = self.model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            verbose=False,
            device=DEVICE,
        )

        if not results:
            return None

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return None

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = self.model.names.get(cls_id, str(cls_id))

            if class_name != TARGET_CLASS_NAME:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            area = max(0, x2 - x1) * max(0, y2 - y1)

            det = {
                "class_name": class_name,
                "conf": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy,
                "area": area,
            }

            if best_det is None or det["area"] > best_det["area"]:
                best_det = det

        return best_det

    def start_depth_measurement(self, det, rgb_shape, rgb_stamp_sec):
        nearest = self.find_nearest_depth_frame(rgb_stamp_sec)

        with self.lock:
            self.measurement_state = {
                "class_name": det["class_name"],
                "conf": det["conf"],
                "x1": det["x1"],
                "y1": det["y1"],
                "x2": det["x2"],
                "y2": det["y2"],
                "x_rgb": det["cx"],
                "y_rgb": det["cy"],
                "rgb_shape": rgb_shape,
                "rgb_stamp_sec": rgb_stamp_sec,
                "samples_m": [],
                "sample_sync_diffs": [],
                "used_depth_stamps": set(),
            }

        if nearest is not None:
            sync_diff = abs(nearest["stamp_sec"] - rgb_stamp_sec)
            if sync_diff <= MAX_SYNC_DIFF_SEC:
                self.add_depth_sample_to_measurement(
                    nearest["depth"],
                    nearest["stamp_sec"],
                    rgb_stamp_sec,
                )
                self.get_logger().info(
                    f"Measurement started: conf={det['conf']:.2f}, "
                    f"center=({det['cx']}, {det['cy']}), "
                    f"nearest depth sync diff={sync_diff:.3f} sec"
                )
            else:
                self.get_logger().warn(
                    f"Nearest depth sync diff too large: {sync_diff:.3f} sec. "
                    f"Waiting for newer depth frames."
                )
        else:
            self.get_logger().info(
                f"Measurement started: conf={det['conf']:.2f}, "
                f"center=({det['cx']}, {det['cy']}), "
                f"waiting for depth frames"
            )

    def find_nearest_depth_frame(self, rgb_stamp_sec):
        with self.lock:
            if not self.depth_buffer:
                return None
            buffer_copy = list(self.depth_buffer)

        best_item = None
        best_diff = None

        for item in buffer_copy:
            diff = abs(item["stamp_sec"] - rgb_stamp_sec)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_item = item

        return best_item

    def update_measurement_with_new_depth(self, depth, depth_stamp_sec):
        with self.lock:
            if self.measurement_state is None:
                return
            rgb_stamp_sec = self.measurement_state["rgb_stamp_sec"]

        self.add_depth_sample_to_measurement(depth, depth_stamp_sec, rgb_stamp_sec)

    def add_depth_sample_to_measurement(self, depth, depth_stamp_sec, rgb_stamp_sec):
        with self.lock:
            if self.measurement_state is None:
                return

            if len(self.measurement_state["samples_m"]) >= DEPTH_SAMPLE_COUNT:
                return

            if depth_stamp_sec in self.measurement_state["used_depth_stamps"]:
                return

            x_rgb = self.measurement_state["x_rgb"]
            y_rgb = self.measurement_state["y_rgb"]
            rgb_shape = self.measurement_state["rgb_shape"]

        depth_shape = depth.shape[:2]
        depth_pixel = self.get_scaled_pixel(x_rgb, y_rgb, rgb_shape, depth_shape)
        if depth_pixel is None:
            return

        x_depth, y_depth = depth_pixel

        if not (0 <= x_depth < depth.shape[1] and 0 <= y_depth < depth.shape[0]):
            return

        raw_depth = self.get_depth_from_center(
            depth,
            x_depth,
            y_depth,
            kernel_size=DEPTH_KERNEL_SIZE,
        )
        z = self.get_depth_in_meters(raw_depth)

        sync_diff = abs(depth_stamp_sec - rgb_stamp_sec)

        with self.lock:
            if self.measurement_state is None:
                return

            self.measurement_state["used_depth_stamps"].add(depth_stamp_sec)

            if MIN_VALID_DEPTH_M < z < MAX_VALID_DEPTH_M:
                self.measurement_state["samples_m"].append(float(z))
                self.measurement_state["sample_sync_diffs"].append(sync_diff)

            sample_count = len(self.measurement_state["samples_m"])

        if sample_count < DEPTH_SAMPLE_COUNT:
            return

        self.finalize_measurement()

    def finalize_measurement(self):
        with self.lock:
            if self.measurement_state is None:
                return

            state = dict(self.measurement_state)
            samples = list(self.measurement_state["samples_m"])
            sync_diffs = list(self.measurement_state["sample_sync_diffs"])

        if len(samples) < OUTLIER_MIN_KEEP_COUNT:
            self.get_logger().warn(
                f"Measurement failed: valid depth samples too few ({len(samples)}). Retrying."
            )
            with self.lock:
                self.measurement_state = None
            return

        filtered_samples = self.remove_depth_outliers(samples)

        if len(filtered_samples) < OUTLIER_MIN_KEEP_COUNT:
            self.get_logger().warn(
                f"Measurement failed after outlier removal: kept {len(filtered_samples)} samples. Retrying."
            )
            with self.lock:
                self.measurement_state = None
            return

        avg_distance = float(np.mean(filtered_samples))

        if self.filtered_distance_ema is None:
            stable_distance = avg_distance
        else:
            stable_distance = (
                DISTANCE_EMA_ALPHA * avg_distance
                + (1.0 - DISTANCE_EMA_ALPHA) * self.filtered_distance_ema
            )

        self.filtered_distance_ema = stable_distance

        if stable_distance > REMEASURE_MAX_DISTANCE_M:
            self.get_logger().warn(
                f"Measured stable distance {stable_distance:.2f} m > "
                f"{REMEASURE_MAX_DISTANCE_M:.2f} m. Retrying."
            )
            with self.lock:
                self.measurement_state = None
            return

        mean_sync_diff = float(np.mean(sync_diffs)) if sync_diffs else 0.0
        max_sync_diff = float(np.max(sync_diffs)) if sync_diffs else 0.0

        with self.lock:
            self.ready_target = {
                "class_name": state["class_name"],
                "conf": state["conf"],
                "x1": state["x1"],
                "y1": state["y1"],
                "x2": state["x2"],
                "y2": state["y2"],
                "x_rgb": state["x_rgb"],
                "y_rgb": state["y_rgb"],
                "z_raw_avg": avg_distance,
                "z_stable": stable_distance,
            }
            self.measurement_state = None

        self.get_logger().info(
            f"Measurement ready: center=({state['x_rgb']}, {state['y_rgb']}), "
            f"raw_samples={len(samples)}, filtered_samples={len(filtered_samples)}, "
            f"avg={avg_distance:.2f} m, stable={stable_distance:.2f} m, "
            f"sync_mean={mean_sync_diff:.3f} sec, sync_max={max_sync_diff:.3f} sec"
        )

    def remove_depth_outliers(self, samples_m):
        arr = np.array(samples_m, dtype=np.float64)

        if arr.size == 0:
            return []

        median = float(np.median(arr))
        abs_diff = np.abs(arr - median)
        mad = float(np.median(abs_diff))

        if mad < 1e-6:
            keep_mask = abs_diff <= OUTLIER_ABS_THRESH_M
        else:
            robust_sigma = 1.4826 * mad
            dynamic_thresh = max(OUTLIER_FALLBACK_THRESH_M, OUTLIER_MAD_SCALE * robust_sigma)
            keep_mask = abs_diff <= dynamic_thresh

        filtered = arr[keep_mask]

        if filtered.size < OUTLIER_MIN_KEEP_COUNT:
            arr_sorted = np.sort(arr)
            trim_count = max(1, int(len(arr_sorted) * 0.1))
            if len(arr_sorted) > 2 * trim_count:
                filtered = arr_sorted[trim_count:-trim_count]
            else:
                filtered = arr_sorted

        return filtered.astype(np.float64).tolist()

    def get_depth_in_meters(self, depth_value):
        return float(depth_value) / 1000.0

    def get_scaled_pixel(self, x_src, y_src, src_shape, dst_shape):
        src_h, src_w = src_shape
        dst_h, dst_w = dst_shape

        if src_w <= 0 or src_h <= 0:
            return None

        scale_x = dst_w / float(src_w)
        scale_y = dst_h / float(src_h)

        x_dst = int(round(x_src * scale_x))
        y_dst = int(round(y_src * scale_y))

        x_dst = max(0, min(dst_w - 1, x_dst))
        y_dst = max(0, min(dst_h - 1, y_dst))

        return x_dst, y_dst

    def get_current_robot_xy(self):
        tf_map_base = self.tf_buffer.lookup_transform(
            "map",
            self.base_frame,
            Time(),
            timeout=Duration(seconds=1.0),
        )
        return tf_map_base.transform.translation.x, tf_map_base.transform.translation.y

    def make_stop_pose_before_target(self, target_x, target_y):
        robot_x, robot_y = self.get_current_robot_xy()

        dx = target_x - robot_x
        dy = target_y - robot_y
        dist = math.hypot(dx, dy)

        if dist < MIN_MOVE_DISTANCE_M:
            self.get_logger().warn(
                f"목표가 너무 가까워 이동하지 않습니다. dist={dist:.3f} m"
            )
            return None

        ux = dx / dist
        uy = dy / dist

        move_dist = dist - STOP_DISTANCE_M

        if move_dist <= 0.0:
            self.get_logger().warn(
                f"이미 목표점과 {STOP_DISTANCE_M:.2f}m 이내입니다. dist={dist:.3f} m"
            )
            return None

        if move_dist < MIN_MOVE_DISTANCE_M:
            self.get_logger().warn(
                f"이동 거리가 너무 짧아 이동하지 않습니다. move_dist={move_dist:.3f} m"
            )
            return None

        goal_x = robot_x + ux * move_dist
        goal_y = robot_y + uy * move_dist
        yaw = math.atan2(dy, dx)

        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=qz,
            w=qw,
        )

        self.get_logger().info(
            f"Robot=({robot_x:.2f}, {robot_y:.2f}), "
            f"Target=({target_x:.2f}, {target_y:.2f}), "
            f"dist={dist:.2f} m, move_dist={move_dist:.2f} m, "
            f"Goal=({goal_x:.2f}, {goal_y:.2f}), remain={STOP_DISTANCE_M:.2f} m"
        )

        return goal_pose

    def get_depth_from_center(self, depth, x, y, kernel_size=5):
        h, w = depth.shape[:2]
        half = kernel_size // 2

        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(w, x + half + 1)
        y2 = min(h, y + half + 1)

        patch = depth[y1:y2, x1:x2]
        valid = patch[patch > 0]

        if valid.size == 0:
            return 0

        return int(np.median(valid))

    def process_detection_goal(self):
        with self.lock:
            if self.goal_in_progress:
                return

            if self.ready_target is None:
                return

            K = self.K.copy() if self.K is not None else None
            frame_id = self.depth_frame_id
            target = dict(self.ready_target)

        if not self.detection_enabled:
            return

        if K is None or frame_id is None:
            return

        if not self.nav_ready:
            return

        try:
            x_rgb = target["x_rgb"]
            y_rgb = target["y_rgb"]
            z = target["z_stable"]

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            X = (x_rgb - cx) * z / fx
            Y = (y_rgb - cy) * z / fy
            Z = z

            pt_camera = PointStamped()
            pt_camera.header.stamp = Time().to_msg()
            pt_camera.header.frame_id = frame_id
            pt_camera.point.x = float(X)
            pt_camera.point.y = float(Y)
            pt_camera.point.z = float(Z)

            pt_map = self.tf_buffer.transform(
                pt_camera,
                "map",
                timeout=Duration(seconds=1.0),
            )

            goal_pose = self.make_stop_pose_before_target(
                pt_map.point.x,
                pt_map.point.y,
            )

            if goal_pose is None:
                with self.lock:
                    self.ready_target = None
                return

            with self.lock:
                self.goal_in_progress = True
                self.ready_target = None

            self.navigator.goToPose(goal_pose)
            self.get_logger().info(
                f"[car] center=({x_rgb},{y_rgb}), "
                f"conf={target['conf']:.2f}, "
                f"depth_avg={target['z_raw_avg']:.2f} m, "
                f"depth_stable={target['z_stable']:.2f} m -> "
                f"Object 앞 {STOP_DISTANCE_M:.2f}m 지점으로 이동"
            )

        except Exception as e:
            self.get_logger().warn(f"TF or goal error: {e}")
            with self.lock:
                self.goal_in_progress = False

    def compose_display_image(self):
        with self.lock:
            rgb = self.rgb_image.copy() if self.rgb_image is not None else None
            depth = self.depth_image.copy() if self.depth_image is not None else None
            det = self.last_detection.copy() if self.last_detection is not None else None
            measurement_state = None if self.measurement_state is None else dict(self.measurement_state)
            ready_target = None if self.ready_target is None else dict(self.ready_target)

        if rgb is None or depth is None:
            return None

        rgb_display = rgb.copy()

        if det is not None:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            cx, cy = det["cx"], det["cy"]
            conf = det["conf"]
            label = f"{det['class_name']} {conf:.2f}"

            cv2.rectangle(rgb_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(rgb_display, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                rgb_display,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        lock_target = None
        if measurement_state is not None:
            lock_target = measurement_state
        elif ready_target is not None:
            lock_target = ready_target

        if lock_target is not None:
            x = int(lock_target["x_rgb"])
            y = int(lock_target["y_rgb"])

            cv2.circle(rgb_display, (x, y), 8, (0, 255, 255), 2)
            cv2.line(rgb_display, (x - 12, y), (x + 12, y), (0, 255, 255), 2)
            cv2.line(rgb_display, (x, y - 12), (x, y + 12), (0, 255, 255), 2)

            if measurement_state is not None:
                text = f"MEASURING {len(measurement_state['samples_m'])}/{DEPTH_SAMPLE_COUNT}"
                cv2.putText(
                    rgb_display,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )

            if ready_target is not None:
                text = f"AVG {ready_target['z_raw_avg']:.2f}m / STABLE {ready_target['z_stable']:.2f}m"
                cv2.putText(
                    rgb_display,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 0),
                    2,
                )

        depth_vis = depth.copy()
        depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        if lock_target is not None:
            rgb_shape = rgb.shape[:2]
            depth_shape = depth.shape[:2]
            depth_pixel = self.get_scaled_pixel(
                lock_target["x_rgb"],
                lock_target["y_rgb"],
                rgb_shape,
                depth_shape,
            )
            if depth_pixel is not None:
                x_depth, y_depth = depth_pixel
                if 0 <= x_depth < depth_shape[1] and 0 <= y_depth < depth_shape[0]:
                    cv2.circle(depth_colored, (x_depth, y_depth), 6, (255, 255, 255), -1)

        if rgb_display.shape[:2] != depth_colored.shape[:2]:
            depth_colored = cv2.resize(
                depth_colored,
                (rgb_display.shape[1], rgb_display.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        combined = np.hstack((rgb_display, depth_colored))
        return combined

    def update_navigation_state(self):
        with self.lock:
            if not self.goal_in_progress:
                return

        try:
            if self.navigator.isTaskComplete():
                result = self.navigator.getResult()
                self.get_logger().info(f"Navigation task complete: {result}")
                with self.lock:
                    self.goal_in_progress = False
        except Exception:
            pass

    def display_images(self):
        combined = self.compose_display_image()
        if combined is not None:
            with self.lock:
                self.display_image = combined.copy()

        self.update_navigation_state()

    def gui_loop(self):
        cv2.namedWindow(GUI_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(GUI_WINDOW_NAME, GUI_WIDTH, GUI_HEIGHT)
        cv2.moveWindow(GUI_WINDOW_NAME, 100, 100)

        while not self.gui_thread_stop.is_set():
            with self.lock:
                img = self.display_image.copy() if self.display_image is not None else None

            if img is not None:
                cv2.imshow(GUI_WINDOW_NAME, img)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    self.get_logger().info("Shutdown requested by user (via GUI).")
                    self.gui_thread_stop.set()
                    break
            else:
                cv2.waitKey(10)

    def destroy(self):
        self.gui_thread_stop.set()
        if self.gui_thread.is_alive():
            self.gui_thread.join(timeout=1.0)
        cv2.destroyAllWindows()

        try:
            if self.nav_ready:
                self.navigator.dock()
        except Exception as e:
            self.get_logger().warn(f"Dock failed during destroy: {e}")


def main():
    rclpy.init()
    node = DepthToMap()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()