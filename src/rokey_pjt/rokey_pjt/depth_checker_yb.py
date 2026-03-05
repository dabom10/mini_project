#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

import numpy as np
import cv2


DEPTH_TOPIC = "/robot3/oakd/stereo/image_raw/compressedDepth"
RGB_TOPIC = "/robot3/oakd/rgb/image_raw/compressed"

WINDOW_NAME = "RGB | Depth"
COMPRESSED_DEPTH_HEADER_BYTES = 12


class DepthChecker(Node):

    def __init__(self):
        super().__init__("depth_checker")

        self.depth_mm = None
        self.rgb_bgr = None

        self.create_subscription(CompressedImage, DEPTH_TOPIC, self.on_depth, 10)
        self.create_subscription(CompressedImage, RGB_TOPIC, self.on_rgb, 10)

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

    def on_depth(self, msg: CompressedImage):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        if data.size <= COMPRESSED_DEPTH_HEADER_BYTES:
            return

        png = data[COMPRESSED_DEPTH_HEADER_BYTES:]
        depth = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
        if depth is None:
            return

        self.depth_mm = depth
        self.draw()

    def on_rgb(self, msg: CompressedImage):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        rgb = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if rgb is None:
            return

        self.rgb_bgr = rgb
        self.draw()

    def draw(self):
        if self.rgb_bgr is None or self.depth_mm is None:
            return

        depth_vis = cv2.convertScaleAbs(self.depth_mm, alpha=0.03)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # 높이를 rgb 기준으로 맞춰서 붙이기
        h = self.rgb_bgr.shape[0]
        if depth_vis.shape[0] != h:
            depth_vis = cv2.resize(depth_vis, (int(depth_vis.shape[1] * h / depth_vis.shape[0]), h), interpolation=cv2.INTER_NEAREST)

        canvas = np.hstack([self.rgb_bgr, depth_vis])

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            rclpy.shutdown()
            cv2.destroyAllWindows()

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.rgb_bgr is None or self.depth_mm is None:
            return

        rgb_h, rgb_w = self.rgb_bgr.shape[:2]

        # 좌측은 RGB, 우측은 Depth
        if x < rgb_w:
            return

        # depth 영역 좌표로 변환
        dx = x - rgb_w

        # draw()에서 depth를 resize 했을 수 있으니, 표시된 depth의 크기를 기준으로 매핑 필요
        depth_h, depth_w = self.depth_mm.shape[:2]

        # 현재 표시된 depth_vis 크기 계산( draw와 동일 로직 )
        shown_h = rgb_h
        shown_w = int(depth_w * shown_h / depth_h)

        if shown_w <= 0:
            return

        # 클릭 좌표(dx, y)를 원본 depth 좌표로 역변환
        u = int(dx * (depth_w / float(shown_w)))
        v = int(y * (depth_h / float(shown_h)))

        if not (0 <= u < depth_w and 0 <= v < depth_h):
            return

        d_mm = int(self.depth_mm[v, u])
        print(f"pixel ({u},{v}) : {d_mm} mm")


def main():
    rclpy.init()
    node = DepthChecker()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()