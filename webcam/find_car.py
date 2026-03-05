#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from ultralytics import YOLO
import cv2


class CarDetectionNode(Node):
    def __init__(self):
        super().__init__('find_car')

        self.model = YOLO('/home/rokey/rokey_ws/car.pt')
        self.cap   = cv2.VideoCapture(2)
        self.pub   = self.create_publisher(Bool, 'car_detecting', 10)

        self._detected_time = None  # 최초 감지 시각

        self.create_timer(0.033, self.process_frame)
        self.get_logger().info('노드 시작')

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        results      = self.model(frame, verbose=False)[0]
        car_detected = any(self.model.names[int(b.cls[0])] == 'car' for b in results.boxes)

        cv2.imshow('Car Detection', results.plot())
        cv2.waitKey(1)

        now = self.get_clock().now().nanoseconds / 1e9

        if car_detected and self._detected_time is None:
            self.get_logger().info('Car 감지! → 1초 후 발행 시작')
            self._detected_time = now

        if self._detected_time is None:
            return

        elapsed = now - self._detected_time

        if elapsed >= 6.0:  # 1초 대기 + 5초 발행
            self.get_logger().info('발행 완료 → 노드 종료')
            raise SystemExit

        if elapsed >= 1.0:
            msg = Bool()
            msg.data = car_detected
            self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CarDetectionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()