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

        self.pub = self.create_publisher(Bool, 'car_detecting', 10)

        self.create_timer(0.033, self.process_frame)
        self.get_logger().info('노드 시작')

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        results = self.model(frame, verbose=False)[0]
        names   = self.model.names

        car_detected = any(
            names[int(box.cls[0])] == 'car'
            for box in results.boxes
        )

        msg = Bool()
        msg.data = car_detected
        self.pub.publish(msg)

        annotated = results.plot()
        cv2.imshow('Car Detection', annotated)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CarDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()