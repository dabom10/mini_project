#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from irobot_create_msgs.action import Undock
from ultralytics import YOLO
import cv2


class CarDetectionUndockNode(Node):
    def __init__(self):
        super().__init__('find_car')

        self.model = YOLO('/home/rokey/rokey_ws/car.pt')
        self.cap   = cv2.VideoCapture(2)

        self._undock_client = ActionClient(self, Undock, '/robot3/undock')
        self._undock_client.wait_for_server()

        self._undocking = False
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

        # 디버그: 바운딩 박스 시각화
        annotated = results.plot()
        cv2.imshow('Car Detection', annotated)
        cv2.waitKey(1)

        if car_detected and not self._undocking:
            self.send_undock()

    def send_undock(self):
        self.get_logger().info('Car 감지 → Undock 전송')
        self._undocking = True
        future = self._undock_client.send_goal_async(Undock.Goal())
        future.add_done_callback(self.on_goal_response)

    def on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Undock 거절됨')
            self._undocking = False
            return
        goal_handle.get_result_async().add_done_callback(self.on_result)

    def on_result(self, future):
        self.get_logger().info('Undock 완료')
        self._undocking = False

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CarDetectionUndockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()