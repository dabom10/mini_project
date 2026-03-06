#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool
import threading

from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator


class MoveThere(Node):
    def __init__(self, navigator: TurtleBot4Navigator):
        super().__init__('move_there')

        self.navigator = navigator

        self.sub = self.create_subscription(Bool, '/robot3/car_detecting', self.car_detecting_cb, 10)
        self.pub = self.create_publisher(Bool, 'arrived_point', 10)

        self._moving  = False
        self._arrived = False

        self.create_timer(0.033, self.publish_arrived)
        self.get_logger().info('move_there 노드 시작 — /car_detecting 구독 중')

    # ── /arrived_point 상시 발행 ────────────────────────────────────────
    def publish_arrived(self):
        msg = Bool()
        msg.data = self._arrived
        self.pub.publish(msg)

    # ── 콜백 ────────────────────────────────────────────────────────────
    def car_detecting_cb(self, msg: Bool):
        if self._moving or not msg.data:
            return
        self.get_logger().info('True 수신 → 이동 시작')
        self._moving = True
        threading.Thread(target=self.run_navigation, daemon=True).start()

    # ── 이동 로직 (스레드에서 실행) ─────────────────────────────────────
    def run_navigation(self):
        navigator = self.navigator
        self.get_logger().info('1. getDockedStatus 시도')

        if not navigator.getDockedStatus():
            self.get_logger().info('2. docking')
            navigator.dock()
        self.get_logger().info('3. setInitialPose')
        initial_pose = navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.NORTH)
        navigator.setInitialPose(initial_pose)
        self.get_logger().info('4. waitUntilNav2Active')
        navigator.waitUntilNav2Active()
        self.get_logger().info('5. goal 설정')

        goal_pose = navigator.getPoseStamped([-2.088, 1.676], TurtleBot4Directions.NORTH)
        navigator.undock()
        navigator.startToPose(goal_pose)

        self._arrived = True
        self.get_logger().info('목표 지점 도착 완료 → /arrived_point True 발행 시작')


# ── main ────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)

    navigator = TurtleBot4Navigator()
    node      = MoveThere(navigator)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(navigator)

    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()