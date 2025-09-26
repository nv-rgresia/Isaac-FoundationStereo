import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from functools import partial

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()

        # Subscribers
        # Subscribers with side info
        self.left_img_sub = self.create_subscription(
            Image,
            '/front_stereo_camera/left/image_rect_color',
            partial(self.rgb_callback, side='left'),
            10
        )
        self.right_img_sub = self.create_subscription(
            Image,
            '/front_stereo_camera/right/image_rect_color',
            partial(self.rgb_callback, side='right'),
            10
        )
        self.left_rgb_saved = False
        self.right_rgb_saved = False

    def rgb_callback(self, msg, side='left'):
        rgb_saved = self.left_rgb_saved if side == 'left' else self.right_rgb_saved
        if not rgb_saved:
            # Convert ROS2 RGB image to OpenCV format
            rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            cv2.imwrite(f'rgb_{side}.png', rgb_img)
            self.get_logger().info(f"Saved rgb_{side}.png")
            
            if side == 'left':
                self.left_rgb_saved = True
            else:
                self.right_rgb_saved = True

            if self.left_rgb_saved and self.right_rgb_saved:
                self.get_logger().info("Both images saved, shutting down.")
                rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
