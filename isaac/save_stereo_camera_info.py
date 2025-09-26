import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo

class CameraInfoSaver(Node):
    def __init__(self):
        super().__init__('camera_info_saver')
        # Subscribe once
        self.create_subscription(CameraInfo, '/front_stereo_camera/left/camera_info', self.callback, 10)

    def callback(self, msg: CameraInfo):
        # Extract intrinsic matrix K
        cam_K = list(msg.k)  # msg.k is already a list of 9 floats

        # Baseline, set default to 0.15 as the hawk stereo camera distance
        baseline = 0.15

        # Save the camera info into a text file:
        # 754.6680908203125 0.0 489.3794860839844 0.0 754.6680908203125 265.16162109375 0.0 0.0 1.0 
        # 0.063


        # Save to K.txt
        with open('K.txt', 'w') as f:
            f.write(" ".join(map(str, cam_K)) + "\n")
            f.write(str(baseline) + "\n")

        self.get_logger().info("Saved K.txt")
        # Exit after saving
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoSaver()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
