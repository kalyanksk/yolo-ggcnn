import numpy as np
import torch
import cv2
import scipy.ndimage as ndimage
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from message_filters import Subscriber, ApproximateTimeSynchronizer
import time
import warnings
import os 
import sys 


sys.path.append(os.path.abspath('ggcnn/'))


bridge = CvBridge()

# Suppress torch warning
warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)

MODEL_FILE = 'ggcnn/ggcnn2_weights_cornell/epoch_50_cornell'
model = torch.load(MODEL_FILE)
device = torch.device("cuda:0")


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection')


        self.cmd_pub = self.create_publisher(Float32MultiArray, 'grasp/command', 1)

        self.prev_mp = np.array([150, 150])

        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/kinect/camera_info', self.camera_info_callback, 1)
        self.camera_info = None

        self.depth_sub = Subscriber(self, Image, '/kinect_depth')
        self.bbox_sub = Subscriber(self, Float32MultiArray, '/yolo/detections')
        self.plain_depth_image_path = 'assets/plain_depth_image.png'

        ats = ApproximateTimeSynchronizer([self.depth_sub, self.bbox_sub], queue_size=1, slop=0.025, allow_headerless=True)
        ats.registerCallback(self.callback)

    def camera_info_callback(self, msg):
        self.camera_info = msg
        K = self.camera_info.k
        self.fx = K[0]
        self.cx = K[2]
        self.fy = K[4]
        self.cy = K[5]

    def callback(self,depth, bbox_msg):
        if self.camera_info is None:
            return
        

        # Save the plain depth image to disk if it doesn't exist
        if not os.path.exists(self.plain_depth_image_path):
            cv2.imwrite(self.plain_depth_image_path, depth)
            self.get_logger().info('Plain depth image saved to disk.')
            return

        # Load the plain depth image from disk
        plain_depth_image = cv2.imread(self.plain_depth_image_path, cv2.IMREAD_UNCHANGED)

        with TimeIt('Crop'):
            depth = bridge.imgmsg_to_cv2(depth, '16UC1')
            depth_copy = depth.copy()

            x_min, y_min, x_max, y_max = 0, 0, 0, 0  

            # Check if bbox_msg is valid
            if bbox_msg and hasattr(bbox_msg, 'data'):
                bbox = bbox_msg.data
                if len(bbox) >= 4:
                    x_min = int(bbox[0])
                    y_min = int(bbox[1])
                    x_max = int(bbox[2])
                    y_max = int(bbox[3])
                else:
                    # print("Received bbox data is not valid")
                    pass
            else:
                # print("No bbox message received or bbox_msg has no data")
                pass

            object_mask = np.zeros(depth.shape, dtype=np.uint8)
            if x_min <= x_max and y_min <= y_max:
                object_mask[y_min:y_max, x_min:x_max] = 255

            object_only_depth = cv2.bitwise_and(depth, depth, mask=object_mask)
            masked_plain_depth_image = cv2.bitwise_and(plain_depth_image, plain_depth_image, mask=255 - object_mask)
            final_depth_image = cv2.add(masked_plain_depth_image, object_only_depth)
 
            crop_size = 400
            depth_crop = cv2.resize(final_depth_image[(480-crop_size)//2:(480-crop_size)//2+crop_size,
                                          (640-crop_size)//2:(640-crop_size)//2+crop_size], (300, 300))        

            depth_crop = depth_crop.copy()
            depth_nan = np.isnan(depth_crop).copy()
            depth_crop[depth_nan] = 0


            depth_crop = depth_crop.copy()
            depth_nan = np.isnan(depth_crop).copy()
            depth_crop[depth_nan] = 0


        with TimeIt('Inpaint'):
            depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
            mask = (depth_crop == 0).astype(np.uint8)
            depth_scale = np.abs(depth_crop).max()
            depth_crop = depth_crop.astype(np.float32) / depth_scale
            depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS)
            depth_crop = depth_crop[1:-1, 1:-1]
            depth_crop = depth_crop * depth_scale

        with TimeIt('Calculate depth'):
            depth_center = depth_crop[100:141, 130:171].flatten()
            depth_center.sort()
            depth_center = depth_center[:10].mean() * 1000.0

        with TimeIt('Inference'):
            depth_crop = depth_crop[np.newaxis, :, :]
            depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
            imageT = torch.from_numpy(depth_crop.reshape(1, 1, 300, 300).astype(np.float32)).to(device)

            with torch.no_grad():
                pred_out = model(imageT)

            points_out = pred_out[0].cpu().numpy().squeeze()
            points_out[depth_nan] = 0

        with TimeIt('Trig'):
            cos_out = pred_out[1].cpu().numpy().squeeze()
            sin_out = pred_out[2].cpu().numpy().squeeze()
            ang_out = np.arctan2(sin_out, cos_out) / 2.0
            width_out = pred_out[3].squeeze() * 150.0

        with TimeIt('Filter'):
            points_out = ndimage.gaussian_filter(points_out, 5.0)
            points_out = np.clip(points_out, 0.0, 1.0 - 1e-3)
            ang_out = ndimage.gaussian_filter(ang_out, 2.0)

        with TimeIt('Control'):
            max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
            self.prev_mp = max_pixel.astype(np.int32)
            ang = ang_out[max_pixel[0], max_pixel[1]]
            width = width_out[max_pixel[0], max_pixel[1]]
            max_pixel = ((np.array(max_pixel) / 300.0 * crop_size) +
                         np.array([(480 - crop_size) // 2, (640 - crop_size) // 2]))
            max_pixel = np.round(max_pixel).astype(np.int32)
            point_depth = depth_copy[max_pixel[0], max_pixel[1]]
            x = (max_pixel[1] - self.cx) / self.fx * point_depth
            y = (max_pixel[0] - self.cy) / self.fy * point_depth
            z = point_depth

            if np.isnan(z):
                return


        with TimeIt('Publish'):
            cmd_msg = Float32MultiArray()
            cmd_msg.data = [float(x) / 1000.0, float(y)/ 1000.0, float(z)/ 1000.0, float(ang), float(width), float(depth_center)]
            self.cmd_pub.publish(cmd_msg)


class TimeIt:
    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None
        self.print_output = False

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        self.t1 = time.time()
        if self.print_output:
            print('%s: %s' % (self.s, self.t1 - self.t0))


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
