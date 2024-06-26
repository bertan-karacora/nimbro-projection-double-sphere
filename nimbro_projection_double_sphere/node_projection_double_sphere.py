import copy

import numpy as np
import torch

from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber as SubscriberFilter
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, ReliabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import nimbro_utils.compat.point_cloud2 as point_cloud2
import nimbro_utils.node as utils_node
from nimbro_utils.parameter_handler import ParameterHandler
from nimbro_utils.tf_oracle import TFOracle

from model_double_sphere import ModelDoubleSphere


class NodeProjectionDoubleSphere(Node):
    def __init__(
        self,
        name_frame_camera="camera_ids_link",
        name_frame_lidar="os_sensor_link",
        topic_image="/camera_ids/image_color",
        topic_info="/camera_ids/camera_info",
        topic_projected_depth="/camera_ids/projected/depth/image",
        topic_projected_points="/ouster/projected/points",
        topic_points="/ouster/points",
    ):
        super().__init__(node_name="projection_double_sphere")

        self.bridge_cv = None
        self.handler_parameters = None
        self.name_frame_camera = name_frame_camera
        self.name_frame_lidar = name_frame_lidar
        self.publisher_depth = None
        self.publisher_points_colored = None
        self.profile_qos = None
        self.subscriber_image = None
        self.subscriber_points = None
        self.tf_broadcaster = None
        self.tf_buffer = None
        self.tf_listener = None
        self.tf_oracle = None
        self.topic_image = topic_image
        self.topic_info = topic_info
        self.topic_projected_depth = topic_projected_depth
        self.topic_projected_points = topic_projected_points
        self.topic_points = topic_points

        self._init()

    def _init(self):
        self.bridge_cv = CvBridge()
        self.profile_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.handler_parameters = ParameterHandler(self, verbose=False)

        self._init_tf_oracle()
        # self._init_parameters()
        self._init_publishers()
        self._init_subscribers()

    def _init_tf_oracle(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_oracle = TFOracle(self)

    def _init_parameters(self): ...

    def _init_publishers(self):
        # namespace_topic = f"{self.get_namespace() if self.get_namespace() != '/' else ''}/{self.get_name()}"
        self.publisher_depth = self.create_publisher(msg_type=Image, topic=self.topic_projected_depth, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup())
        self.publisher_points = self.create_publisher(msg_type=PointCloud2, topic=self.topic_projected_points, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup())

    def _init_subscribers(self):
        self.subscriber_image = SubscriberFilter(self, Image, self.topic_image, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup())
        self.subscriber_info = SubscriberFilter(self, CameraInfo, self.topic_info, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup())
        self.subscriber_points = SubscriberFilter(self, PointCloud2, self.topic_points, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup())

        # TODO
        self.synchronizer = ApproximateTimeSynchronizer(fs=[self.subscriber_image, self.subscriber_info, self.subscriber_points], queue_size=10, slop=0.2)
        self.synchronizer.registerCallback(self.on_messages_received_callback)

    def publish_image(self, message_image, image, stamp):
        header = copy.copy(message_image.header)
        header.stamp = stamp
        message = self.bridge_cv.cv2_to_imgmsg(image, header=header, encoding="mono16")

        self.publisher_depth.publish(message)

    def publish_points(self, message_pointcloud, pointcloud_colored, offset):
        fields = message_pointcloud.fields + [PointField(name="rgb", offset=offset, datatype=PointField.UINT32, count=1)]
        message = point_cloud2.create_cloud(message_pointcloud.header, fields, pointcloud_colored)

        self.publisher_points.publish(message)

    def points2tensor(self, pointcloud):
        """Return image as tensor of shape [N, C, HxW]"""
        points = pointcloud[["x", "y", "z"]]
        points = np.lib.recfunctions.structured_to_unstructured(points)
        points = torch.as_tensor(points)
        points = points.permute(1, 0)[None, ...]

        return points

    def image2tensor(self, image):
        """Return image as tensor of shape [N, C, H, W]"""
        image = torch.as_tensor(image)
        image = image.permute(2, 0, 1)[None, ...]
        return image

    def colors2numpy(self, colors):
        colors = colors.numpy(force=True)
        colors = colors.astype(np.uint32)

        r = colors[0]
        g = colors[1]
        b = colors[2]
        colors = (r << 16) | (g << 8) | b

        return colors

    def create_dtype_with_rgb(self, dtype):
        # There is no better way to access this >:(
        # Also np.lib.recfunctions.append_fields does not work with offsets, formats, itemsize
        formats = [field[0] for field in dtype.fields.values()]
        offsets = [field[1] for field in dtype.fields.values()]
        offset = offsets[-1] + np.dtype(formats[-1]).itemsize

        dtype = np.dtype(
            {
                "names": list(dtype.names) + ["rgb"],
                "formats": formats + ["<u4"],
                "offsets": offsets + [offset],
                # Round up to next multiple of 8
                "itemsize": int(np.ceil((offset + 4) / 8)) * 8,
            }
        )

        return dtype, offset

    def compute_pointcloud_colored(self, model_double_sphere, coords_uv, image, mask_valid, pointcloud):
        colors = model_double_sphere.sample_color(coords_uv, image, mask_valid)
        colors = colors[0]

        colors = self.colors2numpy(colors)

        dtype, offset = self.create_dtype_with_rgb(pointcloud.dtype)
        # Filling a new array for performance (see: https://stackoverflow.com/questions/25427197/numpy-how-to-add-a-column-to-an-existing-structured-array)
        pointcloud_colored = np.empty(pointcloud.shape, dtype=dtype)
        names_fields_before = list(pointcloud.dtype.names)
        pointcloud_colored[names_fields_before] = pointcloud[names_fields_before]
        pointcloud_colored["rgb"] = colors

        return pointcloud_colored, offset

    def compute_depth_image(self, model_double_sphere, coords_uv, points, mask_valid):
        image_depth = model_double_sphere.sample_depth(coords_uv, points, mask_valid, use_knn_interpolate=True, ratio_downsampling=8, k=1)
        image_depth = image_depth[0, 0]

        image_depth = image_depth.numpy(force=True)
        image_depth = image_depth.astype(np.uint16)

        return image_depth

    def on_messages_received_callback(self, message_image, message_info, message_points):
        success, message, message_points = self.tf_oracle.transform_to_frame(message_points, self.name_frame_camera)
        if not success:
            self.get_logger().debug(message)
            return

        # Methods like torch.frombuffer or np.frombuffer do not work if incoming data has points with padded bytes
        # E.g. the ouster/points topic has an itemsize of 16 while it publishes only xyz in float32 each
        pointcloud = point_cloud2.read_points(message_points, skip_nans=True)
        points = self.points2tensor(pointcloud)

        image = self.bridge_cv.imgmsg_to_cv2(message_image, desired_encoding="passthrough")
        image = self.image2tensor(image)

        model_double_sphere = ModelDoubleSphere.from_camera_info_message(message_info)

        coords_uv_points, mask_valid = model_double_sphere.project_points_onto_image(points, use_invalid_coords=True, use_mask_fov=True)

        pointcloud_colored, offset = self.compute_pointcloud_colored(model_double_sphere, coords_uv_points, image, mask_valid, pointcloud)
        self.publish_points(message_points, pointcloud_colored, offset)

        image_depth = self.compute_depth_image(model_double_sphere, coords_uv_points, points, mask_valid)
        self.publish_image(message_image, image_depth, message_points.header.stamp)
