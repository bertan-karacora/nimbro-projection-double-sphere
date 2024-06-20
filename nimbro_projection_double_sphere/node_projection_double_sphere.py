import numpy as np
import torch

from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber as SubscriberFilter
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, ReliabilityPolicy, QoSProfile
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor, ParameterType
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField, RegionOfInterest
import sensor_msgs_py.point_cloud2 as pc2_py
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


import nimbro_utils.compat.point_cloud2 as point_cloud2
import nimbro_utils.geometry as utils_geometry
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
        self.device = None
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def _init_publishers(self):
        # namespace_topic = f"{self.get_namespace() if self.get_namespace() != '/' else ''}/{self.get_name()}"
        self.publisher_depth = self.create_publisher(
            msg_type=Image, topic=self.topic_projected_depth, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup()
        )

        self.publisher_points = self.create_publisher(
            msg_type=PointCloud2, topic=self.topic_projected_points, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup()
        )

    def _init_subscribers(self):
        self.subscriber_image = SubscriberFilter(self, Image, self.topic_image, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup())
        self.subscriber_info = SubscriberFilter(self, CameraInfo, self.topic_info, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup())
        self.subscriber_points = SubscriberFilter(self, PointCloud2, self.topic_points, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup())

        self.synchronizer = ApproximateTimeSynchronizer(fs=[self.subscriber_image, self.subscriber_info, self.subscriber_points], queue_size=10, slop=0.2)
        self.synchronizer.registerCallback(self.on_messages_received_callback)

    def publish_image(self, message_image: Image, image: torch.Tensor):
        message = self.bridge_cv.cv2_to_imgmsg(image.get_numpy_3D(), header=message_image.header, encoding="mono16")

        self.publisher_depth.publish(message)

    def publish_points(self, message_pointcloud: PointCloud2, points: torch.Tensor, offset: int):
        fields = message_pointcloud.fields + [PointField(name="rgb", offset=offset, datatype=PointField.UINT32, count=1)]
        message = point_cloud2.create_cloud(message_pointcloud.header, fields, points)

        self.publisher_points.publish(message)

    def points2tensor(self, pointcloud: np.ndarray):
        """Return image as tensor of shape [N, C, HxW]"""
        points = pointcloud[["x", "y", "z"]]
        points = np.lib.recfunctions.structured_to_unstructured(points)
        points = torch.as_tensor(points, device=self.device)
        points = points.permute(1, 0)[None, ...]

        return points

    def image2tensor(self, image: torch.Tensor):
        """Return image as tensor of shape [C, H, W]"""
        image = torch.as_tensor(image, device=self.device)
        image = image.permute(2, 0, 1)[None, ...]
        return image

    def colors2numpy(self, colors: torch.Tensor):
        colors = colors.permute(1, 0)
        colors = colors.numpy(force=True).astype(np.uint32)
        r = colors[..., 0]
        g = colors[..., 1]
        b = colors[..., 2]
        colors = (r << 16) | (g << 8) | b
        return colors

    def create_dtype_with_rgb(self, dtype: np.dtype):
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

    @torch.inference_mode()
    def sample_color(self, coords_uv: torch.Tensor, image: torch.Tensor, mask_valid: torch.Tensor, color_invalid: tuple = (255, 87, 51)):
        coords_uv[..., 0] = 2.0 * coords_uv[..., 0] / image.shape[-1] - 1.0
        coords_uv[..., 1] = 2.0 * coords_uv[..., 1] / image.shape[-2] - 1.0

        # Interpolation is only implemented for floats
        image = image.float()
        colors = torch.nn.functional.grid_sample(image, coords_uv[..., None, :, :], align_corners=True)[..., 0, :]
        image = image.byte()

        colors[:, 0, :].masked_fill_(~mask_valid, color_invalid[0])
        colors[:, 1, :].masked_fill_(~mask_valid, color_invalid[1])
        colors[:, 2, :].masked_fill_(~mask_valid, color_invalid[2])

        return colors

    @torch.inference_mode()
    def sample_depth(self, coords_xyz: torch.Tensor, points: torch.Tensor, mask_valid: torch.Tensor, depth_invalid: float = 0):
        depth = ...

        depth.masked_fill_(~mask_valid, depth_invalid)

        return depth

    def compute_color_points(self, pointcloud: np.ndarray, points: torch.Tensor, image: torch.Tensor, model_double_sphere: ModelDoubleSphere):
        coords_uv, mask_valid = model_double_sphere.project_points_onto_image(points, use_invalid_coords=True)

        colors = self.sample_color(coords_uv, image, mask_valid)
        colors = colors[0]

        colors = self.colors2numpy(colors)

        dtype, offset = self.create_dtype_with_rgb(pointcloud.dtype)
        # Filling a new array for performance (see: https://stackoverflow.com/questions/25427197/numpy-how-to-add-a-column-to-an-existing-structured-array)
        pointcloud_colored = np.empty(pointcloud.shape, dtype=dtype)
        pointcloud_colored[list(pointcloud.dtype.names)] = pointcloud[list(pointcloud.dtype.names)]
        pointcloud_colored["rgb"] = colors

        return pointcloud_colored, offset

    def compute_depth_image(self, pointcloud: np.ndarray, points: torch.Tensor, image: torch.Tensor, model_double_sphere: ModelDoubleSphere):
        coords_uv = torch.stack(torch.meshgrid(torch.arange(image.shape[0]), torch.arange(image.shape[1])))[None, ...]
        coords_xyz, mask_valid = model_double_sphere.project_image_onto_points(coords_uv, use_invalid_coords=True)

        image_depth = self.sample_depth(coords_xyz, points, mask_valid)
        image_depth = image_depth[0]

        return image_depth

    def on_messages_received_callback(self, message_image: Image, message_info: CameraInfo, message_points: PointCloud2):
        # TODO: use half precision

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

        model_double_sphere = ModelDoubleSphere.from_camera_info_message(message_info, device=self.device)

        points_colored, offset = self.compute_color_points(pointcloud, points, image, model_double_sphere)
        self.publish_points(message_points, points_colored, offset)

        # image_depth = self.compute_depth_image(model_double_sphere)
        # self.publish_image(image_depth)
