import copy
import threading

import numpy as np
import torch

from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Cache, Subscriber as SubscriberFilter
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, ReliabilityPolicy, QoSProfile
from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor, ParameterType
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import nimbro_utils.compat.point_cloud2 as point_cloud2
from nimbro_utils.parameter_handler import ParameterHandler
from nimbro_utils.tf_oracle import TFOracle

from nimbro_projection_double_sphere.model_double_sphere import ModelDoubleSphere
from nimbro_projection_double_sphere.sampler_color import SamplerColor
from nimbro_projection_double_sphere.sampler_depth import SamplerDepth


class NodeProjectionDoubleSphere(Node):
    def __init__(
        self,
        name_frame_camera="camera_ids_link",
        name_frame_lidar="os_sensor_link",
        slop_synchronizer=0.05,
        topic_image="/camera_ids/image_color",
        topic_info="/camera_ids/camera_info",
        topic_projected_depth="/camera_ids/projected/depth/image",
        topic_projected_points="/ouster/projected/points",
        topic_points="/ouster/points",
        color_invalid="(255, 87, 51)",
        factor_downsampling=8,
        use_knn_interpolation=True,
        k_knn=1,
        mode_interpolation="nearest",
    ):
        super().__init__(node_name="projection_double_sphere")

        self.bridge_cv = None
        self.cache_times_points_message = []
        self.color_invalid = color_invalid
        self.coords_uv_full_flat = None
        self.handler_parameters = None
        self.k_knn = k_knn
        self.lock = None
        self.mode_interpolation = mode_interpolation
        self.name_frame_camera = name_frame_camera
        self.name_frame_lidar = name_frame_lidar
        self.publisher_depth = None
        self.publisher_points_colored = None
        self.profile_qos = None
        self.factor_downsampling = factor_downsampling
        self.sampler_color = None
        self.sampler_depth = None
        self.slop_synchronizer = slop_synchronizer
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
        self.use_knn_interpolation = use_knn_interpolation

        self._init()

    def _init(self):
        self.lock = threading.Lock()
        self.bridge_cv = CvBridge()
        self.profile_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.handler_parameters = ParameterHandler(self, verbose=False)

        self.sampler_color = SamplerColor(color_invalid=self.color_invalid)
        self.sampler_depth = SamplerDepth(factor_downsampling=self.factor_downsampling, k_knn=self.k_knn, mode_interpolation=self.mode_interpolation)

        self._init_parameters()

        self._init_tf_oracle()
        self._init_publishers()
        self._init_subscribers()

    def _init_tf_oracle(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_oracle = TFOracle(self)

    def _init_publishers(self):
        # namespace_topic = f"{self.get_namespace() if self.get_namespace() != '/' else ''}/{self.get_name()}"
        self.publisher_depth = self.create_publisher(msg_type=Image, topic=self.topic_projected_depth, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup())
        self.publisher_points = self.create_publisher(msg_type=PointCloud2, topic=self.topic_projected_points, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup())

    def _init_subscribers(self):
        self.subscriber_image = SubscriberFilter(self, Image, self.topic_image, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup())
        self.subscriber_info = SubscriberFilter(self, CameraInfo, self.topic_info, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup())
        self.subscriber_points = SubscriberFilter(self, PointCloud2, self.topic_points, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup())

        # ApproximateTimeSynchronizer not working as expected. Slop is disregarded and messages are often reused more than once
        # self.synchronizer = ApproximateTimeSynchronizer(fs=[self.subscriber_points, self.subscriber_image, self.subscriber_info], queue_size=3, slop=self.slop_synchronizer)
        # self.synchronizer.registerCallback(self.on_messages_received_callback)

        self.cache_image = Cache(self.subscriber_image, 15)
        self.cache_info = Cache(self.subscriber_info, 15)
        self.cache_points = Cache(self.subscriber_points, 15)

        self.cache_image.registerCallback(self.on_message_image_received_callback)
        self.cache_points.registerCallback(self.on_message_points_received_callback)

    def publish_image(self, message_image, image, stamp=None):
        if stamp is not None:
            header = copy.copy(message_image.header)
            header.stamp = stamp
        else:
            header = message_image.header

        message = self.bridge_cv.cv2_to_imgmsg(image, header=header, encoding="mono16")

        self.publisher_depth.publish(message)

    def publish_points(self, message_pointcloud, pointcloud_colored, offset):
        fields = message_pointcloud.fields + [PointField(name="rgb", offset=offset, datatype=PointField.UINT32, count=1)]
        message = point_cloud2.create_cloud(message_pointcloud.header, fields, pointcloud_colored)

        self.publisher_points.publish(message)

    def points2tensor(self, pointcloud):
        """Return image as tensor of shape [B, C, HxW]"""
        points = pointcloud[["x", "y", "z"]]
        points = np.lib.recfunctions.structured_to_unstructured(points)
        points = torch.as_tensor(points)
        points = points.permute(1, 0)
        points = points[None, ...]

        return points

    def image2tensor(self, image):
        """Return image as tensor of shape [B, C, H, W]"""
        image = torch.as_tensor(image)
        image = image.permute(2, 0, 1)
        images = image[None, ...]
        return images

    def colors_to_numpy(self, colors):
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

    def compute_pointcloud_colored(self, coords_uv, images, pointcloud, mask_valid=None):
        colors = self.sampler_color(coords_uv, images, mask_valid, use_half_precision=True)
        colors = colors[0]
        colors = self.colors_to_numpy(colors)

        dtype, offset = self.create_dtype_with_rgb(pointcloud.dtype)
        # Filling a new array for performance (see: https://stackoverflow.com/questions/25427197/numpy-how-to-add-a-column-to-an-existing-structured-array)
        pointcloud_colored = np.empty(pointcloud.shape, dtype=dtype)
        names_fields_before = list(pointcloud.dtype.names)
        pointcloud_colored[names_fields_before] = pointcloud[names_fields_before]
        pointcloud_colored["rgb"] = colors

        return pointcloud_colored, offset

    def compute_depth_image(self, coords_uv, points, mask_valid=None):
        image_depth = self.sampler_depth(coords_uv, points, mask_valid, use_knn_interpolation=self.use_knn_interpolation)
        image_depth = image_depth[0]
        image_depth = image_depth.permute(1, 2, 0)

        image_depth = image_depth.numpy(force=True)
        image_depth = image_depth.astype(np.uint16)

        return image_depth

    def on_message_image_received_callback(self, message_image):
        time_message = Time.from_msg(message_image.header.stamp)
        self.on_messages_received_callback(time_message, message_image=message_image)

    def on_message_points_received_callback(self, message_points):
        time_message = Time.from_msg(message_points.header.stamp)
        self.on_messages_received_callback(time_message, message_points=message_points)

    def on_messages_received_callback(self, time_message, message_points=None, message_image=None, message_info=None):
        message_points = self.cache_points.getElemBeforeTime(time_message) if message_points is None else message_points
        message_image = self.cache_image.getElemBeforeTime(time_message) if message_image is None else message_image
        message_info = self.cache_info.getElemBeforeTime(time_message) if message_info is None else message_info

        if message_info is None or message_image is None or message_points is None:
            self.get_logger().debug(f"Cache empty")
            return

        time_image = Time.from_msg(message_image.header.stamp)
        time_info = Time.from_msg(message_info.header.stamp)
        time_points = Time.from_msg(message_points.header.stamp)

        if time_image != time_info:
            self.get_logger().debug("Image and info topic stamps unequal")
            return

        duration_difference = time_points - time_image if time_points > time_image else time_image - time_points
        if duration_difference > Duration(seconds=self.slop_synchronizer):
            self.get_logger().debug(f"Image/Info and point-cloud topic stamps difference too big: {duration_difference}")
            return

        self.lock.acquire()

        if any(time_points == time_cached for time_cached in self.cache_times_points_message):
            self.get_logger().debug(f"Messages skipped because the pointcloud has been used already")
            self.lock.release()
            return

        self.cache_times_points_message += [time_points]
        self.cache_times_points_message = self.cache_times_points_message[-15:]

        self.lock.release()

        success, message, message_points = self.tf_oracle.transform_to_frame(message_points, self.name_frame_camera)
        if not success:
            self.get_logger().debug(message)
            return

        # Methods like torch.frombuffer or np.frombuffer do not work if incoming data has points with padded bytes
        # E.g. the ouster/points topic has an itemsize of 16 while it publishes only xyz in float32
        pointcloud = point_cloud2.read_points(message_points, skip_nans=True)
        points = self.points2tensor(pointcloud)

        image = self.bridge_cv.imgmsg_to_cv2(message_image, desired_encoding="passthrough")
        images = self.image2tensor(image)

        model_double_sphere = ModelDoubleSphere.from_camera_info_message(message_info)
        coords_uv_points, mask_valid = model_double_sphere.project_points_onto_image(points, use_invalid_coords=True, use_mask_fov=True, use_half_precision=True)

        pointcloud_colored, offset = self.compute_pointcloud_colored(coords_uv_points, images, pointcloud, mask_valid)
        self.publish_points(message_points, pointcloud_colored, offset)

        # Number of channels unknown
        self.sampler_depth.shape_image = (-1, message_info.height, message_info.width)
        image_depth = self.compute_depth_image(coords_uv_points, points, mask_valid)
        self.publish_image(message_image, image_depth, stamp=message_points.header.stamp)

    def _init_parameters(self):
        self.add_on_set_parameters_callback(self.handler_parameters.parameter_callback)

        self._init_parameter_name_frame_camera()
        self._init_parameter_name_frame_lidar()
        self._init_parameter_topic_image()
        self._init_parameter_topic_info()
        self._init_parameter_topic_points()
        self._init_parameter_topic_projected_depth()
        self._init_parameter_topic_projected_points()
        self._init_parameter_slop_synchronizer()
        self._init_parameter_color_invalid()
        self._init_parameter_factor_downsampling()
        self._init_parameter_k_knn()
        self._init_parameter_mode_interpolation()
        self._init_parameter_use_knn_interpolation()

        self.handler_parameters.all_declared()

    def _init_parameter_name_frame_camera(self):
        descriptor = ParameterDescriptor(
            name="name_frame_camera",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the camera frame in the tf transform tree",
            read_only=False,
        )
        self.parameter_descriptors += [descriptor]
        self.declare_parameter(descriptor.name, self.name_frame_camera, descriptor)

    def _init_parameter_name_frame_lidar(self):
        descriptor = ParameterDescriptor(
            name="name_frame_lidar",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the lidar frame in the tf transform tree",
            read_only=False,
        )
        self.parameter_descriptors += [descriptor]
        self.declare_parameter(descriptor.name, self.name_frame_lidar, descriptor)

    def _init_parameter_topic_image(self):
        descriptor = ParameterDescriptor(
            name="topic_image",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the rgb image topic (for subscriber)",
            read_only=False,
        )
        self.parameter_descriptors += [descriptor]
        self.declare_parameter(descriptor.name, self.topic_image, descriptor)

    def _init_parameter_topic_info(self):
        descriptor = ParameterDescriptor(
            name="topic_info",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the camera info topic (for subscriber)",
            read_only=False,
        )
        self.parameter_descriptors += [descriptor]
        self.declare_parameter(descriptor.name, self.topic_info, descriptor)

    def _init_parameter_topic_points(self):
        descriptor = ParameterDescriptor(
            name="topic_points",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the pointcloud topic (for subscriber)",
            read_only=False,
        )
        self.parameter_descriptors += [descriptor]
        self.declare_parameter(descriptor.name, self.topic_points, descriptor)

    def _init_parameter_topic_projected_depth(self):
        descriptor = ParameterDescriptor(
            name="topic_projected_depth",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the projected depth topic (for publisher)",
            read_only=False,
        )
        self.parameter_descriptors += [descriptor]
        self.declare_parameter(descriptor.name, self.topic_projected_depth, descriptor)

    def _init_parameter_topic_projected_points(self):
        descriptor = ParameterDescriptor(
            name="topic_projected_points",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the colored pointcloud topic (for publisher)",
            read_only=False,
        )
        self.parameter_descriptors += [descriptor]
        self.declare_parameter(descriptor.name, self.topic_projected_points, descriptor)

    def _init_parameter_slop_synchronizer(self):
        descriptor = ParameterDescriptor(
            name="slop_synchronizer",
            type=ParameterType.PARAMETER_DOUBLE,
            description="Maximum time disparity between associated image and pointcloud messages",
            read_only=False,
            floating_point_range=(
                FloatingPointRange(
                    from_value=0.0,
                    to_value=2.0,
                    step=0.0,
                ),
            ),
        )
        self.parameter_descriptors += [descriptor]
        self.declare_parameter(descriptor.name, self.slop_synchronizer, descriptor)

    def _init_parameter_color_invalid(self):
        descriptor = ParameterDescriptor(
            name="color_invalid",
            type=ParameterType.PARAMETER_STRING,
            description="Rgb color given to invalid points",
            read_only=False,
        )
        self.parameter_descriptors += [descriptor]
        self.declare_parameter(descriptor.name, self.color_invalid, descriptor)

    def _init_parameter_factor_downsampling(self):
        descriptor = ParameterDescriptor(
            name="factor_downsampling",
            type=ParameterType.PARAMETER_INTEGER,
            description="Downsampling factor used with knn interpolation",
            read_only=False,
            integer_range=(
                IntegerRange(
                    from_value=1,
                    to_value=16,
                    step=1,
                ),
            ),
        )
        self.parameter_descriptors += [descriptor]
        self.declare_parameter(descriptor.name, self.factor_downsampling, descriptor)

    def _init_parameter_k_knn(self):
        descriptor = ParameterDescriptor(
            name="k_knn",
            type=ParameterType.PARAMETER_INTEGER,
            description="Number of neighbors used with knn interpolation",
            read_only=False,
            integer_range=(
                IntegerRange(
                    from_value=1,
                    to_value=10,
                    step=1,
                ),
            ),
        )
        self.parameter_descriptors += [descriptor]
        self.declare_parameter(descriptor.name, self.k_knn, descriptor)

    def _init_parameter_mode_interpolation(self):
        descriptor = ParameterDescriptor(
            name="mode_interpolation",
            type=ParameterType.PARAMETER_STRING,
            description="Interpolation mode for upsampling used with knn interpolation",
            read_only=False,
        )
        self.parameter_descriptors += [descriptor]
        self.declare_parameter(descriptor.name, self.mode_interpolation, descriptor)

    def _init_parameter_use_knn_interpolation(self):
        descriptor = ParameterDescriptor(
            name="use_knn_interpolation",
            type=ParameterType.PARAMETER_BOOL,
            description="Usage of knn interpolation",
            read_only=False,
        )
        self.parameter_descriptors += [descriptor]
        self.declare_parameter(descriptor.name, self.use_knn_interpolation, descriptor)

    def parameter_changed(self, parameter):
        try:
            func_update = getattr(NodeProjectionDoubleSphere, f"update_{parameter.name}")
            success, reason = func_update(self, parameter.value)
        except Exception as e:
            self.get_logger().info(f"Error: {e}")

        return success, reason

    def update_name_frame_camera(self, name_frame_camera):
        self.name_frame_camera = name_frame_camera

        success = True
        reason = ""
        return success, reason

    def update_name_frame_lidar(self, name_frame_lidar):
        self.name_frame_lidar = name_frame_lidar

        success = True
        reason = ""
        return success, reason

    def update_topic_image(self, topic_image):
        self.topic_image = topic_image

        self._init_subscribers()

        success = True
        reason = ""
        return success, reason

    def update_topic_info(self, topic_info):
        self.topic_info = topic_info

        self._init_subscribers()

        success = True
        reason = ""
        return success, reason

    def update_topic_points(self, topic_points):
        self.topic_points = topic_points

        self._init_subscribers()

        success = True
        reason = ""
        return success, reason

    def update_topic_projected_depth(self, topic_projected_depth):
        self.topic_projected_depth = topic_projected_depth

        self._init_publishers()

        success = True
        reason = ""
        return success, reason

    def update_topic_projected_points(self, topic_projected_points):
        self.topic_projected_points = topic_projected_points

        self._init_publishers()

        success = True
        reason = ""
        return success, reason

    def update_slop_synchronizer(self, slop_synchronizer):
        self.slop_synchronizer = slop_synchronizer

        self._init_subscribers()

        success = True
        reason = ""
        return success, reason

    def update_color_invalid(self, color_invalid):
        # TODO: This is a little dangerous
        self.color_invalid = eval(color_invalid)

        self.sampler_color = SamplerColor(color_invalid=self.color_invalid)

        success = True
        reason = ""
        return success, reason

    def update_factor_downsampling(self, factor_downsampling):
        self.factor_downsampling = factor_downsampling

        self.sampler_depth = SamplerDepth(factor_downsampling=self.factor_downsampling, k_knn=self.k_knn, mode_interpolation=self.mode_interpolation)

        success = True
        reason = ""
        return success, reason

    def update_k_knn(self, k_knn):
        self.k_knn = k_knn

        self.sampler_depth = SamplerDepth(factor_downsampling=self.factor_downsampling, k_knn=self.k_knn, mode_interpolation=self.mode_interpolation)

        success = True
        reason = ""
        return success, reason

    def update_mode_interpolation(self, mode_interpolation):
        self.mode_interpolation = mode_interpolation

        self.sampler_depth = SamplerDepth(factor_downsampling=self.factor_downsampling, k_knn=self.k_knn, mode_interpolation=self.mode_interpolation)

        success = True
        reason = ""
        return success, reason

    def update_use_knn_interpolation(self, use_knn_interpolation):
        self.use_knn_interpolation = use_knn_interpolation

        success = True
        reason = ""
        return success, reason
