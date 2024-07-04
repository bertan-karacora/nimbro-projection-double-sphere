from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

launch_args = [
    DeclareLaunchArgument("ids_name_frame_camera", description="Name of the camera frame in the tf transform tree", default_value="camera_ids_link"),
    DeclareLaunchArgument("ids_name_frame_lidar", description="Name of the lidar frame in the tf transform tree", default_value="os_sensor_link"),
    DeclareLaunchArgument("ids_topic_image", description="Name of the rgb image topic (for subscriber)", default_value="/camera_ids/image_color"),
    DeclareLaunchArgument("ids_topic_info", description="Name of the camera info topic (for subscriber)", default_value="/camera_ids/camera_info"),
    DeclareLaunchArgument("ids_topic_points", description="Name of the pointcloud topic (for subscriber)", default_value="/ouster/points"),
    DeclareLaunchArgument("ids_topic_projected_depth", description="Name of the projected depth topic (for publisher)", default_value="/camera_ids/projected/depth/image"),
    DeclareLaunchArgument("ids_topic_projected_points", description="Name of the colored pointcloud topic (for publisher)", default_value="/ouster/projected/points"),
    DeclareLaunchArgument("ids_slop_synchronizer", description="Maximum time disparity between associated image and pointcloud messages", default_value="0.2"),
    DeclareLaunchArgument("ids_color_invalid", description="Rgb color given to invalid points", default_value="(255, 87, 51)"),
    DeclareLaunchArgument("ids_factor_downsampling", description="Downsampling factor used with knn interpolation", default_value="8"),
    DeclareLaunchArgument("ids_k_knn", description="Number of neighbors used with knn interpolation", default_value="1"),
    DeclareLaunchArgument(
        "ids_mode_interpolation",
        description="Interpolation mode for upsampling used with knn interpolation",
        choices=["nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"],
        default_value="nearest",
    ),
    DeclareLaunchArgument("ids_use_knn_interpolation", description="Usage of knn interpolation", choices=["True", "False"], default_value="True"),
]

node = Node(
    package="nimbro_projection_double_sphere",
    namespace="",
    executable="spin",
    name="nimbro_projection_double_sphere",
    output="screen",
    parameters=[
        {
            "name_frame_camera": LaunchConfiguration("ids_name_frame_camera"),
            "name_frame_lidar": LaunchConfiguration("ids_name_frame_lidar"),
            "topic_image": LaunchConfiguration("ids_topic_image"),
            "topic_info": LaunchConfiguration("ids_topic_info"),
            "topic_points": LaunchConfiguration("ids_topic_points"),
            "topic_projected_depth": LaunchConfiguration("ids_topic_projected_depth"),
            "topic_projected_points": LaunchConfiguration("ids_topic_projected_points"),
            "slop_synchronizer": LaunchConfiguration("ids_slop_synchronizer"),
            "color_invalid": LaunchConfiguration("ids_color_invalid"),
            "factor_downsampling": LaunchConfiguration("ids_factor_downsampling"),
            "k_knn": LaunchConfiguration("ids_k_knn"),
            "mode_interpolation": LaunchConfiguration("ids_mode_interpolation"),
            "use_knn_interpolation": LaunchConfiguration("ids_use_knn_interpolation"),
        }
    ],
    respawn=True,
    respawn_delay=1.0,
)


def generate_launch_description():
    ld = LaunchDescription(launch_args)
    ld.add_action(node)
    return ld
