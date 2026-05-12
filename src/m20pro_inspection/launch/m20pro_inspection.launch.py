import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    inspection_share = get_package_share_directory("m20pro_inspection")
    default_config = os.path.join(inspection_share, "config", "yolov8_inspection.yaml")
    default_model = os.path.join(inspection_share, "models", "inspection.rknn")
    default_classes = os.path.join(inspection_share, "models", "classes.txt")

    config_file = LaunchConfiguration("config_file")
    model_path = LaunchConfiguration("model_path")
    class_names_path = LaunchConfiguration("class_names_path")
    backend = LaunchConfiguration("backend")
    source_type = LaunchConfiguration("source_type")
    rtsp_url = LaunchConfiguration("rtsp_url")
    image_topic = LaunchConfiguration("image_topic")
    camera_name = LaunchConfiguration("camera_name")

    return LaunchDescription([
        DeclareLaunchArgument("config_file", default_value=default_config),
        DeclareLaunchArgument("model_path", default_value=default_model),
        DeclareLaunchArgument("class_names_path", default_value=default_classes),
        DeclareLaunchArgument("backend", default_value="auto"),
        DeclareLaunchArgument("source_type", default_value="rtsp"),
        DeclareLaunchArgument("rtsp_url", default_value="rtsp://10.21.31.103:8554/video1"),
        DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument("camera_name", default_value="front_wide"),
        Node(
            package="m20pro_inspection",
            executable="yolov8_inspection",
            name="m20pro_yolov8_inspection",
            output="screen",
            parameters=[
                config_file,
                {
                    "model_path": model_path,
                    "class_names_path": class_names_path,
                    "backend": backend,
                    "source_type": source_type,
                    "rtsp_url": rtsp_url,
                    "image_topic": image_topic,
                    "camera_name": camera_name,
                },
            ],
        ),
    ])
