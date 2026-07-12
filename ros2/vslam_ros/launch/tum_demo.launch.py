"""Demo TUM RGB-D: los 4 nodos (dataset + frontend + backend + mapper).

    ros2 launch vslam_ros tum_demo.launch.py \
        root:=/workspace/data/tum/rgbd_dataset_freiburg1_desk rate:=15.0 \
        rviz:=true      # criterio de v0.8 (necesita WSLg, docker/README.md)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "root",
            default_value="/workspace/data/tum/rgbd_dataset_freiburg1_desk"),
        DeclareLaunchArgument("rate", default_value="15.0"),
        DeclareLaunchArgument("rviz", default_value="false"),
        Node(package="vslam_ros", executable="dataset_node",
             name="vslam_dataset", output="screen",
             parameters=[{"root": LaunchConfiguration("root"),
                          "rate": LaunchConfiguration("rate")}]),
        Node(package="vslam_ros", executable="frontend_node",
             name="vslam_frontend", output="screen"),
        Node(package="vslam_ros", executable="backend_node",
             name="vslam_backend", output="screen"),
        Node(package="vslam_ros", executable="mapper_node",
             name="vslam_mapper", output="screen",
             parameters=[{"seed_step": 4, "map_period": 1.0}]),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             output="screen",
             arguments=["-d", "/workspace/ros2/vslam_ros/rviz/vslam.rviz"],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
