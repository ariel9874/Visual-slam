"""Demo EuRoC ESTÉREO: el mismo pipeline con la otra personalidad del dataset.

    ros2 launch vslam_ros euroc_demo.launch.py \
        root:=/workspace/data/euroc/V1_01_easy rate:=10.0 rviz:=true

El dataset_node publica el par rectificado + profundidad SGBM (lección 37);
el frontend recibe stereo_bf/depth_max por parámetro (el bf del rig no viaja
en CameraInfo). bf=48.02 y depth_max=40 son los del rig V1 (docs/05 §3).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Consumidores antes que el productor (ver tum_demo.launch.py).
_LIFECYCLE_UP = ExecuteProcess(
    cmd=["bash", "-c",
         "sleep 4; for n in vslam_mapper vslam_backend vslam_frontend; do "
         "ros2 lifecycle set /$n configure && ros2 lifecycle set /$n activate; "
         "done"],
    name="lifecycle_bringup", output="screen")


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("root",
                              default_value="/workspace/data/euroc/V1_01_easy"),
        DeclareLaunchArgument("rate", default_value="10.0"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("stereo_bf", default_value="48.02"),
        Node(package="vslam_ros", executable="dataset_node",
             name="vslam_dataset", output="screen",
             parameters=[{"root": LaunchConfiguration("root"),
                          "rate": LaunchConfiguration("rate"),
                          "dataset": "euroc"}]),
        Node(package="vslam_ros", executable="frontend_node",
             name="vslam_frontend", output="screen",
             parameters=[{"stereo_bf": LaunchConfiguration("stereo_bf"),
                          "depth_max": 40.0}]),
        Node(package="vslam_ros", executable="backend_node",
             name="vslam_backend", output="screen"),
        Node(package="vslam_ros", executable="mapper_node",
             name="vslam_mapper", output="screen",
             parameters=[{"seed_step": 4, "map_period": 1.0}]),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             output="screen",
             arguments=["-d", "/workspace/ros2/vslam_ros/rviz/vslam.rviz"],
             condition=IfCondition(LaunchConfiguration("rviz"))),
        _LIFECYCLE_UP,
    ])
