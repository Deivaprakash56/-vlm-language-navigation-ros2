"""Gazebo TurtleBot3 world + Nav2 (from nav2_bringup) + lang_nav node.

  ros2 launch lang_nav lang_nav_sim.launch.py                    # Claude backend
  ros2 launch lang_nav lang_nav_sim.launch.py backend:=keyword   # offline, no API key
  ros2 launch lang_nav lang_nav_sim.launch.py headless:=False    # show Gazebo GUI
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    nav2_dir = get_package_share_directory('nav2_bringup')
    backend = LaunchConfiguration('backend')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_dir, 'launch', 'tb3_simulation_launch.py')),
        launch_arguments={'headless': LaunchConfiguration('headless'),
                          'use_rviz': LaunchConfiguration('use_rviz')}.items())

    init_pose = Node(package='lang_nav', executable='set_initial_pose', output='screen',
                     parameters=[{'use_sim_time': True, 'x': -2.0, 'y': -0.5, 'yaw': 0.0}])

    lang_nav = Node(
        package='lang_nav', executable='lang_nav_node', name='lang_nav', output='screen',
        parameters=[{'backend': backend, 'use_sim_time': True,
                     'model': LaunchConfiguration('model'),
                     'effort': LaunchConfiguration('effort')}])

    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='claude'),
        DeclareLaunchArgument('model', default_value='claude-opus-5-5'),
        DeclareLaunchArgument('effort', default_value='low'),
        DeclareLaunchArgument('headless', default_value='True'),
        DeclareLaunchArgument('use_rviz', default_value='True'),
        # don't let gzserver hang on the retired online model database
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
        sim,
        TimerAction(period=5.0, actions=[init_pose]),
        TimerAction(period=8.0, actions=[lang_nav]),
    ])
