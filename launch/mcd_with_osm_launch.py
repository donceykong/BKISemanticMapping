"""Launch MCD node and OSM visualizer node together as independent nodes."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    pkg_arg = DeclareLaunchArgument(
        'pkg',
        default_value='semantic_bki',
        description='Package name'
    )
    
    method_arg = DeclareLaunchArgument(
        'method',
        default_value='semantic_bki',
        description='Method name'
    )
    
    dataset_arg = DeclareLaunchArgument(
        'dataset',
        default_value='mcd',
        description='Dataset name'
    )
    
    osm_dataset_arg = DeclareLaunchArgument(
        'osm_dataset',
        default_value='osm_visualizer',
        description='OSM visualizer config dataset name'
    )
    
    return LaunchDescription([
        pkg_arg,
        method_arg,
        dataset_arg,
        osm_dataset_arg,
        OpaqueFunction(function=launch_setup)
    ])


def launch_setup(context):
    # Get launch argument values
    method = context.launch_configurations.get('method', 'semantic_bki')
    dataset = context.launch_configurations.get('dataset', 'mcd')
    osm_dataset = context.launch_configurations.get('osm_dataset', 'osm_visualizer')
    
    # Get package share directory
    pkg_share_dir = get_package_share_directory('semantic_bki')
    
    # Construct paths
    method_config_path = os.path.join(pkg_share_dir, 'config', 'methods', f'{method}.yaml')
    data_config_path = os.path.join(pkg_share_dir, 'config', 'datasets', f'{dataset}.yaml')
    osm_config_path = os.path.join(pkg_share_dir, 'config', 'datasets', f'{osm_dataset}.yaml')
    data_dir_path = os.path.join(pkg_share_dir, 'data', dataset)
    calib_file_path = os.path.join(pkg_share_dir, 'data', dataset, 'hhs_calib.yaml')
    rviz_config_path = os.path.join(pkg_share_dir, 'rviz', 'mcd_node.rviz')
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_config_path],
        output='screen'
    )
    
    # MCD node (scan processing)
    mcd_node = Node(
        package='semantic_bki',
        executable='mcd_node',
        name='mcd_node',
        output='screen',
        parameters=[
            {'dir': data_dir_path},
            {'calibration_file': calib_file_path},
            method_config_path,
            data_config_path
        ]
    )
    
    # OSM visualizer node (independent, publishes OSM buildings)
    # Pass data_dir so pose file can be found relative to it
    osm_node = Node(
        package='semantic_bki',
        executable='osm_visualizer_node',
        name='osm_visualizer_node',
        output='screen',
        parameters=[
            osm_config_path,
            {'data_dir': data_dir_path}
        ]
    )
    
    return [rviz_node, mcd_node, osm_node]
