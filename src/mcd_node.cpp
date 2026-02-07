#include <string>
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <fstream>

#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include "bkioctomap.h"
#include "markerarray_pub.h"
#include "mcd_util.h"
#include "osm_visualizer.h"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("mcd_node");
    
    if (!node) {
        RCLCPP_WARN_STREAM(rclcpp::get_logger("mcd_node"), "WARNING: Failed to create ROS2 node!");
        return 1;
    }
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Node created successfully");

    std::string map_topic("/occupied_cells_vis_array");
    int block_depth = 4;
    double sf2 = 1.0;
    double ell = 1.0;
    float prior = 1.0f;
    float var_thresh = 1.0f;
    double free_thresh = 0.3;
    double occupied_thresh = 0.7;
    double resolution = 0.1;
    int num_class = 2;
    double free_resolution = 0.5;
    double ds_resolution = 0.1;
    int scan_num = 0;
    double max_range = -1;
    int skip_frames = 0;
    
    // MCD Dataset
    std::string dir;
    std::string input_data_prefix;
    std::string input_label_prefix;
    std::string lidar_pose_file;
    std::string gt_label_prefix;
    std::string evaluation_result_prefix;
    bool query = false;
    bool visualize = false;

    // Declare parameters
    node->declare_parameter<std::string>("map_topic", map_topic);
    node->declare_parameter<int>("block_depth", block_depth);
    node->declare_parameter<double>("sf2", sf2);
    node->declare_parameter<double>("ell", ell);
    node->declare_parameter<float>("prior", prior);
    node->declare_parameter<float>("var_thresh", var_thresh);
    node->declare_parameter<double>("free_thresh", free_thresh);
    node->declare_parameter<double>("occupied_thresh", occupied_thresh);
    node->declare_parameter<double>("resolution", resolution);
    node->declare_parameter<int>("num_class", num_class);
    node->declare_parameter<double>("free_resolution", free_resolution);
    node->declare_parameter<double>("ds_resolution", ds_resolution);
    node->declare_parameter<int>("scan_num", scan_num);
    node->declare_parameter<double>("max_range", max_range);
    node->declare_parameter<int>("skip_frames", skip_frames);
    node->declare_parameter<std::string>("dir", dir);
    node->declare_parameter<std::string>("input_data_prefix", input_data_prefix);
    node->declare_parameter<std::string>("input_label_prefix", input_label_prefix);
    node->declare_parameter<std::string>("lidar_pose_file", lidar_pose_file);
    node->declare_parameter<std::string>("gt_label_prefix", gt_label_prefix);
    node->declare_parameter<std::string>("evaluation_result_prefix", evaluation_result_prefix);
    node->declare_parameter<bool>("query", query);
    node->declare_parameter<bool>("visualize", visualize);
    node->declare_parameter<std::string>("colors_file", "");
    node->declare_parameter<bool>("show_osm", false);
    node->declare_parameter<std::string>("osm_bin_file", "");
    node->declare_parameter<std::string>("calibration_file", "");

    // Get parameters
    node->get_parameter<std::string>("map_topic", map_topic);
    node->get_parameter<int>("block_depth", block_depth);
    node->get_parameter<double>("sf2", sf2);
    node->get_parameter<double>("ell", ell);
    node->get_parameter<float>("prior", prior);
    node->get_parameter<float>("var_thresh", var_thresh);
    node->get_parameter<double>("free_thresh", free_thresh);
    node->get_parameter<double>("occupied_thresh", occupied_thresh);
    node->get_parameter<double>("resolution", resolution);
    node->get_parameter<int>("num_class", num_class);
    node->get_parameter<double>("free_resolution", free_resolution);
    node->get_parameter<double>("ds_resolution", ds_resolution);
    node->get_parameter<int>("scan_num", scan_num);
    node->get_parameter<double>("max_range", max_range);
    node->get_parameter<int>("skip_frames", skip_frames);
    node->get_parameter<std::string>("dir", dir);
    node->get_parameter<std::string>("input_data_prefix", input_data_prefix);
    node->get_parameter<std::string>("input_label_prefix", input_label_prefix);
    node->get_parameter<std::string>("lidar_pose_file", lidar_pose_file);
    node->get_parameter<std::string>("gt_label_prefix", gt_label_prefix);
    node->get_parameter<std::string>("evaluation_result_prefix", evaluation_result_prefix);
    node->get_parameter<bool>("query", query);
    node->get_parameter<bool>("visualize", visualize);
    
    // Color configuration
    std::string colors_file;
    node->get_parameter<std::string>("colors_file", colors_file);
    
    // OSM visualization
    bool show_osm = false;
    std::string osm_bin_file;
    node->get_parameter<bool>("show_osm", show_osm);
    node->get_parameter<std::string>("osm_bin_file", osm_bin_file);
    
    // Calibration file
    std::string calibration_file;
    node->get_parameter<std::string>("calibration_file", calibration_file);
    
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: All parameters retrieved. dir=" << dir << ", lidar_pose_file=" << lidar_pose_file << ", calibration_file=" << calibration_file);

    RCLCPP_INFO_STREAM(node->get_logger(), "Parameters:" << std::endl <<
      "block_depth: " << block_depth << std::endl <<
      "sf2: " << sf2 << std::endl <<
      "ell: " << ell << std::endl <<
      "prior:" << prior << std::endl <<
      "var_thresh: " << var_thresh << std::endl <<
      "free_thresh: " << free_thresh << std::endl <<
      "occupied_thresh: " << occupied_thresh << std::endl <<
      "resolution: " << resolution << std::endl <<
      "num_class: " << num_class << std::endl << 
      "free_resolution: " << free_resolution << std::endl <<
      "ds_resolution: " << ds_resolution << std::endl <<
      "scan_num: " << scan_num << std::endl <<
      "max_range: " << max_range << std::endl <<
      "skip_frames: " << skip_frames << std::endl <<

      "MCD:" << std::endl <<
      "dir: " << dir << std::endl <<
      "input_data_prefix: " << input_data_prefix << std::endl <<
      "input_label_prefix: " << input_label_prefix << std::endl <<
      "lidar_pose_file: " << lidar_pose_file << std::endl <<
      "gt_label_prefix: " << gt_label_prefix << std::endl <<
      "evaluation_result_prefix: " << evaluation_result_prefix << std::endl <<
      "query: " << query << std::endl <<
      "visualize:" << visualize
      );

    
    // Publish static transform for "map" frame (required for RViz visualization)
    // This ensures the map frame exists even before any scans are processed
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to publish static transform");
    try {
        tf2_ros::StaticTransformBroadcaster static_tf_broadcaster(node);
        geometry_msgs::msg::TransformStamped static_transform;
        static_transform.header.stamp = node->now();
        static_transform.header.frame_id = "map";
        static_transform.child_frame_id = "odom";  // Common base frame, or use "base_link" if preferred
        static_transform.transform.translation.x = 0.0;
        static_transform.transform.translation.y = 0.0;
        static_transform.transform.translation.z = 0.0;
        static_transform.transform.rotation.x = 0.0;
        static_transform.transform.rotation.y = 0.0;
        static_transform.transform.rotation.z = 0.0;
        static_transform.transform.rotation.w = 1.0;
        static_tf_broadcaster.sendTransform(static_transform);
        RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Static transform published successfully");
        RCLCPP_INFO(node->get_logger(), "Published static transform: map -> odom (identity)");
    } catch (const std::exception& e) {
        RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: Exception publishing static transform: " << e.what());
        RCLCPP_ERROR_STREAM(node->get_logger(), "Exception publishing static transform: " << e.what());
    }
    
    ///////// Build Map /////////////////////
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to create MCDData object");
    MCDData mcd_data(node, resolution, block_depth, sf2, ell, num_class, free_thresh, occupied_thresh, var_thresh, ds_resolution, free_resolution, max_range, map_topic, prior);
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: MCDData object created successfully");
    
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to read lidar poses from: " << (dir + '/' + lidar_pose_file));
    if (!mcd_data.read_lidar_poses(dir + '/' + lidar_pose_file)) {
        RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: Failed to read lidar poses!");
        return 1;
    }
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Lidar poses read successfully");
    
    // Load body-to-lidar calibration from hhs_calib.yaml (if provided via ROS parameters)
    // The calibration file can be loaded via rosparam or included in the launch file
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to load calibration from params");
    if (!mcd_data.load_calibration_from_params()) {
      RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: Failed to load body-to-lidar calibration!");
      RCLCPP_FATAL(node->get_logger(), "Failed to load body-to-lidar calibration! Cannot proceed without calibration.");
      return 1;
    }
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Calibration loaded successfully");
    
    // Load colors from YAML file specified in colors_file parameter
    // Load directly into MarkerArrayPub instead of using ROS parameters
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to load colors");
    if (colors_file.empty()) {
      RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: No colors_file specified in dataset config. Using default hardcoded colors.");
    } else {
      std::string pkg_path = ament_index_cpp::get_package_share_directory("semantic_bki");
      if (pkg_path.empty()) {
        RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: Could not find semantic_bki package path!");
      } else {
        std::string colors_file_path = pkg_path + "/config/datasets/" + colors_file;
        RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Colors file path: " << colors_file_path);
        RCLCPP_INFO_STREAM(node->get_logger(), "Loading colors from file specified in config: " << colors_file_path);
        // Load colors directly into MarkerArrayPub (bypasses ROS parameter system)
        if (mcd_data.load_colors_from_yaml(colors_file_path)) {
          RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Colors loaded successfully");
          RCLCPP_INFO_STREAM(node->get_logger(), "Successfully loaded colors from: " << colors_file_path);
        } else {
          RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: Failed to load colors from: " << colors_file_path << ". Using default hardcoded colors.");
        }
      }
    }
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Color loading completed");
    
    // Load and visualize OSM geometries if enabled (do this BEFORE processing scans)
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to check OSM visualization. show_osm=" << show_osm << ", osm_bin_file=" << osm_bin_file);
    semantic_bki::OSMVisualizer* osm_visualizer = nullptr;
    if (show_osm && !osm_bin_file.empty()) {
        RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: OSM visualization enabled. Loading OSM bin file: " << osm_bin_file);
        RCLCPP_INFO_STREAM(node->get_logger(), "OSM visualization enabled. Loading OSM bin file: " << osm_bin_file);
        
        try {
            // Construct full path to OSM bin file
            std::string full_osm_bin_path;
            if (osm_bin_file[0] == '/') {
                // Absolute path
                full_osm_bin_path = osm_bin_file;
            } else {
                // Relative to dir (which already contains the full path to data/mcd)
                full_osm_bin_path = dir + "/" + osm_bin_file;
            }
            
            RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: OSM bin file path: " << full_osm_bin_path);
            RCLCPP_INFO_STREAM(node->get_logger(), "Looking for OSM bin file at: " << full_osm_bin_path);
            
            // Check if bin file exists
            std::ifstream bin_check(full_osm_bin_path);
            if (!bin_check.good()) {
                RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: OSM bin file not found: " << full_osm_bin_path);
                RCLCPP_ERROR_STREAM(node->get_logger(), "OSM bin file not found: " << full_osm_bin_path);
            } else {
                bin_check.close();
                RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: OSM bin file exists, about to create visualizer");
                
                // Load directly from binary file (created by create_map_OSM_BEV_GEOM.py)
                // The file is already in raw binary format, similar to lidar .bin files
                std::string binary_file = full_osm_bin_path;
                
                // Create OSM visualizer and load from binary file
                std::string osm_topic = "/osm_geometries";
                RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Creating OSM visualizer with topic: " << osm_topic);
                RCLCPP_INFO_STREAM(node->get_logger(), "Creating OSM visualizer with topic: " << osm_topic);
                RCLCPP_INFO_STREAM(node->get_logger(), "Loading OSM geometries from binary file: " << binary_file);
                osm_visualizer = new semantic_bki::OSMVisualizer(node, osm_topic);
                RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: OSM visualizer created successfully");
                
                // Wait a bit for publisher to be ready
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to load OSM geometries from binary file");
                if (osm_visualizer->loadFromBinary(binary_file)) {
                    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: OSM geometries loaded successfully");
                    RCLCPP_INFO(node->get_logger(), "Successfully loaded OSM geometries.");
                    
                    // Transform OSM data to be relative to first pose origin (same as scans/map)
                    // This applies the same transformation: first_pose_inverse
                    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to transform OSM data to first pose origin");
                    Eigen::Matrix4d original_first_pose = mcd_data.getOriginalFirstPose();
                    osm_visualizer->transformToFirstPoseOrigin(original_first_pose);
                    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: OSM data transformed successfully");
                    
                    RCLCPP_INFO(node->get_logger(), "Starting periodic publishing...");
                    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to call publish() immediately");
                    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: osm_visualizer pointer: " << osm_visualizer);
                    // Publish immediately
                    if (osm_visualizer) {
                        osm_visualizer->publish();
                        RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Immediate publish() call completed");
                    } else {
                        RCLCPP_ERROR(node->get_logger(), "CHECKPOINT: ERROR - osm_visualizer is null!");
                    }
                    // Start periodic publishing at 2 Hz (more frequent for better visibility)
                    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to start periodic publishing");
                    if (osm_visualizer) {
                        osm_visualizer->startPeriodicPublishing(2.0);
                        RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: OSM publishing started");
                        RCLCPP_INFO_STREAM(node->get_logger(), "OSM geometries will be published continuously to topic: " << osm_topic << " at 2 Hz");
                    } else {
                        RCLCPP_ERROR(node->get_logger(), "CHECKPOINT: ERROR - Cannot start periodic publishing, osm_visualizer is null!");
                    }
                    
                    // Give publisher time to be ready and ensure messages are sent
                    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Waiting for publisher to be ready...");
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Publisher ready, continuing...");
                } else {
                    RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: Failed to load OSM geometries from binary file!");
                    RCLCPP_ERROR(node->get_logger(), "Failed to load OSM geometries from binary file.");
                    delete osm_visualizer;
                    osm_visualizer = nullptr;
                }
            }
        } catch (const std::exception& e) {
            RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: Exception while creating/loading OSM visualizer: " << e.what());
            RCLCPP_ERROR_STREAM(node->get_logger(), "Exception while creating/loading OSM visualizer: " << e.what());
            if (osm_visualizer) {
                delete osm_visualizer;
                osm_visualizer = nullptr;
            }
        }
    } else {
        if (!show_osm) {
            RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: OSM visualization disabled");
            RCLCPP_INFO(node->get_logger(), "OSM visualization is disabled (show_osm: false). Set show_osm: true in mcd.yaml to enable.");
        } else if (osm_bin_file.empty()) {
            RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: OSM visualization enabled but osm_bin_file is empty!");
            RCLCPP_WARN(node->get_logger(), "OSM visualization is enabled but osm_bin_file is not specified");
        }
    }
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: OSM visualization setup completed");

    // Now process scans (OSM markers are already publishing)
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to set up evaluation");
    mcd_data.set_up_evaluation(dir + '/' + gt_label_prefix, dir + '/' + evaluation_result_prefix);
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Evaluation setup completed");
    
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to process scans. input_data_prefix=" << input_data_prefix << ", scan_num=" << scan_num);
    mcd_data.process_scans(dir + '/' + input_data_prefix, dir + '/' + input_label_prefix, scan_num, skip_frames, query, visualize);
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Scan processing completed, about to spin");
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Node pointer: " << node.get());
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Starting rclcpp::spin(node)...");
    
    rclcpp::spin(node);
    
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: rclcpp::spin() returned (node shutdown)");
    
    if (osm_visualizer) {
        delete osm_visualizer;
    }
    
    rclcpp::shutdown();
    return 0;
}
