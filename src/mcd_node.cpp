#include <string>
#include <iostream>
#include <ros/ros.h>
#include <ros/package.h>
#include <cstdlib>
#include <fstream>

#include "bkioctomap.h"
#include "markerarray_pub.h"
#include "mcd_util.h"
#include "osm_visualizer.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "mcd_node");
    ros::NodeHandle nh("~");

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

    nh.param<int>("block_depth", block_depth, block_depth);
    nh.param<double>("sf2", sf2, sf2);
    nh.param<double>("ell", ell, ell);
    nh.param<float>("prior", prior, prior);
    nh.param<float>("var_thresh", var_thresh, var_thresh);
    nh.param<double>("free_thresh", free_thresh, free_thresh);
    nh.param<double>("occupied_thresh", occupied_thresh, occupied_thresh);
    nh.param<double>("resolution", resolution, resolution);
    nh.param<int>("num_class", num_class, num_class);
    nh.param<double>("free_resolution", free_resolution, free_resolution);
    nh.param<double>("ds_resolution", ds_resolution, ds_resolution);
    nh.param<int>("scan_num", scan_num, scan_num);
    nh.param<double>("max_range", max_range, max_range);
    nh.param<int>("skip_frames", skip_frames, skip_frames);

    // MCD
    nh.param<std::string>("dir", dir, dir);
    nh.param<std::string>("input_data_prefix", input_data_prefix, input_data_prefix);
    nh.param<std::string>("input_label_prefix", input_label_prefix, input_label_prefix);
    nh.param<std::string>("lidar_pose_file", lidar_pose_file, lidar_pose_file);
    nh.param<std::string>("gt_label_prefix", gt_label_prefix, gt_label_prefix);
    nh.param<std::string>("evaluation_result_prefix", evaluation_result_prefix, evaluation_result_prefix);
    nh.param<bool>("query", query, query);
    nh.param<bool>("visualize", visualize, visualize);
    
    // Color configuration
    std::string colors_file;
    nh.param<std::string>("colors_file", colors_file, "");
    
    // OSM visualization
    bool show_osm = false;
    std::string osm_bin_file;
    nh.param<bool>("show_osm", show_osm, false);
    nh.param<std::string>("osm_bin_file", osm_bin_file, "");

    ROS_INFO_STREAM("Parameters:" << std::endl <<
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

    
    ///////// Build Map /////////////////////
    MCDData mcd_data(nh, resolution, block_depth, sf2, ell, num_class, free_thresh, occupied_thresh, var_thresh, ds_resolution, free_resolution, max_range, map_topic, prior);
    mcd_data.read_lidar_poses(dir + '/' + lidar_pose_file);
    
    // Load body-to-lidar calibration from hhs_calib.yaml (if provided via ROS parameters)
    // The calibration file can be loaded via rosparam or included in the launch file
    if (!mcd_data.load_calibration_from_params()) {
      ROS_FATAL("Failed to load body-to-lidar calibration! Cannot proceed without calibration.");
      return 1;
    }
    
    // Load colors from YAML file specified in colors_file parameter
    if (colors_file.empty()) {
      ROS_WARN_STREAM("No colors_file specified in dataset config. Using default hardcoded colors.");
    } else {
      std::string pkg_path = ros::package::getPath("semantic_bki");
      if (!pkg_path.empty()) {
        std::string colors_file_path = pkg_path + "/config/datasets/" + colors_file;
        ROS_INFO_STREAM("Loading colors from file specified in config: " << colors_file_path);
        // Use rosparam command to load the file into ROS parameters
        std::string rosparam_cmd = "rosparam load " + colors_file_path + " / 2>/dev/null";
        int result = system(rosparam_cmd.c_str());
        if (result != 0) {
          ROS_WARN_STREAM("Failed to load colors file: " << colors_file_path << ". Using default hardcoded colors.");
        } else {
          ROS_INFO_STREAM("Successfully loaded colors from: " << colors_file_path);
        }
      } else {
        ROS_WARN_STREAM("Could not find semantic_bki package path. Using default hardcoded colors.");
      }
    }
    
    // Load colors from ROS parameters (loaded from colors_file above)
    mcd_data.load_colors_from_params();
    
    // Load and visualize OSM geometries if enabled (do this BEFORE processing scans)
    semantic_bki::OSMVisualizer* osm_visualizer = nullptr;
    if (show_osm && !osm_bin_file.empty()) {
        ROS_INFO_STREAM("OSM visualization enabled. Loading OSM bin file: " << osm_bin_file);
        
        // Get package path
        std::string pkg_path = ros::package::getPath("semantic_bki");
        if (pkg_path.empty()) {
            ROS_WARN("Could not find semantic_bki package path. Cannot load OSM data.");
        } else {
            ROS_INFO_STREAM("Package path: " << pkg_path);
            
            // Construct full path to OSM bin file
            std::string full_osm_bin_path;
            if (osm_bin_file[0] == '/') {
                // Absolute path
                full_osm_bin_path = osm_bin_file;
            } else {
                // Relative to dir (which already contains the full path to data/mcd)
                full_osm_bin_path = dir + "/" + osm_bin_file;
            }
            
            ROS_INFO_STREAM("Looking for OSM bin file at: " << full_osm_bin_path);
            
            // Check if bin file exists
            std::ifstream bin_check(full_osm_bin_path);
            if (!bin_check.good()) {
                ROS_ERROR_STREAM("OSM bin file not found: " << full_osm_bin_path);
            } else {
                bin_check.close();
                
                // Load directly from binary file (created by create_map_OSM_BEV_GEOM.py)
                // The file is already in raw binary format, similar to lidar .bin files
                std::string binary_file = full_osm_bin_path;
                
                // Create OSM visualizer and load from binary file
                std::string osm_topic = "/osm_geometries";
                ROS_INFO_STREAM("Creating OSM visualizer with topic: " << osm_topic);
                ROS_INFO_STREAM("Loading OSM geometries from binary file: " << binary_file);
                osm_visualizer = new semantic_bki::OSMVisualizer(nh, osm_topic);
                
                // Wait a bit for publisher to be ready
                ros::Duration(0.1).sleep();
                
                if (osm_visualizer->loadFromBinary(binary_file)) {
                    ROS_INFO("Successfully loaded OSM geometries.");
                    
                    // Transform OSM data to be relative to first pose origin (same as scans/map)
                    // This applies the same transformation: first_pose_inverse
                    Eigen::Matrix4d original_first_pose = mcd_data.getOriginalFirstPose();
                    osm_visualizer->transformToFirstPoseOrigin(original_first_pose);
                    
                    ROS_INFO("Starting periodic publishing...");
                    // Publish immediately
                    osm_visualizer->publish();
                    // Start periodic publishing at 1 Hz
                    osm_visualizer->startPeriodicPublishing(1.0);
                    ROS_INFO_STREAM("OSM geometries will be published continuously to topic: " << osm_topic);
                } else {
                    ROS_ERROR("Failed to load OSM geometries from binary file.");
                    delete osm_visualizer;
                    osm_visualizer = nullptr;
                }
            }
        }
    } else {
        if (!show_osm) {
            ROS_INFO("OSM visualization is disabled (show_osm: false). Set show_osm: true in mcd.yaml to enable.");
        } else if (osm_bin_file.empty()) {
            ROS_WARN("OSM visualization is enabled but osm_bin_file is not specified");
        }
    }

    // Now process scans (OSM markers are already publishing)
    mcd_data.set_up_evaluation(dir + '/' + gt_label_prefix, dir + '/' + evaluation_result_prefix);
    mcd_data.process_scans(dir + '/' + input_data_prefix, dir + '/' + input_label_prefix, scan_num, skip_frames, query, visualize);

    ros::spin();
    
    if (osm_visualizer) {
        delete osm_visualizer;
    }
    
    return 0;
}
