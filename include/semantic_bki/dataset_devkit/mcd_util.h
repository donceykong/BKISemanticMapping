#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <ros/ros.h>
#include <ros/param.h>
#include <xmlrpcpp/XmlRpcValue.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include "bkioctomap.h"
#include "markerarray_pub.h"

class MCDData {
  public:
    MCDData(ros::NodeHandle& nh,
             double resolution, double block_depth,
             double sf2, double ell,
             int num_class, double free_thresh,
             double occupied_thresh, float var_thresh, 
             double ds_resolution,
             double free_resolution, double max_range,
             std::string map_topic,
             float prior)
      : nh_(nh)
      , resolution_(resolution)
      , ds_resolution_(ds_resolution)
      , free_resolution_(free_resolution)
      , max_range_(max_range) {
        map_ = new semantic_bki::SemanticBKIOctoMap(resolution, block_depth, num_class, sf2, ell, prior, var_thresh, free_thresh, occupied_thresh);
        m_pub_ = new semantic_bki::MarkerArrayPub(nh_, map_topic, resolution);
        // Publisher for individual scan point clouds
        pointcloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/mcd_scan_pointcloud", 1);
        // Identity transformation for MCD (poses are already in world frame)
        init_trans_to_ground_ = Eigen::Matrix4d::Identity();
        // Body to LiDAR transformation (must be loaded from calibration - no default identity)
        // Will be set by load_calibration_from_params() - if not set, will error
        body_to_lidar_tf_ = Eigen::Matrix4d::Zero();  // Set to zero to detect if not loaded
        original_first_pose_ = Eigen::Matrix4d::Identity();  // Will be set when poses are loaded
        scan_indices_.clear();  // Initialize scan indices vector
      }

    bool read_lidar_poses(const std::string lidar_pose_name) {
      std::ifstream fPoses;
      fPoses.open(lidar_pose_name.c_str());
      if (!fPoses.is_open()) {
        ROS_ERROR_STREAM("Cannot open pose file " << lidar_pose_name);
        return false;
      }
      
      // Skip header line if present
      std::string header_line;
      std::getline(fPoses, header_line);
      
      // Check if header contains column names (common in CSV)
      bool has_header = (header_line.find("num") != std::string::npos || 
                         header_line.find("timestamp") != std::string::npos ||
                         header_line.find("x") != std::string::npos);
      
      if (!has_header) {
        // No header, rewind to beginning
        fPoses.close();
        fPoses.open(lidar_pose_name.c_str());
      }

      while (!fPoses.eof()) {
        std::string s;
        std::getline(fPoses, s);
        
        // Skip empty lines and comments
        if (s.empty() || s[0] == '#') {
          continue;
        }

        std::stringstream ss(s);
        std::string token;
        std::vector<double> values;
        
        // Parse CSV line (handles comma-separated or space-separated)
        char delimiter = ',';
        if (s.find(',') == std::string::npos) {
          delimiter = ' ';
        }
        
        while (std::getline(ss, token, delimiter)) {
          try {
            double val = std::stod(token);
            values.push_back(val);
          } catch (...) {
            // Skip invalid tokens
            continue;
          }
        }

        // Expect at least 8 values: num, timestamp, x, y, z, qx, qy, qz, qw
        // Or 9 values if first is index
        if (values.size() < 8) {
          continue;
        }

        // Extract scan index (num) - first column in CSV
        int scan_index = (int)values[0];
        
        // Extract pose values (skip num and timestamp, they're at indices 0 and 1)
        double x = values[2];
        double y = values[3];
        double z = values[4];
        double qx = values[5];
        double qy = values[6];
        double qz = values[7];
        double qw = values.size() > 8 ? values[8] : 1.0;  // Default qw to 1.0 if not provided

        // Convert quaternion to rotation matrix
        Eigen::Quaterniond quat(qw, qx, qy, qz);
        quat.normalize();
        
        // Build transformation matrix
        Eigen::Matrix4d t_matrix = Eigen::Matrix4d::Identity();
        t_matrix.block<3, 3>(0, 0) = quat.toRotationMatrix();
        t_matrix(0, 3) = x;
        t_matrix(1, 3) = y;
        t_matrix(2, 3) = z;
        
        lidar_poses_.push_back(t_matrix);
        scan_indices_.push_back(scan_index);
      }
      
      fPoses.close();
      
      if (lidar_poses_.empty()) {
        ROS_ERROR_STREAM("No poses loaded from " << lidar_pose_name);
        return false;
      }
      
      // Store original first pose before transformation (needed for OSM data alignment)
      original_first_pose_ = lidar_poses_[0];
      
      // Make all poses relative to the first pose (set first pose to origin)
      // This means: transform all poses by the inverse of the first pose
      Eigen::Matrix4d first_pose_inverse = lidar_poses_[0].inverse();
      
      ROS_INFO_STREAM("First pose before alignment:");
      ROS_INFO_STREAM("  Translation: [" << lidar_poses_[0](0,3) << ", " << lidar_poses_[0](1,3) << ", " << lidar_poses_[0](2,3) << "]");
      
      // Transform all poses to be relative to the first pose
      for (size_t i = 0; i < lidar_poses_.size(); ++i) {
        lidar_poses_[i] = first_pose_inverse * lidar_poses_[i];
      }
      
      // Verify first pose is now at origin
      ROS_INFO_STREAM("After alignment - First pose should be identity:");
      ROS_INFO_STREAM("  Translation: [" << lidar_poses_[0](0,3) << ", " << lidar_poses_[0](1,3) << ", " << lidar_poses_[0](2,3) << "]");
      ROS_INFO_STREAM("Loaded " << lidar_poses_.size() << " poses from " << lidar_pose_name << " (all relative to first pose)");
      
      return true;
    }
    
    // Get the original first pose (before transformation to origin)
    // This is needed to align OSM data with the same coordinate frame
    Eigen::Matrix4d getOriginalFirstPose() const {
      return original_first_pose_;
    } 

    bool process_scans(std::string input_data_dir, std::string input_label_dir, int scan_num, int skip_frames, bool query, bool visualize) {
      semantic_bki::point3f origin;
      
      // Only process scans that have corresponding poses
      int num_scans_to_process = std::min(scan_num, (int)lidar_poses_.size());
      
      // Skip frames logic: if skip_frames=2, process frames 0, 3, 6, 9, etc. (skip 2, process 1)
      int processed_count = 0;
      int insertion_count = 0;
      
      for (int pose_idx = 0; pose_idx < num_scans_to_process; ++pose_idx) {
        // Skip frames based on skip_frames parameter
        if (skip_frames > 0 && processed_count % (skip_frames + 1) != 0) {
          processed_count++;
          continue;  // Skip this frame
        }
        processed_count++;
        // Get the actual scan file number from CSV
        int scan_file_num = scan_indices_[pose_idx];
        
        // Use 10-digit format for MCD file naming (e.g., 0000000011.bin)
        char scan_id_c[256];
        sprintf(scan_id_c, "%010d", scan_file_num);
        std::string scan_name = input_data_dir + "/" + std::string(scan_id_c) + ".bin";
        std::string label_name = input_label_dir + "/" + std::string(scan_id_c) + ".bin";
        
        pcl::PointCloud<pcl::PointXYZL>::Ptr cloud = mcd2pcl(scan_name, label_name);
        if (cloud->points.empty()) {
          ROS_WARN_STREAM("Empty point cloud at scan file " << scan_file_num << " (pose index " << pose_idx << "), skipping");
          continue;
        }

        Eigen::Matrix4d transform = lidar_poses_[pose_idx];  // This is body_to_world from pose
        
        // Verify body-to-lidar transform is loaded (should not be zero matrix)
        if (body_to_lidar_tf_.isZero(1e-10)) {
          ROS_FATAL_STREAM("ERROR: body_to_lidar_tf_ is not initialized! Calibration must be loaded before processing scans.");
          ROS_FATAL_STREAM("Call load_calibration_from_params() and ensure it returns true.");
          exit(1);
        }
        
        // Apply body-to-lidar transformation
        // The poses in CSV are body/IMU poses, need to transform from lidar frame to world frame
        // Following Python code: transform_matrix = body_to_world @ lidar_to_body
        // where lidar_to_body = inv(body_to_lidar_tf)
        Eigen::Matrix4d lidar_to_body = body_to_lidar_tf_.inverse();
        Eigen::Matrix4d lidar_to_map = transform * lidar_to_body;  // T_lidar_to_map = body_to_world * lidar_to_body
        
        // Publish TF transform from 'map' to 'lidar' frame
        // In ROS TF: when publishing parent->child, TF will transform points FROM child TO parent
        // So when we publish "map"->"lidar", TF will use it as: point_in_map = T_map_to_lidar * point_in_lidar
        // We have lidar_to_map which transforms points FROM lidar TO map: point_in_map = lidar_to_map * point_in_lidar
        // Therefore: T_map_to_lidar = lidar_to_map (use lidar_to_map directly, not its inverse!)
        // NOTE: This is correct because lidar_to_map already represents "where is lidar in map frame"
        
        // Extract rotation and translation for TF from lidar_to_map (not its inverse)
        tf::Transform tf_transform;
        Eigen::Matrix3d rotation = lidar_to_map.block<3, 3>(0, 0);
        Eigen::Vector3d translation = lidar_to_map.block<3, 1>(0, 3);
        
        tf::Matrix3x3 tf_rotation(
          rotation(0, 0), rotation(0, 1), rotation(0, 2),
          rotation(1, 0), rotation(1, 1), rotation(1, 2),
          rotation(2, 0), rotation(2, 1), rotation(2, 2)
        );
        tf::Vector3 tf_translation(translation(0), translation(1), translation(2));
        
        tf_transform.setBasis(tf_rotation);
        tf_transform.setOrigin(tf_translation);
        
        ros::Time current_time = ros::Time::now();
        tf_broadcaster_.sendTransform(tf::StampedTransform(tf_transform, current_time, "map", "lidar"));
        
        // Debug: Print transform info for first few scans
        if (pose_idx < 3) {
          ROS_INFO_STREAM("Scan " << pose_idx << " Transform info:");
          ROS_INFO_STREAM("  Body-to-world translation from CSV: [" << transform(0,3) << ", " << transform(1,3) << ", " << transform(2,3) << "]");
          ROS_INFO_STREAM("  Lidar-to-map translation: [" << lidar_to_map(0,3) << ", " << lidar_to_map(1,3) << ", " << lidar_to_map(2,3) << "]");
          ROS_INFO_STREAM("  TF translation (from lidar_to_map): [" << translation(0) << ", " << translation(1) << ", " << translation(2) << "]");
          
          // Verify: transform origin from lidar frame should give lidar_to_map translation
          Eigen::Vector4d lidar_origin(0, 0, 0, 1);
          Eigen::Vector4d map_origin_test = lidar_to_map * lidar_origin;
          ROS_INFO_STREAM("  Verification - lidar origin in map coords: [" << map_origin_test(0) << ", " << map_origin_test(1) << ", " << map_origin_test(2) << "]");
        }
        
        // Publish individual scan as PointCloud2 (in lidar frame, before transformation)
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.frame_id = "lidar";
        cloud_msg.header.stamp = current_time;
        pointcloud_pub_.publish(cloud_msg);
        
        if (pose_idx == 0) {
          ROS_INFO_STREAM("Published PointCloud2 with " << cloud->points.size() << " points in frame 'lidar'");
        }
        
        // Now transform cloud to world frame for map insertion
        pcl::transformPointCloud(*cloud, *cloud, lidar_to_map);
        
        origin.x() = transform(0, 3);
        origin.y() = transform(1, 3);
        origin.z() = transform(2, 3);
        
        map_->insert_pointcloud(*cloud, origin, ds_resolution_, free_resolution_, max_range_);
        insertion_count++;
        ROS_INFO_STREAM("Inserted point cloud " << scan_file_num << " (pose index " << pose_idx << ") from " << scan_name << " (" << cloud->points.size() << " points) - insertion #" << insertion_count);
        
        if (query) {
          // Query previous scans (use pose indices, not file numbers)
          for (int query_pose_idx = pose_idx - 10; query_pose_idx >= 0 && query_pose_idx <= pose_idx; ++query_pose_idx) {
            query_scan(input_data_dir, input_label_dir, query_pose_idx);
          }
        }

        if (visualize) {
          publish_map();
          // Small delay to allow rviz to process the visualization
          ros::Duration(0.1).sleep();
        }
      }
      
      // Final publish after all scans are processed
      if (visualize) {
        ROS_INFO_STREAM("All scans processed. Publishing final map visualization...");
        publish_map();
      }
      
      return true;
    }

    void publish_map() {
      m_pub_->clear_map(resolution_);
      int voxel_count = 0;
      for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
        if (it.get_node().get_state() == semantic_bki::State::OCCUPIED) {
          semantic_bki::point3f p = it.get_loc();
          m_pub_->insert_point3d_semantics(p.x(), p.y(), p.z(), it.get_size(), it.get_node().get_semantics(), 2);
          voxel_count++;
        }
      }
      m_pub_->publish();
      ROS_INFO_STREAM("Published map visualization with " << voxel_count << " occupied voxels in frame 'map'");
    }
    
    // Load colors from ROS parameters
    bool load_colors_from_params() {
      if (m_pub_) {
        return m_pub_->load_colors_from_params(nh_);
      }
      return false;
    }

    void set_up_evaluation(const std::string gt_label_dir, const std::string evaluation_result_dir) {
      gt_label_dir_ = gt_label_dir;
      evaluation_result_dir_ = evaluation_result_dir;
    }

    bool load_calibration_from_params() {
      // Load body-to-lidar transform from ROS parameters
      // Expected path: body/os_sensor/T (from hhs_calib.yaml)
      // Format: T is a list of 4 lists, each containing 4 doubles
      
      XmlRpc::XmlRpcValue body_param;
      if (!nh_.getParam("body", body_param)) {
        ROS_ERROR_STREAM("ERROR: 'body' parameter not found in ROS parameter server!");
        ROS_ERROR_STREAM("Make sure hhs_calib.yaml is loaded via rosparam in the launch file.");
        return false;
      }
      
      // Navigate to body/os_sensor/T
      if (body_param.getType() != XmlRpc::XmlRpcValue::TypeStruct) {
        ROS_ERROR_STREAM("ERROR: 'body' parameter is not a struct!");
        return false;
      }
      
      if (!body_param.hasMember("os_sensor")) {
        ROS_ERROR_STREAM("ERROR: 'body/os_sensor' not found in calibration!");
        return false;
      }
      
      XmlRpc::XmlRpcValue os_sensor_param = body_param["os_sensor"];
      if (os_sensor_param.getType() != XmlRpc::XmlRpcValue::TypeStruct) {
        ROS_ERROR_STREAM("ERROR: 'body/os_sensor' is not a struct!");
        return false;
      }
      
      if (!os_sensor_param.hasMember("T")) {
        ROS_ERROR_STREAM("ERROR: 'body/os_sensor/T' not found in calibration!");
        return false;
      }
      
      XmlRpc::XmlRpcValue T_param = os_sensor_param["T"];
      if (T_param.getType() != XmlRpc::XmlRpcValue::TypeArray || T_param.size() != 4) {
        ROS_ERROR_STREAM("ERROR: 'body/os_sensor/T' must be an array of 4 arrays!");
        return false;
      }
      
      // Parse the 4x4 matrix
      for (int i = 0; i < 4; ++i) {
        if (T_param[i].getType() != XmlRpc::XmlRpcValue::TypeArray || T_param[i].size() != 4) {
          ROS_ERROR_STREAM("ERROR: Row " << i << " of body/os_sensor/T must be an array of 4 elements!");
          return false;
        }
        for (int j = 0; j < 4; ++j) {
          // XmlRpcValue might store as double or int, convert to double
          if (T_param[i][j].getType() == XmlRpc::XmlRpcValue::TypeDouble) {
            body_to_lidar_tf_(i, j) = static_cast<double>(T_param[i][j]);
          } else if (T_param[i][j].getType() == XmlRpc::XmlRpcValue::TypeInt) {
            body_to_lidar_tf_(i, j) = static_cast<double>(static_cast<int>(T_param[i][j]));
          } else {
            ROS_ERROR_STREAM("ERROR: body/os_sensor/T[" << i << "][" << j << "] is not a number!");
            return false;
          }
        }
      }
      
      ROS_INFO_STREAM("Successfully loaded body-to-lidar transform from body/os_sensor/T");
      ROS_INFO_STREAM("Transform matrix:");
      ROS_INFO_STREAM("  [" << body_to_lidar_tf_(0, 0) << ", " << body_to_lidar_tf_(0, 1) << ", " << body_to_lidar_tf_(0, 2) << ", " << body_to_lidar_tf_(0, 3) << "]");
      ROS_INFO_STREAM("  [" << body_to_lidar_tf_(1, 0) << ", " << body_to_lidar_tf_(1, 1) << ", " << body_to_lidar_tf_(1, 2) << ", " << body_to_lidar_tf_(1, 3) << "]");
      ROS_INFO_STREAM("  [" << body_to_lidar_tf_(2, 0) << ", " << body_to_lidar_tf_(2, 1) << ", " << body_to_lidar_tf_(2, 2) << ", " << body_to_lidar_tf_(2, 3) << "]");
      ROS_INFO_STREAM("  [" << body_to_lidar_tf_(3, 0) << ", " << body_to_lidar_tf_(3, 1) << ", " << body_to_lidar_tf_(3, 2) << ", " << body_to_lidar_tf_(3, 3) << "]");
      
      return true;
    }

    void query_scan(std::string input_data_dir, std::string input_label_dir, int pose_idx) {
      if (pose_idx < 0 || pose_idx >= (int)lidar_poses_.size()) {
        return;
      }

      // Get the actual scan file number from CSV
      int scan_file_num = scan_indices_[pose_idx];
      
      // Use 10-digit format for MCD file naming
      char scan_id_c[256];
      sprintf(scan_id_c, "%010d", scan_file_num);
      std::string scan_name = input_data_dir + "/" + std::string(scan_id_c) + ".bin";
      std::string gt_name = gt_label_dir_ + "/" + std::string(scan_id_c) + ".bin";
      std::string result_name = evaluation_result_dir_ + "/" + std::string(scan_id_c) + ".txt";
      
      pcl::PointCloud<pcl::PointXYZL>::Ptr cloud = mcd2pcl(scan_name, gt_name);
      if (cloud->points.empty()) {
        return;
      }

      Eigen::Matrix4d transform = lidar_poses_[pose_idx];  // This is body_to_world from pose
      
      // Apply body-to-lidar transformation (same as in process_scans)
      // Following Python code: transform_matrix = body_to_world @ lidar_to_body
      Eigen::Matrix4d lidar_to_body = body_to_lidar_tf_.inverse();
      Eigen::Matrix4d new_transform = transform * lidar_to_body;  // body_to_world * lidar_to_body
      pcl::transformPointCloud(*cloud, *cloud, new_transform);

      std::ofstream result_file;
      result_file.open(result_name);
      for (int i = 0; i < (int)cloud->points.size(); ++i) {
        semantic_bki::SemanticOcTreeNode node = map_->search(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
        int pred_label = 0;
        if (node.get_state() == semantic_bki::State::OCCUPIED)
          pred_label = node.get_semantics();
        result_file << (int)cloud->points[i].label << " " << pred_label << "\n";
      }
      result_file.close();
    }

  
  private:
    ros::NodeHandle nh_;
    double resolution_;
    double ds_resolution_;
    double free_resolution_;
    double max_range_;
    semantic_bki::SemanticBKIOctoMap* map_;
    semantic_bki::MarkerArrayPub* m_pub_;
    ros::Publisher color_octomap_publisher_;
    ros::Publisher pointcloud_pub_;  // Publisher for individual scan point clouds
    tf::TransformListener listener_;
    tf::TransformBroadcaster tf_broadcaster_;
    std::ofstream pose_file_;
    std::vector<Eigen::Matrix4d> lidar_poses_;
    std::vector<int> scan_indices_;  // Maps pose index to actual scan file number (from CSV "num" column)
    std::string gt_label_dir_;
    std::string evaluation_result_dir_;
    Eigen::Matrix4d init_trans_to_ground_;
    Eigen::Matrix4d body_to_lidar_tf_;  // Body to LiDAR transformation from calibration
    Eigen::Matrix4d original_first_pose_;  // Original first pose before transformation to origin (for OSM alignment)

    pcl::PointCloud<pcl::PointXYZL>::Ptr mcd2pcl(std::string fn, std::string fn_label) {
      // Open scan file
      FILE* fp = std::fopen(fn.c_str(), "rb");
      if (!fp) {
        ROS_WARN_STREAM("Cannot open scan file: " << fn);
        return pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);
      }

      // Open label file
      FILE* fp_label = std::fopen(fn_label.c_str(), "rb");
      if (!fp_label) {
        ROS_WARN_STREAM("Cannot open label file: " << fn_label);
        std::fclose(fp);
        return pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);
      }

      // Get file size for scan (x, y, z, intensity = 4 floats per point)
      std::fseek(fp, 0L, SEEK_END);
      size_t sz = std::ftell(fp);
      std::rewind(fp);
      int n_hits = sz / (sizeof(float) * 4);

      // Preallocate point cloud for better performance (avoids reallocation)
      pcl::PointCloud<pcl::PointXYZL>::Ptr pc(new pcl::PointCloud<pcl::PointXYZL>);
      pc->points.reserve(n_hits);  // Preallocate to avoid reallocation overhead
      pc->width = n_hits;
      pc->height = 1;
      pc->is_dense = false;

      // Read data in a tighter loop
      for (int i = 0; i < n_hits; i++) {
        pcl::PointXYZL point;
        float intensity;
        uint32_t label;

        // Read point data (x, y, z, intensity as floats)
        if (fread(&point.x, sizeof(float), 1, fp) != 1) break;
        if (fread(&point.y, sizeof(float), 1, fp) != 1) break;
        if (fread(&point.z, sizeof(float), 1, fp) != 1) break;
        if (fread(&intensity, sizeof(float), 1, fp) != 1) break;

        // Read label (uint32)
        if (fread(&label, sizeof(uint32_t), 1, fp_label) != 1) break;

        point.label = (int)label;
        pc->points.push_back(point);
      }
      
      std::fclose(fp);
      std::fclose(fp_label);
      
      return pc;
    }
};
