#include "osm_visualizer.h"
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <memory>
#include <Eigen/Dense>

namespace semantic_bki {

    OSMVisualizer::OSMVisualizer(rclcpp::Node::SharedPtr node, const std::string& topic) 
        : node_(node), topic_(topic), frame_id_("map") {
        if (!node_) {
            RCLCPP_ERROR_STREAM(rclcpp::get_logger("osm_visualizer"), "ERROR: OSMVisualizer constructor: node_ is null!");
            return;
        }
        RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: OSMVisualizer constructor: Creating publisher for topic: " << topic_);
        // Use default QoS for ROS2 (compatible with ros2 topic hz and RViz)
        // transient_local requires matching QoS on subscriber side, which ros2 topic hz doesn't use
        // Use default reliable QoS instead for better compatibility
        pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
            topic_, 
            rclcpp::QoS(10).reliable());
        if (!pub_) {
            RCLCPP_ERROR_STREAM(node_->get_logger(), "ERROR: Failed to create OSM publisher!");
        } else {
            RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: OSM publisher created successfully");
            RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Publisher topic name: " << pub_->get_topic_name());
            RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Publisher subscription count: " << pub_->get_subscription_count());
            RCLCPP_INFO_STREAM(node_->get_logger(), "OSMVisualizer: Publisher created for topic: " << topic_ << " with reliable QoS");
        }
    }

    bool OSMVisualizer::loadFromBinary(const std::string& bin_file) {
        std::ifstream file(bin_file, std::ios::binary);
        if (!file.is_open()) {
            RCLCPP_ERROR_STREAM(node_->get_logger(), "Failed to open binary file: " << bin_file);
            return false;
        }
        
        // Get file size for debugging
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        RCLCPP_INFO_STREAM(node_->get_logger(), "OSM binary file size: " << file_size << " bytes");
        
        buildings_.clear();
        roads_.clear();
        grasslands_.clear();
        trees_.clear();
        wood_.clear();
        
        // Binary format (matches create_map_OSM_BEV_GEOM.py):
        // For each geometry type (buildings, roads, grasslands, trees, wood):
        //   uint32_t num_geometries (4 bytes, native endian)
        //   For each geometry:
        //     uint32_t num_points (4 bytes)
        //     num_points * (float x, float y) (8 bytes per point: 4 bytes x, 4 bytes y)
        // Format: struct.pack('I', count) then struct.pack('I', num_points) then struct.pack('ff', x, y) for each point
        
        // Helper lambda to read geometry data
        auto read_geometries = [&](std::vector<Geometry2D>& geometries, const std::string& name) -> bool {
            // Check file state before reading
            if (!file.good()) {
                RCLCPP_ERROR_STREAM(node_->get_logger(), "File not in good state before reading " << name);
                return false;
            }
            
            size_t pos_before = file.tellg();
            uint32_t num_geometries;
            if (file.read(reinterpret_cast<char*>(&num_geometries), sizeof(uint32_t)).gcount() != sizeof(uint32_t)) {
                RCLCPP_ERROR_STREAM(node_->get_logger(), "Failed to read number of " << name << " from binary file (at position " << pos_before << ", file size " << file_size << ")");
                return false;
            }
            
            RCLCPP_DEBUG_STREAM(node_->get_logger(), "Reading " << num_geometries << " " << name << " geometries");
            
            for (uint32_t i = 0; i < num_geometries; ++i) {
                Geometry2D geom;
                uint32_t num_points;
                if (file.read(reinterpret_cast<char*>(&num_points), sizeof(uint32_t)).gcount() != sizeof(uint32_t)) {
                    RCLCPP_ERROR_STREAM(node_->get_logger(), "Failed to read point count for " << name << " " << i);
                    return false;
                }
                
                for (uint32_t j = 0; j < num_points; ++j) {
                    float x, y;
                    if (file.read(reinterpret_cast<char*>(&x), sizeof(float)).gcount() != sizeof(float) ||
                        file.read(reinterpret_cast<char*>(&y), sizeof(float)).gcount() != sizeof(float)) {
                        RCLCPP_ERROR_STREAM(node_->get_logger(), "Failed to read coordinates for " << name << " " << i << ", point " << j);
                        return false;
                    }
                    geom.coords.push_back(std::make_pair(x, y));
                }
                geometries.push_back(geom);
            }
            
            size_t pos_after = file.tellg();
            RCLCPP_DEBUG_STREAM(node_->get_logger(), "Finished reading " << name << ", file position now at " << pos_after);
            return true;
        };

        if (!read_geometries(buildings_, "buildings")) { 
            RCLCPP_ERROR_STREAM(node_->get_logger(), "Failed to read buildings"); 
            file.close(); 
            return false; 
        }
        if (!read_geometries(roads_, "roads")) { 
            RCLCPP_ERROR_STREAM(node_->get_logger(), "Failed to read roads"); 
            file.close(); 
            return false; 
        }
        if (!read_geometries(grasslands_, "grasslands")) { 
            RCLCPP_ERROR_STREAM(node_->get_logger(), "Failed to read grasslands"); 
            file.close(); 
            return false; 
        }
        
        // Trees and wood are optional (for backward compatibility with old binary files)
        // Check if we've reached end of file before trying to read them
        size_t current_pos = file.tellg();
        if (current_pos < file_size && (file_size - current_pos) >= sizeof(uint32_t)) {
            // Save position before attempting to read trees
            auto trees_start_pos = file.tellg();
            // Try to read trees (may fail if file is in old format without trees/wood data)
            if (!read_geometries(trees_, "trees")) { 
                RCLCPP_WARN_STREAM(node_->get_logger(), "Failed to read trees (file may be in old format without trees/wood data)"); 
                // Reset file state and rewind to start of trees block
                file.clear();
                file.seekg(trees_start_pos);
                trees_.clear();
            }
        } else {
            RCLCPP_INFO_STREAM(node_->get_logger(), "Reached end of file after grasslands. Binary file is in old format (no trees/wood data).");
        }
        
        // Try to read wood if there's still data
        current_pos = file.tellg();
        if (current_pos < file_size && (file_size - current_pos) >= sizeof(uint32_t)) {
            // Save position before attempting to read wood
            auto wood_start_pos = file.tellg();
            if (!read_geometries(wood_, "wood")) { 
                RCLCPP_WARN_STREAM(node_->get_logger(), "Failed to read wood (file may be in old format without wood data)"); 
                // Reset file state and rewind to start of wood block
                file.clear();
                file.seekg(wood_start_pos);
                wood_.clear();
            }
        }
        
        file.close();
        
        RCLCPP_INFO_STREAM(node_->get_logger(), "Loaded OSM geometries from binary file: " << buildings_.size() << " buildings, " 
                      << roads_.size() << " roads, " << grasslands_.size() << " grasslands, "
                      << trees_.size() << " trees, " << wood_.size() << " wood");
        
        // Count total points for debugging
        size_t total_points = 0;
        for (const auto& b : buildings_) total_points += b.coords.size();
        for (const auto& r : roads_) total_points += r.coords.size();
        for (const auto& g : grasslands_) total_points += g.coords.size();
        for (const auto& t : trees_) total_points += t.coords.size();
        for (const auto& w : wood_) total_points += w.coords.size();
        RCLCPP_INFO_STREAM(node_->get_logger(), "Total OSM points loaded: " << total_points);
        RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: OSM data loaded successfully - buildings=" << buildings_.size() << ", roads=" << roads_.size() << ", grasslands=" << grasslands_.size() << ", trees=" << trees_.size() << ", wood=" << wood_.size());
        
        return true;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createBuildingMarker(const std::vector<Geometry2D>& buildings) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_buildings";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.5; // Line width (increased for visibility)
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0;

        for (const auto& building : buildings) {
            if (building.coords.size() < 2) continue;
            
            // Check for NaN or invalid coordinates and skip if found
            bool has_invalid = false;
            for (const auto& coord : building.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) || 
                    std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            if (has_invalid) {
                RCLCPP_WARN_STREAM(node_->get_logger(), "Skipping building polygon with invalid (NaN/Inf) coordinates");
                continue;
            }
            
            // Create closed polygon by connecting points
            for (size_t i = 0; i < building.coords.size(); ++i) {
                geometry_msgs::msg::Point p1, p2;
                p1.x = building.coords[i].first;
                p1.y = building.coords[i].second;
                p1.z = 0.0;
                
                size_t next_idx = (i + 1) % building.coords.size();
                p2.x = building.coords[next_idx].first;
                p2.y = building.coords[next_idx].second;
                p2.z = 0.0;
                
                marker.points.push_back(p1);
                marker.points.push_back(p2);
            }
        }

        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createRoadMarker(const std::vector<Geometry2D>& roads) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = rclcpp::Time(0);
        marker.ns = "osm_roads";
        marker.id = 1;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.6; // Line width (increased for visibility)
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        for (const auto& road : roads) {
            if (road.coords.size() < 2) continue;
            
            // Check for NaN or invalid coordinates and skip if found
            bool has_invalid = false;
            for (const auto& coord : road.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) || 
                    std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            if (has_invalid) {
                RCLCPP_WARN_STREAM(node_->get_logger(), "Skipping road linestring with invalid (NaN/Inf) coordinates");
                continue;
            }
            
            // Connect consecutive points within each road segment using LINE_LIST
            for (size_t i = 0; i < road.coords.size() - 1; ++i) {
                geometry_msgs::msg::Point p1, p2;
                p1.x = road.coords[i].first;
                p1.y = road.coords[i].second;
                p1.z = 0.0;
                
                p2.x = road.coords[i + 1].first;
                p2.y = road.coords[i + 1].second;
                p2.z = 0.0;
                
                marker.points.push_back(p1);
                marker.points.push_back(p2);
            }
        }

        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createGrasslandMarker(const std::vector<Geometry2D>& grasslands) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = rclcpp::Time(0);
        marker.ns = "osm_grasslands";
        marker.id = 2;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.5; // Line width (increased for visibility)
        marker.color.r = 0.5;
        marker.color.g = 0.8;
        marker.color.b = 0.3;
        marker.color.a = 1.0; // Fully opaque for better visibility

        for (const auto& grassland : grasslands) {
            if (grassland.coords.size() < 3) continue;
            
            // Check for NaN or invalid coordinates and skip if found
            bool has_invalid = false;
            for (const auto& coord : grassland.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) || 
                    std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            if (has_invalid) {
                RCLCPP_WARN_STREAM(node_->get_logger(), "Skipping grassland polygon with invalid (NaN/Inf) coordinates");
                continue;
            }
            
            // Create closed polygon by connecting consecutive points
            for (size_t i = 0; i < grassland.coords.size(); ++i) {
                geometry_msgs::msg::Point p1, p2;
                p1.x = grassland.coords[i].first;
                p1.y = grassland.coords[i].second;
                p1.z = 0.0;
                
                size_t next_idx = (i + 1) % grassland.coords.size();
                p2.x = grassland.coords[next_idx].first;
                p2.y = grassland.coords[next_idx].second;
                p2.z = 0.0;
                
                marker.points.push_back(p1);
                marker.points.push_back(p2);
            }
        }
        
        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createTreeMarker(const std::vector<Geometry2D>& trees) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = rclcpp::Time(0);
        marker.ns = "osm_trees";
        marker.id = 3;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.5; // Line width (increased for visibility)
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0; // Fully opaque for better visibility

        for (const auto& tree : trees) {
            if (tree.coords.empty()) continue;
            
            // Check for NaN or invalid coordinates
            bool has_invalid = false;
            for (const auto& coord : tree.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) || 
                    std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            if (has_invalid) {
                RCLCPP_WARN_STREAM(node_->get_logger(), "Skipping tree geometry with invalid (NaN/Inf) coordinates");
                continue;
            }
            
            if (tree.coords.size() == 1) {
                // Single point - create a small circle by adding 4 points around it
                // This makes it visible as a small square/circle
                float x = tree.coords[0].first;
                float y = tree.coords[0].second;
                float radius = 0.5; // Small radius for visibility
                
                // Create a small square around the point
                geometry_msgs::msg::Point p1, p2, p3, p4;
                p1.x = x - radius; p1.y = y - radius; p1.z = 0.0;
                p2.x = x + radius; p2.y = y - radius; p2.z = 0.0;
                p3.x = x + radius; p3.y = y + radius; p3.z = 0.0;
                p4.x = x - radius; p4.y = y + radius; p4.z = 0.0;
                
                // Connect points to form a square
                marker.points.push_back(p1); marker.points.push_back(p2);
                marker.points.push_back(p2); marker.points.push_back(p3);
                marker.points.push_back(p3); marker.points.push_back(p4);
                marker.points.push_back(p4); marker.points.push_back(p1);
            } else if (tree.coords.size() >= 2) {
                // LineString or Polygon
                for (size_t i = 0; i < tree.coords.size() - 1; ++i) {
                    geometry_msgs::msg::Point p1, p2;
                    p1.x = tree.coords[i].first;
                    p1.y = tree.coords[i].second;
                    p1.z = 0.0;
                    
                    p2.x = tree.coords[i + 1].first;
                    p2.y = tree.coords[i + 1].second;
                    p2.z = 0.0;
                    
                    marker.points.push_back(p1);
                    marker.points.push_back(p2);
                }
                // Close polygon if first and last points are the same
                if (tree.coords.size() > 2 && 
                    tree.coords[0].first == tree.coords.back().first &&
                    tree.coords[0].second == tree.coords.back().second) {
                    // Already closed
                }
            }
        }

        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createWoodMarker(const std::vector<Geometry2D>& wood) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = rclcpp::Time(0);
        marker.ns = "osm_wood";
        marker.id = 4;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.5; // Line width (increased for visibility)
        marker.color.r = 0.0;
        marker.color.g = 0.4;
        marker.color.b = 0.0;
        marker.color.a = 1.0; // Fully opaque for better visibility

        for (const auto& wood_area : wood) {
            if (wood_area.coords.size() < 3) continue;
            
            // Check for NaN or invalid coordinates
            bool has_invalid = false;
            for (const auto& coord : wood_area.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) || 
                    std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            if (has_invalid) {
                RCLCPP_WARN_STREAM(node_->get_logger(), "Skipping wood polygon with invalid (NaN/Inf) coordinates");
                continue;
            }
            
            // Create closed polygon
            for (size_t i = 0; i < wood_area.coords.size(); ++i) {
                geometry_msgs::msg::Point p1, p2;
                p1.x = wood_area.coords[i].first;
                p1.y = wood_area.coords[i].second;
                p1.z = 0.0;
                
                size_t next_idx = (i + 1) % wood_area.coords.size();
                p2.x = wood_area.coords[next_idx].first;
                p2.y = wood_area.coords[next_idx].second;
                p2.z = 0.0;
                
                marker.points.push_back(p1);
                marker.points.push_back(p2);
            }
        }

        return marker;
    }

    void OSMVisualizer::publish() {
        RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: publish() called");
        
        if (!pub_) {
            RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Cannot publish - publisher is null!");
            return;
        }
        
        if (!node_) {
            RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Cannot publish - node is null!");
            return;
        }
        
        RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Publisher and node are valid, creating marker array");
        visualization_msgs::msg::MarkerArray marker_array;
        
        int total_points = 0;
        if (!buildings_.empty()) {
            auto marker = createBuildingMarker(buildings_);
            total_points += marker.points.size();
            marker_array.markers.push_back(marker);
            RCLCPP_INFO_STREAM(node_->get_logger(), "OSM: Added " << buildings_.size() << " buildings with " << marker.points.size() << " points");
        }
        if (!roads_.empty()) {
            auto marker = createRoadMarker(roads_);
            total_points += marker.points.size();
            marker_array.markers.push_back(marker);
            RCLCPP_INFO_STREAM(node_->get_logger(), "OSM: Added " << roads_.size() << " roads with " << marker.points.size() << " points");
        }
        if (!grasslands_.empty()) {
            auto marker = createGrasslandMarker(grasslands_);
            total_points += marker.points.size();
            marker_array.markers.push_back(marker);
            RCLCPP_INFO_STREAM(node_->get_logger(), "OSM: Added " << grasslands_.size() << " grasslands with " << marker.points.size() << " points");
        }
        if (!trees_.empty()) {
            auto marker = createTreeMarker(trees_);
            total_points += marker.points.size();
            marker_array.markers.push_back(marker);
            RCLCPP_INFO_STREAM(node_->get_logger(), "OSM: Added " << trees_.size() << " trees with " << marker.points.size() << " points");
        }
        if (!wood_.empty()) {
            auto marker = createWoodMarker(wood_);
            total_points += marker.points.size();
            marker_array.markers.push_back(marker);
            RCLCPP_INFO_STREAM(node_->get_logger(), "OSM: Added " << wood_.size() << " wood areas with " << marker.points.size() << " points");
        }
        
        if (marker_array.markers.empty()) {
            RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: No markers to publish! (buildings=" << buildings_.size() << ", roads=" << roads_.size() << ", grasslands=" << grasslands_.size() << ", trees=" << trees_.size() << ", wood=" << wood_.size() << ")");
            return;
        }
        
        // Set lifetime for all markers - use zero duration (never expire) like ROS1
        // In ROS2, zero duration means never expire
        rclcpp::Duration zero_lifetime(0, 0); // Never expire (same as ROS1 ros::Duration())
        for (auto& marker : marker_array.markers) {
            marker.lifetime = zero_lifetime;
            // Ensure frame_id is set correctly
            marker.header.frame_id = frame_id_;
            marker.header.stamp = rclcpp::Time(0);
        }
        
        RCLCPP_INFO_STREAM(node_->get_logger(), "OSMVisualizer: Publishing " << marker_array.markers.size() << " markers with " << total_points << " total points to topic " << topic_ << " in frame " << frame_id_);
        
        // Log first marker details for debugging
        if (!marker_array.markers.empty()) {
            const auto& first_marker = marker_array.markers[0];
            RCLCPP_INFO_STREAM(node_->get_logger(), "First marker: ns=" << first_marker.ns << ", id=" << first_marker.id << ", type=" << first_marker.type << ", points=" << first_marker.points.size() << ", frame=" << first_marker.header.frame_id);
            if (!first_marker.points.empty()) {
                RCLCPP_INFO_STREAM(node_->get_logger(), "First point: [" << first_marker.points[0].x << ", " << first_marker.points[0].y << ", " << first_marker.points[0].z << "]");
            }
        }
        
        // TEST: Add a simple cube marker to verify publisher works
        visualization_msgs::msg::Marker test_marker;
        test_marker.header.frame_id = "map";
        test_marker.header.stamp = rclcpp::Time(0);
        test_marker.ns = "test";
        test_marker.id = 999;
        test_marker.type = visualization_msgs::msg::Marker::CUBE;
        test_marker.action = visualization_msgs::msg::Marker::ADD;
        test_marker.pose.position.x = 0.0;
        test_marker.pose.position.y = 0.0;
        test_marker.pose.position.z = 0.0;
        test_marker.pose.orientation.w = 1.0;
        test_marker.scale.x = 5.0;
        test_marker.scale.y = 5.0;
        test_marker.scale.z = 5.0;
        test_marker.color.r = 1.0;
        test_marker.color.g = 0.0;
        test_marker.color.b = 0.0;
        test_marker.color.a = 1.0;
        test_marker.lifetime = rclcpp::Duration(0, 0);
        marker_array.markers.push_back(test_marker);
        RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Added test cube marker (id=999) at origin");
        
        RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: About to publish MarkerArray with " << marker_array.markers.size() << " markers");
        RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Publisher valid: " << (pub_ ? "YES" : "NO"));
        RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Topic: " << topic_);
        
        try {
            pub_->publish(marker_array);
            RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Successfully published MarkerArray to " << topic_ << " with " << marker_array.markers.size() << " markers");
        } catch (const std::exception& e) {
            RCLCPP_ERROR_STREAM(node_->get_logger(), "OSMVisualizer: Exception while publishing: " << e.what());
        }
    }

    void OSMVisualizer::startPeriodicPublishing(double rate) {
        if (rate <= 0.0) {
            RCLCPP_WARN(node_->get_logger(), "OSMVisualizer: Invalid publishing rate, using default 1.0 Hz");
            rate = 1.0;
        }
        RCLCPP_INFO_STREAM(node_->get_logger(), "OSMVisualizer: Creating timer for periodic publishing at " << rate << " Hz");
        publish_timer_ = node_->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / rate)),
            std::bind(&OSMVisualizer::timerCallback, this));
        if (!publish_timer_) {
            RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Failed to create publishing timer!");
        } else {
            RCLCPP_INFO_STREAM(node_->get_logger(), "OSMVisualizer: Started periodic publishing at " << rate << " Hz");
        }
    }

    void OSMVisualizer::timerCallback() {
        RCLCPP_INFO_STREAM(node_->get_logger(), "OSMVisualizer: Timer callback triggered, republishing markers");
        publish();
    }

    void OSMVisualizer::transformToFirstPoseOrigin(const Eigen::Matrix4d& first_pose) {
        Eigen::Matrix4d first_pose_inverse = first_pose.inverse();
        
        RCLCPP_INFO_STREAM(node_->get_logger(), "Transforming OSM geometries to be relative to first pose...");
        RCLCPP_INFO_STREAM(node_->get_logger(), "First pose translation: [" << first_pose(0,3) << ", " << first_pose(1,3) << ", " << first_pose(2,3) << "]");
        RCLCPP_INFO_STREAM(node_->get_logger(), "Applying full pose transformation (rotation + translation) to match scans/map");
        
        auto transformPoint = [&first_pose_inverse](float& x, float& y) {
            Eigen::Vector4d point(x, y, 0.0, 1.0);
            Eigen::Vector4d transformed = first_pose_inverse * point;
            x = static_cast<float>(transformed(0));
            y = static_cast<float>(transformed(1));
        };
        
        for (auto& building : buildings_) {
            for (auto& coord : building.coords) {
                transformPoint(coord.first, coord.second);
            }
        }
        for (auto& road : roads_) {
            for (auto& coord : road.coords) {
                transformPoint(coord.first, coord.second);
            }
        }
        for (auto& grassland : grasslands_) {
            for (auto& coord : grassland.coords) {
                transformPoint(coord.first, coord.second);
            }
        }
        for (auto& tree : trees_) {
            for (auto& coord : tree.coords) {
                transformPoint(coord.first, coord.second);
            }
        }
        for (auto& wood_area : wood_) {
            for (auto& coord : wood_area.coords) {
                transformPoint(coord.first, coord.second);
            }
        }
        RCLCPP_INFO_STREAM(node_->get_logger(), "OSM geometries transformed to first pose origin frame.");
    }

} // namespace semantic_bki
