#include "osm_visualizer.h"
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <ros/package.h>
#include <memory>
#include <Eigen/Dense>

namespace semantic_bki {

    OSMVisualizer::OSMVisualizer(ros::NodeHandle nh, const std::string& topic) 
        : nh_(nh), topic_(topic), frame_id_("map") {
        pub_ = nh_.advertise<visualization_msgs::MarkerArray>(topic_, 1, true);
        ROS_INFO_STREAM("OSMVisualizer: Advertising topic: " << topic_);
    }

    bool OSMVisualizer::loadFromBinary(const std::string& bin_file) {
        std::ifstream file(bin_file, std::ios::binary);
        if (!file.is_open()) {
            ROS_ERROR_STREAM("Failed to open binary file: " << bin_file);
            return false;
        }
        
        buildings_.clear();
        roads_.clear();
        grasslands_.clear();
        
        // Read buildings
        uint32_t num_buildings;
        if (file.read(reinterpret_cast<char*>(&num_buildings), sizeof(uint32_t)).gcount() != sizeof(uint32_t)) {
            ROS_ERROR_STREAM("Failed to read number of buildings from binary file");
            file.close();
            return false;
        }
        
        for (uint32_t i = 0; i < num_buildings; ++i) {
            Geometry2D geom;
            uint32_t num_points;
            if (file.read(reinterpret_cast<char*>(&num_points), sizeof(uint32_t)).gcount() != sizeof(uint32_t)) {
                ROS_ERROR_STREAM("Failed to read point count for building " << i);
                file.close();
                return false;
            }
            
            for (uint32_t j = 0; j < num_points; ++j) {
                float x, y;
                if (file.read(reinterpret_cast<char*>(&x), sizeof(float)).gcount() != sizeof(float) ||
                    file.read(reinterpret_cast<char*>(&y), sizeof(float)).gcount() != sizeof(float)) {
                    ROS_ERROR_STREAM("Failed to read coordinates for building " << i << ", point " << j);
                    file.close();
                    return false;
                }
                geom.coords.push_back(std::make_pair(x, y));
            }
            buildings_.push_back(geom);
        }
        
        // Read roads
        uint32_t num_roads;
        if (file.read(reinterpret_cast<char*>(&num_roads), sizeof(uint32_t)).gcount() != sizeof(uint32_t)) {
            ROS_ERROR_STREAM("Failed to read number of roads from binary file");
            file.close();
            return false;
        }
        
        for (uint32_t i = 0; i < num_roads; ++i) {
            Geometry2D geom;
            uint32_t num_points;
            if (file.read(reinterpret_cast<char*>(&num_points), sizeof(uint32_t)).gcount() != sizeof(uint32_t)) {
                ROS_ERROR_STREAM("Failed to read point count for road " << i);
                file.close();
                return false;
            }
            
            for (uint32_t j = 0; j < num_points; ++j) {
                float x, y;
                if (file.read(reinterpret_cast<char*>(&x), sizeof(float)).gcount() != sizeof(float) ||
                    file.read(reinterpret_cast<char*>(&y), sizeof(float)).gcount() != sizeof(float)) {
                    ROS_ERROR_STREAM("Failed to read coordinates for road " << i << ", point " << j);
                    file.close();
                    return false;
                }
                geom.coords.push_back(std::make_pair(x, y));
            }
            roads_.push_back(geom);
        }
        
        // Read grasslands
        uint32_t num_grasslands;
        if (file.read(reinterpret_cast<char*>(&num_grasslands), sizeof(uint32_t)).gcount() != sizeof(uint32_t)) {
            ROS_ERROR_STREAM("Failed to read number of grasslands from binary file");
            file.close();
            return false;
        }
        
        for (uint32_t i = 0; i < num_grasslands; ++i) {
            Geometry2D geom;
            uint32_t num_points;
            if (file.read(reinterpret_cast<char*>(&num_points), sizeof(uint32_t)).gcount() != sizeof(uint32_t)) {
                ROS_ERROR_STREAM("Failed to read point count for grassland " << i);
                file.close();
                return false;
            }
            
            for (uint32_t j = 0; j < num_points; ++j) {
                float x, y;
                if (file.read(reinterpret_cast<char*>(&x), sizeof(float)).gcount() != sizeof(float) ||
                    file.read(reinterpret_cast<char*>(&y), sizeof(float)).gcount() != sizeof(float)) {
                    ROS_ERROR_STREAM("Failed to read coordinates for grassland " << i << ", point " << j);
                    file.close();
                    return false;
                }
                geom.coords.push_back(std::make_pair(x, y));
            }
            grasslands_.push_back(geom);
        }
        
        file.close();
        
        ROS_INFO_STREAM("Loaded OSM geometries from binary file: " << buildings_.size() << " buildings, " 
                       << roads_.size() << " roads, " << grasslands_.size() << " grasslands");
        
        return true;
    }

    visualization_msgs::Marker OSMVisualizer::createBuildingMarker(const std::vector<Geometry2D>& buildings) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = ros::Time::now();
        marker.ns = "osm_buildings";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.2; // Line width
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
                ROS_WARN_STREAM("Skipping building polygon with invalid (NaN/Inf) coordinates");
                continue;
            }
            
            // Create closed polygon by connecting points
            for (size_t i = 0; i < building.coords.size(); ++i) {
                geometry_msgs::Point p1, p2;
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

    visualization_msgs::Marker OSMVisualizer::createRoadMarker(const std::vector<Geometry2D>& roads) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = ros::Time::now();
        marker.ns = "osm_roads";
        marker.id = 1;
        marker.type = visualization_msgs::Marker::LINE_LIST;  // Use LINE_LIST instead of LINE_STRIP to avoid NaN issues
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.3; // Line width
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
                ROS_WARN_STREAM("Skipping road linestring with invalid (NaN/Inf) coordinates");
                continue;
            }
            
            // Connect consecutive points within each road segment using LINE_LIST
            // This naturally breaks between different roads without needing NaN
            for (size_t i = 0; i < road.coords.size() - 1; ++i) {
                geometry_msgs::Point p1, p2;
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

    visualization_msgs::Marker OSMVisualizer::createGrasslandMarker(const std::vector<Geometry2D>& grasslands) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = ros::Time::now();
        marker.ns = "osm_grasslands";
        marker.id = 2;
        marker.type = visualization_msgs::Marker::LINE_LIST;  // Use LINE_LIST for closed polygons
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.2; // Line width
        marker.color.r = 0.5;
        marker.color.g = 0.8;
        marker.color.b = 0.3;
        marker.color.a = 0.5; // Semi-transparent

        // Create closed polygons using LINE_LIST (similar to buildings)
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
                ROS_WARN_STREAM("Skipping grassland polygon with invalid (NaN/Inf) coordinates");
                continue;
            }
            
            // Create closed polygon by connecting consecutive points
            for (size_t i = 0; i < grassland.coords.size(); ++i) {
                geometry_msgs::Point p1, p2;
                p1.x = grassland.coords[i].first;
                p1.y = grassland.coords[i].second;
                p1.z = 0.0;
                
                // Connect to next point (wrap around to close polygon)
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

    void OSMVisualizer::publish() {
        visualization_msgs::MarkerArray marker_array;
        
        if (!buildings_.empty()) {
            marker_array.markers.push_back(createBuildingMarker(buildings_));
        }
        if (!roads_.empty()) {
            marker_array.markers.push_back(createRoadMarker(roads_));
        }
        if (!grasslands_.empty()) {
            marker_array.markers.push_back(createGrasslandMarker(grasslands_));
        }
        
        if (marker_array.markers.empty()) {
            ROS_WARN("OSMVisualizer: No markers to publish!");
            return;
        }
        
        // Set lifetime for all markers
        for (auto& marker : marker_array.markers) {
            marker.lifetime = ros::Duration(); // Never expire
        }
        
        pub_.publish(marker_array);
    }

    void OSMVisualizer::startPeriodicPublishing(double rate) {
        if (rate <= 0.0) {
            ROS_WARN("OSMVisualizer: Invalid publishing rate, using default 1.0 Hz");
            rate = 1.0;
        }
        publish_timer_ = nh_.createTimer(ros::Duration(1.0 / rate), &OSMVisualizer::timerCallback, this);
        ROS_INFO_STREAM("OSMVisualizer: Started periodic publishing at " << rate << " Hz");
    }

    void OSMVisualizer::timerCallback(const ros::TimerEvent& event) {
        publish();
    }

    void OSMVisualizer::transformToFirstPoseOrigin(const Eigen::Matrix4d& first_pose) {
        // Compute inverse of first pose (same transformation applied to scans/map)
        // This includes both rotation and translation to align OSM with the same coordinate frame as scans
        Eigen::Matrix4d first_pose_inverse = first_pose.inverse();
        
        ROS_INFO_STREAM("Transforming OSM geometries to be relative to first pose...");
        ROS_INFO_STREAM("First pose translation: [" << first_pose(0,3) << ", " << first_pose(1,3) << ", " << first_pose(2,3) << "]");
        ROS_INFO_STREAM("Applying full pose transformation (rotation + translation) to match scans/map");
        
        // Transform all coordinates: new_point = first_pose_inverse * [x, y, 0, 1]
        // For 2D points, we only need x and y components
        auto transformPoint = [&first_pose_inverse](float& x, float& y) {
            Eigen::Vector4d point(x, y, 0.0, 1.0);
            Eigen::Vector4d transformed = first_pose_inverse * point;
            x = static_cast<float>(transformed(0));
            y = static_cast<float>(transformed(1));
        };
        
        // Transform buildings
        for (auto& building : buildings_) {
            for (auto& coord : building.coords) {
                transformPoint(coord.first, coord.second);
            }
        }
        
        // Transform roads
        for (auto& road : roads_) {
            for (auto& coord : road.coords) {
                transformPoint(coord.first, coord.second);
            }
        }
        
        // Transform grasslands
        for (auto& grassland : grasslands_) {
            for (auto& coord : grassland.coords) {
                transformPoint(coord.first, coord.second);
            }
        }
        
        ROS_INFO_STREAM("OSM geometries transformed to first pose origin frame.");
    }

} // namespace semantic_bki
