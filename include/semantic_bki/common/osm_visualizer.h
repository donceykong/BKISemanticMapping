#pragma once

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>

namespace semantic_bki {

    struct Geometry2D {
        std::vector<std::pair<float, float>> coords;
    };

    class OSMVisualizer {
    public:
        OSMVisualizer(rclcpp::Node::SharedPtr node, const std::string& topic);
        ~OSMVisualizer() = default;

        /**
         * Load OSM geometries from simple binary format file (C++ native).
         * @param bin_file Path to binary .bin file
         * @return true if loaded successfully
         */
        bool loadFromBinary(const std::string& bin_file);

        /**
         * Publish OSM geometries as MarkerArray messages to RViz.
         */
        void publish();

        /**
         * Start periodic publishing at the specified rate.
         * @param rate Publishing rate in Hz (default: 1.0 Hz)
         */
        void startPeriodicPublishing(double rate = 1.0);

        /**
         * Transform all OSM coordinates by the inverse of the first pose.
         * This aligns OSM data with the same coordinate frame as scans/map (relative to first pose).
         * @param first_pose 4x4 transformation matrix of the original first pose (before alignment)
         */
        void transformToFirstPoseOrigin(const Eigen::Matrix4d& first_pose);

    private:
        /**
         * Timer callback for periodic publishing.
         */
        void timerCallback();

        /**
         * Create Marker message for buildings (blue lines).
         */
        visualization_msgs::msg::Marker createBuildingMarker(const std::vector<Geometry2D>& buildings);

        /**
         * Create Marker message for roads (red lines).
         */
        visualization_msgs::msg::Marker createRoadMarker(const std::vector<Geometry2D>& roads);

        /**
         * Create Marker message for grasslands (green filled polygons).
         */
        visualization_msgs::msg::Marker createGrasslandMarker(const std::vector<Geometry2D>& grasslands);

        /**
         * Create Marker message for trees (green markers/polygons).
         */
        visualization_msgs::msg::Marker createTreeMarker(const std::vector<Geometry2D>& trees);

        /**
         * Create Marker message for wood/forests (dark green filled polygons).
         */
        visualization_msgs::msg::Marker createWoodMarker(const std::vector<Geometry2D>& wood);

        rclcpp::Node::SharedPtr node_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;
        rclcpp::TimerBase::SharedPtr publish_timer_;
        std::string topic_;
        std::string frame_id_;

        std::vector<Geometry2D> buildings_;
        std::vector<Geometry2D> roads_;
        std::vector<Geometry2D> grasslands_;
        std::vector<Geometry2D> trees_;
        std::vector<Geometry2D> wood_;
    };

} // namespace semantic_bki
