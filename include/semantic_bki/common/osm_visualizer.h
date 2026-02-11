#pragma once

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>
#include <osmium/io/any_input.hpp>
#include <osmium/handler.hpp>
#include <osmium/visitor.hpp>
#include <osmium/osm/way.hpp>
#include <osmium/osm/node.hpp>
#include <osmium/index/map/sparse_mem_array.hpp>
#include <osmium/handler/node_locations_for_ways.hpp>
#include <osmium/osm/location.hpp>

namespace semantic_bki {

    struct Geometry2D {
        std::vector<std::pair<float, float>> coords;
    };

    class OSMVisualizer {
    public:
        OSMVisualizer(rclcpp::Node::SharedPtr node, const std::string& topic);
        ~OSMVisualizer() = default;

        /**
         * Load OSM geometries from .osm XML file.
         * Extracts buildings, roads, and sidewalks using libosmium for proper parsing.
         * @param osm_file Path to .osm XML file
         * @param origin_lat Latitude of local coordinate origin (degrees)
         * @param origin_lon Longitude of local coordinate origin (degrees)
         * @return true if loaded successfully
         */
        bool loadFromOSM(const std::string& osm_file, double origin_lat, double origin_lon);

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

        /**
         * Set the lidar trajectory path (sequence of x,y points in map frame) for debugging.
         * Drawn as a polyline. Call after transformToFirstPoseOrigin if using same frame.
         */
        void setPath(const std::vector<std::pair<float, float>>& path);

        /**
         * Save OSM buildings and roads visualization as a PNG image.
         * @param output_path Path to save the PNG file
         * @param image_width Width of the output image in pixels (default: 2048)
         * @param image_height Height of the output image in pixels (default: 2048)
         * @param margin_pixels Margin around the geometries in pixels (default: 50)
         * @return true if saved successfully
         */
        bool saveAsPNG(const std::string& output_path, int image_width = 2048, int image_height = 2048, int margin_pixels = 50);

    private:
        /**
         * Timer callback for periodic publishing.
         */
        void timerCallback();

        /**
         * Create Marker message for buildings (line outlines).
         */
        visualization_msgs::msg::Marker createBuildingMarker(const std::vector<Geometry2D>& buildings);

        /**
         * Create Marker message for roads and sidewalks (red polylines).
         */
        visualization_msgs::msg::Marker createRoadMarker(const std::vector<Geometry2D>& roads);

        /**
         * Create Marker message for lidar path (green polyline).
         */
        visualization_msgs::msg::Marker createPathMarker() const;

        /**
         * Create Marker message for grasslands (greenish polygon outlines).
         */
        visualization_msgs::msg::Marker createGrasslandMarker(const std::vector<Geometry2D>& grasslands);

        /**
         * Create Marker message for trees/forest (dark green polygon outlines).
         */
        visualization_msgs::msg::Marker createTreeMarker(const std::vector<Geometry2D>& trees);

        /**
         * Create Marker message for tree points (single-node trees as spheres).
         */
        visualization_msgs::msg::Marker createTreePointsMarker() const;

        rclcpp::Node::SharedPtr node_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;
        rclcpp::TimerBase::SharedPtr publish_timer_;
        std::string topic_;
        std::string frame_id_;

        std::vector<Geometry2D> buildings_;
        std::vector<Geometry2D> roads_;
        std::vector<Geometry2D> grasslands_;
        std::vector<Geometry2D> trees_;           // Forest/wood polygons
        std::vector<std::pair<float, float>> tree_points_;  // Single-point trees (natural=tree nodes)
        std::vector<std::pair<float, float>> path_;  // Lidar trajectory for debugging

        bool transformed_; // Flag to track if data has already been transformed
    };

} // namespace semantic_bki
