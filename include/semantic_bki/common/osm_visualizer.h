#pragma once

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

namespace semantic_bki {

    struct Geometry2D {
        std::vector<std::pair<float, float>> coords;
    };

    class OSMVisualizer {
    public:
        OSMVisualizer(ros::NodeHandle nh, const std::string& topic);
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
        void timerCallback(const ros::TimerEvent& event);

        /**
         * Create Marker message for buildings (blue lines).
         */
        visualization_msgs::Marker createBuildingMarker(const std::vector<Geometry2D>& buildings);

        /**
         * Create Marker message for roads (red lines).
         */
        visualization_msgs::Marker createRoadMarker(const std::vector<Geometry2D>& roads);

        /**
         * Create Marker message for grasslands (green filled polygons).
         */
        visualization_msgs::Marker createGrasslandMarker(const std::vector<Geometry2D>& grasslands);

        ros::NodeHandle nh_;
        ros::Publisher pub_;
        ros::Timer publish_timer_;
        std::string topic_;
        std::string frame_id_;

        std::vector<Geometry2D> buildings_;
        std::vector<Geometry2D> roads_;
        std::vector<Geometry2D> grasslands_;
    };

} // namespace semantic_bki
