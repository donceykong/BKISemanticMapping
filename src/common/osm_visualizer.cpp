#include "osm_visualizer.h"
#include <sstream>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <limits>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <osmium/io/any_input.hpp>
#include <osmium/handler.hpp>
#include <osmium/visitor.hpp>
#include <osmium/osm/way.hpp>
#include <osmium/osm/node.hpp>
#include <osmium/osm/relation.hpp>
#include <osmium/osm/area.hpp>
#include <osmium/index/map/sparse_mem_array.hpp>
#include <osmium/handler/node_locations_for_ways.hpp>
#include <osmium/osm/location.hpp>
#include <osmium/area/assembler.hpp>
#include <osmium/area/multipolygon_manager.hpp>
#include <osmium/tags/tags_filter.hpp>

namespace {
    // Handler class for extracting buildings, roads, sidewalks, grasslands, trees (ways + nodes + multipolygons) from OSM data using libosmium
    class OSMGeometryHandler : public osmium::handler::Handler {
    public:
        OSMGeometryHandler(double origin_lat, double origin_lon,
                          std::vector<semantic_bki::Geometry2D>& buildings,
                          std::vector<semantic_bki::Geometry2D>& roads,
                          std::vector<semantic_bki::Geometry2D>& grasslands,
                          std::vector<semantic_bki::Geometry2D>& trees,
                          std::vector<std::pair<float, float>>& tree_points)
            : origin_lat_(origin_lat), origin_lon_(origin_lon),
              buildings_(buildings), roads_(roads), grasslands_(grasslands), trees_(trees), tree_points_(tree_points) {
            kMetersPerDegLat_ = 110540.0;
            kMetersPerDegLon_ = 111320.0 * std::cos(origin_lat * M_PI / 180.0);
        }

        void node(const osmium::Node& node) {
            // Single-point trees: natural=tree (node with one coordinate)
            const char* natural_tag = node.tags()["natural"];
            if (!natural_tag || std::string(natural_tag) != "tree") {
                return;
            }
            const osmium::Location& loc = node.location();
            if (!loc.valid()) {
                return;
            }
            double lat = loc.lat();
            double lon = loc.lon();
            float x = static_cast<float>((lon - origin_lon_) * kMetersPerDegLon_);
            float y = static_cast<float>((lat - origin_lat_) * kMetersPerDegLat_);
            tree_points_.push_back({x, y});
        }

        void way(const osmium::Way& way) {
            // Extract node coordinates
            semantic_bki::Geometry2D geom;
            for (const auto& node_ref : way.nodes()) {
                const osmium::Location& location = node_ref.location();
                if (location.valid()) {
                    double lat = location.lat();
                    double lon = location.lon();
                    // Relative meters from origin (East, North) - same convention as ROS1 binary / create_scan_osm_topdown.py
                    float x = static_cast<float>((lon - origin_lon_) * kMetersPerDegLon_);
                    float y = static_cast<float>((lat - origin_lat_) * kMetersPerDegLat_);
                    geom.coords.push_back({x, y});
                }
            }

            if (geom.coords.size() < 2) {
                return;
            }

            // Check if this way is a building
            const char* building_tag = way.tags()["building"];
            if (building_tag) {
                if (geom.coords.size() >= 3) {
                    buildings_.push_back(geom);
                }
                return;
            }

            // Check if this way is a road or sidewalk
            const char* highway_tag = way.tags()["highway"];
            if (highway_tag) {
                std::string highway(highway_tag);
                if (highway == "motorway" || highway == "trunk" || highway == "primary" ||
                    highway == "secondary" || highway == "tertiary" ||
                    highway == "unclassified" || highway == "residential" ||
                    highway == "motorway_link" || highway == "trunk_link" ||
                    highway == "primary_link" || highway == "secondary_link" ||
                    highway == "tertiary_link" || highway == "living_street" ||
                    highway == "service" || highway == "pedestrian" ||
                    highway == "road" || highway == "cycleway" ||
                    highway == "footway" || highway == "path" || highway == "foot") {
                    roads_.push_back(geom);
                }
                return;
            }

            const char* footway_tag = way.tags()["footway"];
            if (footway_tag && std::string(footway_tag) == "sidewalk") {
                roads_.push_back(geom);
                return;
            }

            // Grassland: landuse=grass, natural=grassland, landuse=meadow, landuse=greenfield
            const char* landuse_tag = way.tags()["landuse"];
            if (landuse_tag) {
                std::string landuse(landuse_tag);
                if (landuse == "grass" || landuse == "meadow" || landuse == "greenfield") {
                    if (geom.coords.size() >= 3) {
                        grasslands_.push_back(geom);
                    }
                    return;
                }
            }
            const char* natural_tag = way.tags()["natural"];
            if (natural_tag) {
                std::string natural(natural_tag);
                if (natural == "grassland" || natural == "heath" || natural == "scrub") {
                    if (geom.coords.size() >= 3) {
                        grasslands_.push_back(geom);
                    }
                    return;
                }
                // Trees/forest: natural=wood, or landuse=forest
                if (natural == "wood" || natural == "forest") {
                    if (geom.coords.size() >= 3) {
                        trees_.push_back(geom);
                    }
                    return;
                }
            }
            if (landuse_tag && std::string(landuse_tag) == "forest") {
                if (geom.coords.size() >= 3) {
                    trees_.push_back(geom);
                }
            }
        }

        // Handle areas (from multipolygons or closed ways) - called by MultipolygonManager
        void area(const osmium::Area& area) {
            // Check tags to determine type
            const char* building_tag = area.tags()["building"];
            if (building_tag) {
                // Extract outer rings as separate polygons (each outer ring is a building)
                for (const auto& outer_ring : area.outer_rings()) {
                    semantic_bki::Geometry2D geom;
                    for (const auto& node_ref : outer_ring) {
                        const osmium::Location& location = node_ref.location();
                        if (location.valid()) {
                            double lat = location.lat();
                            double lon = location.lon();
                            float x = static_cast<float>((lon - origin_lon_) * kMetersPerDegLon_);
                            float y = static_cast<float>((lat - origin_lat_) * kMetersPerDegLat_);
                            geom.coords.push_back({x, y});
                        }
                    }
                    // Note: inner rings (holes) are ignored for now - we treat multipolygons as filled polygons
                    // To handle holes properly, we'd need to use a polygon-with-holes data structure
                    if (geom.coords.size() >= 3) {
                        buildings_.push_back(geom);
                    }
                }
                return;
            }

            // Check for grassland/meadow areas
            const char* landuse_tag = area.tags()["landuse"];
            if (landuse_tag) {
                std::string landuse(landuse_tag);
                if (landuse == "grass" || landuse == "meadow" || landuse == "greenfield") {
                    for (const auto& outer_ring : area.outer_rings()) {
                        semantic_bki::Geometry2D geom;
                        for (const auto& node_ref : outer_ring) {
                            const osmium::Location& location = node_ref.location();
                            if (location.valid()) {
                                double lat = location.lat();
                                double lon = location.lon();
                                float x = static_cast<float>((lon - origin_lon_) * kMetersPerDegLon_);
                                float y = static_cast<float>((lat - origin_lat_) * kMetersPerDegLat_);
                                geom.coords.push_back({x, y});
                            }
                        }
                        if (geom.coords.size() >= 3) {
                            grasslands_.push_back(geom);
                        }
                    }
                    return;
                }
            }

            // Check for forest/wood areas
            const char* natural_tag = area.tags()["natural"];
            if (natural_tag) {
                std::string natural(natural_tag);
                if (natural == "wood" || natural == "forest") {
                    for (const auto& outer_ring : area.outer_rings()) {
                        semantic_bki::Geometry2D geom;
                        for (const auto& node_ref : outer_ring) {
                            const osmium::Location& location = node_ref.location();
                            if (location.valid()) {
                                double lat = location.lat();
                                double lon = location.lon();
                                float x = static_cast<float>((lon - origin_lon_) * kMetersPerDegLon_);
                                float y = static_cast<float>((lat - origin_lat_) * kMetersPerDegLat_);
                                geom.coords.push_back({x, y});
                            }
                        }
                        if (geom.coords.size() >= 3) {
                            trees_.push_back(geom);
                        }
                    }
                    return;
                }
            }
            if (landuse_tag && std::string(landuse_tag) == "forest") {
                for (const auto& outer_ring : area.outer_rings()) {
                    semantic_bki::Geometry2D geom;
                    for (const auto& node_ref : outer_ring) {
                        const osmium::Location& location = node_ref.location();
                        if (location.valid()) {
                            double lat = location.lat();
                            double lon = location.lon();
                            float x = static_cast<float>((lon - origin_lon_) * kMetersPerDegLon_);
                            float y = static_cast<float>((lat - origin_lat_) * kMetersPerDegLat_);
                            geom.coords.push_back({x, y});
                        }
                    }
                    if (geom.coords.size() >= 3) {
                        trees_.push_back(geom);
                    }
                }
            }
        }

    private:
        double origin_lat_, origin_lon_;
        double kMetersPerDegLat_, kMetersPerDegLon_;
        std::vector<semantic_bki::Geometry2D>& buildings_;
        std::vector<semantic_bki::Geometry2D>& roads_;
        std::vector<semantic_bki::Geometry2D>& grasslands_;
        std::vector<semantic_bki::Geometry2D>& trees_;
        std::vector<std::pair<float, float>>& tree_points_;
    };
}

namespace semantic_bki {

    OSMVisualizer::OSMVisualizer(rclcpp::Node::SharedPtr node, const std::string& topic) 
        : node_(node), topic_(topic), frame_id_("map"), transformed_(false) {
        if (!node_) {
            RCLCPP_ERROR_STREAM(rclcpp::get_logger("osm_visualizer"), "ERROR: OSMVisualizer constructor: node_ is null!");
            return;
        }
        // Only create publisher if topic is not empty (allows using OSMVisualizer just for loading/transforming OSM data)
        if (!topic_.empty()) {
            // RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: OSMVisualizer constructor: Creating publisher for topic: " << topic_);
            // Use default QoS for ROS2 (compatible with ros2 topic hz and RViz)
            // transient_local requires matching QoS on subscriber side, which ros2 topic hz doesn't use
            // Use default reliable QoS instead for better compatibility
            pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
                topic_, 
                rclcpp::QoS(10).reliable());
            if (!pub_) {
                RCLCPP_ERROR_STREAM(node_->get_logger(), "ERROR: Failed to create OSM publisher!");
            } else {
                // RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: OSM publisher created successfully");
                // RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Publisher topic name: " << pub_->get_topic_name());
                // RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Publisher subscription count: " << pub_->get_subscription_count());
                // RCLCPP_INFO_STREAM(node_->get_logger(), "OSMVisualizer: Publisher created for topic: " << topic_ << " with reliable QoS");
            }
        } else {
            pub_ = nullptr;  // No publisher needed if topic is empty
        }
    }

    bool OSMVisualizer::loadFromOSM(const std::string& osm_file, double origin_lat, double origin_lon) {
        buildings_.clear();
        roads_.clear();
        grasslands_.clear();
        trees_.clear();
        tree_points_.clear();
        
        // RCLCPP_INFO_STREAM(node_->get_logger(), "OSMVisualizer::loadFromOSM called with file: " << osm_file);
        // RCLCPP_INFO_STREAM(node_->get_logger(), "  Origin: (" << origin_lat << ", " << origin_lon << ")");
        
        try {
            osmium::io::File input_file(osm_file);
            
            // Create index to store node locations
            osmium::index::map::SparseMemArray<osmium::unsigned_object_id_type, osmium::Location> index;
            
            // Handler to store node locations
            osmium::handler::NodeLocationsForWays<osmium::index::map::SparseMemArray<osmium::unsigned_object_id_type, osmium::Location>> 
                location_handler(index);
            
            // Create handler to extract buildings, roads, sidewalks, grasslands, trees (ways + point trees + multipolygons)
            OSMGeometryHandler handler(origin_lat, origin_lon, buildings_, roads_, grasslands_, trees_, tree_points_);
            
            // MultipolygonManager to convert multipolygon relations to areas
            // Configure assembler and filter for multipolygons (buildings, landuse, natural)
            osmium::area::AssemblerConfig assembler_config;
            osmium::TagsFilter filter(false);  // Start with false (reject all)
            filter.add_rule(true, "building");  // Accept buildings
            filter.add_rule(true, "landuse", "grass");
            filter.add_rule(true, "landuse", "meadow");
            filter.add_rule(true, "landuse", "greenfield");
            filter.add_rule(true, "landuse", "forest");
            filter.add_rule(true, "natural", "grassland");
            filter.add_rule(true, "natural", "heath");
            filter.add_rule(true, "natural", "scrub");
            filter.add_rule(true, "natural", "wood");
            filter.add_rule(true, "natural", "forest");
            using MultipolygonManager = osmium::area::MultipolygonManager<osmium::area::Assembler>;
            MultipolygonManager mp_manager(assembler_config, filter);
            
            // First pass: read all objects
            // - location_handler stores node locations
            // - handler processes ways and nodes (buildings, roads, etc. as simple ways)
            // - mp_manager collects multipolygon relations and their member ways
            osmium::io::Reader reader1(input_file);
            osmium::apply(reader1, location_handler, handler, mp_manager);
            reader1.close();
            
            // Prepare MultipolygonManager for second pass (required before lookup)
            mp_manager.prepare_for_lookup();
            
            // Second pass: read again, MultipolygonManager outputs completed areas (multipolygons)
            osmium::io::Reader reader2(input_file);
            osmium::apply(reader2, location_handler, mp_manager.handler([&handler](osmium::memory::Buffer&& buffer) {
                // This callback receives buffers containing completed areas (multipolygons)
                // Process only areas - handler.area() will be called for each area
                for (const auto& item : buffer) {
                    if (item.type() == osmium::item_type::area) {
                        handler.area(static_cast<const osmium::Area&>(item));
                    }
                }
            }));
            reader2.close();
            
            // RCLCPP_INFO_STREAM(node_->get_logger(), "Loaded " << buildings_.size() << " buildings, " << roads_.size() << " roads/sidewalks, " << grasslands_.size() << " grasslands, " << trees_.size() << " tree/forest polygons, " << tree_points_.size() << " tree points from OSM file using libosmium");
            
            if (buildings_.empty() && roads_.empty() && grasslands_.empty() && trees_.empty() && tree_points_.empty()) {
                // RCLCPP_WARN(node_->get_logger(), "WARNING: No buildings, roads, grasslands, or trees found in OSM file.");
            }
            
            size_t total_building_points = 0, total_road_points = 0, total_grass_points = 0, total_tree_points = 0;
            for (const auto& b : buildings_) total_building_points += b.coords.size();
            for (const auto& r : roads_) total_road_points += r.coords.size();
            for (const auto& g : grasslands_) total_grass_points += g.coords.size();
            for (const auto& t : trees_) total_tree_points += t.coords.size();
            // RCLCPP_INFO_STREAM(node_->get_logger(), "Total points - buildings: " << total_building_points << ", roads: " << total_road_points << ", grasslands: " << total_grass_points << ", tree polygons: " << total_tree_points << ", tree points: " << tree_points_.size());
            
            return true;
        } catch (const std::exception& e) {
            RCLCPP_ERROR_STREAM(node_->get_logger(), "Error parsing OSM file with libosmium: " << e.what());
            return false;
        }
    }

    visualization_msgs::msg::Marker OSMVisualizer::createBuildingMarker(const std::vector<Geometry2D>& buildings) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_buildings";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST; // Use LINE_LIST for building outlines
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.5; // Line width in meters
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0; // Fully opaque blue lines

        for (const auto& building : buildings) {
            if (building.coords.size() < 2) continue; // Need at least 2 points for a line
            
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
                // RCLCPP_WARN_STREAM(node_->get_logger(), "Skipping building polygon with invalid (NaN/Inf) coordinates");
                continue;
            }
            
            // Draw closed polygon outline by connecting consecutive points
            for (size_t i = 0; i < building.coords.size(); ++i) {
                geometry_msgs::msg::Point p1, p2;
                p1.x = building.coords[i].first;
                p1.y = building.coords[i].second;
                p1.z = 0.0;
                
                // Connect to next point (wrap around for closed polygon)
                size_t next_idx = (i + 1) % building.coords.size();
                p2.x = building.coords[next_idx].first;
                p2.y = building.coords[next_idx].second;
                p2.z = 0.0;
                
                // Add line segment: p1 -> p2
                marker.points.push_back(p1);
                marker.points.push_back(p2);
            }
        }

        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createRoadMarker(const std::vector<Geometry2D>& roads) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_roads";
        marker.id = 1;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST; // Use LINE_LIST for road polylines
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.3; // Line width in meters (thinner than buildings)
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0; // Fully opaque red lines

        for (const auto& road : roads) {
            if (road.coords.size() < 2) continue; // Need at least 2 points for a line
            
            // Check for NaN or invalid coordinates
            bool has_invalid = false;
            for (const auto& coord : road.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) ||
                    std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            if (has_invalid) {
                // RCLCPP_WARN_STREAM(node_->get_logger(), "Skipping road polyline with invalid (NaN/Inf) coordinates");
                continue;
            }
            
            // Draw polyline by connecting consecutive points
            for (size_t i = 0; i < road.coords.size() - 1; ++i) {
                geometry_msgs::msg::Point p1, p2;
                p1.x = road.coords[i].first;
                p1.y = road.coords[i].second;
                p1.z = 0.0;
                
                p2.x = road.coords[i + 1].first;
                p2.y = road.coords[i + 1].second;
                p2.z = 0.0;
                
                // Add line segment: p1 -> p2
                marker.points.push_back(p1);
                marker.points.push_back(p2);
            }
        }

        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createPathMarker() const {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "lidar_path";
        marker.id = 2;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.4;  // Line width
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;  // Green, opaque

        for (const auto& pt : path_) {
            geometry_msgs::msg::Point p;
            p.x = pt.first;
            p.y = pt.second;
            p.z = 0.0;
            marker.points.push_back(p);
        }
        return marker;
    }

    void OSMVisualizer::setPath(const std::vector<std::pair<float, float>>& path) {
        path_ = path;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createGrasslandMarker(const std::vector<Geometry2D>& grasslands) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_grasslands";
        marker.id = 3;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.25;
        marker.color.r = 0.4f;
        marker.color.g = 0.85f;
        marker.color.b = 0.35f;
        marker.color.a = 0.7f;

        for (const auto& grassland : grasslands) {
            if (grassland.coords.size() < 3) continue;
            bool has_invalid = false;
            for (const auto& coord : grassland.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) || std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            if (has_invalid) continue;
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
        marker.header.stamp = node_->now();
        marker.ns = "osm_trees";
        marker.id = 4;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.3;
        marker.color.r = 0.1f;
        marker.color.g = 0.5f;
        marker.color.b = 0.2f;
        marker.color.a = 0.9f;

        for (const auto& tree : trees) {
            if (tree.coords.size() < 3) continue;
            bool has_invalid = false;
            for (const auto& coord : tree.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) || std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            if (has_invalid) continue;
            for (size_t i = 0; i < tree.coords.size(); ++i) {
                geometry_msgs::msg::Point p1, p2;
                p1.x = tree.coords[i].first;
                p1.y = tree.coords[i].second;
                p1.z = 0.0;
                size_t next_idx = (i + 1) % tree.coords.size();
                p2.x = tree.coords[next_idx].first;
                p2.y = tree.coords[next_idx].second;
                p2.z = 0.0;
                marker.points.push_back(p1);
                marker.points.push_back(p2);
            }
        }
        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createTreePointsMarker() const {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_tree_points";
        marker.id = 5;
        marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 10.0;  // sphere diameter in meters (10x for visibility)
        marker.scale.y = 10.0;
        marker.scale.z = 10.0;
        marker.color.r = 0.1f;
        marker.color.g = 0.5f;
        marker.color.b = 0.2f;
        marker.color.a = 0.9f;
        for (const auto& pt : tree_points_) {
            if (std::isnan(pt.first) || std::isnan(pt.second) || std::isinf(pt.first) || std::isinf(pt.second)) continue;
            geometry_msgs::msg::Point p;
            p.x = pt.first;
            p.y = pt.second;
            p.z = 0.0;
            marker.points.push_back(p);
        }
        return marker;
    }

    void OSMVisualizer::publish() {
        // RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: publish() called, buildings_.size()=" << buildings_.size());
        
        if (!pub_) {
            RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Cannot publish - publisher is null!");
            return;
        }
        
        if (!node_) {
            RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Cannot publish - node is null!");
            return;
        }
        
        // RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Publisher and node are valid, creating marker array");
        visualization_msgs::msg::MarkerArray marker_array;
        
        if (!buildings_.empty()) {
            auto marker = createBuildingMarker(buildings_);
            marker_array.markers.push_back(marker);
            // RCLCPP_INFO_STREAM(node_->get_logger(), "OSM: Added " << buildings_.size() << " buildings with " << marker.points.size() << " line points");
        } else {
            // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: No buildings loaded! buildings_.size()=" << buildings_.size());
        }
        
        if (!roads_.empty()) {
            auto marker = createRoadMarker(roads_);
            marker_array.markers.push_back(marker);
            // RCLCPP_INFO_STREAM(node_->get_logger(), "OSM: Added " << roads_.size() << " roads/sidewalks with " << marker.points.size() << " line points");
        } else {
            // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: No roads loaded! roads_.size()=" << roads_.size());
        }
        
        if (!grasslands_.empty()) {
            auto marker = createGrasslandMarker(grasslands_);
            marker_array.markers.push_back(marker);
            // RCLCPP_INFO_STREAM(node_->get_logger(), "OSM: Added " << grasslands_.size() << " grasslands with " << marker.points.size() << " line points");
        }
        
        if (!trees_.empty()) {
            auto marker = createTreeMarker(trees_);
            marker_array.markers.push_back(marker);
            // RCLCPP_INFO_STREAM(node_->get_logger(), "OSM: Added " << trees_.size() << " trees/forests with " << marker.points.size() << " line points");
        }
        if (!tree_points_.empty()) {
            auto marker = createTreePointsMarker();
            marker_array.markers.push_back(marker);
            // RCLCPP_INFO_STREAM(node_->get_logger(), "OSM: Added " << tree_points_.size() << " tree points (single-node trees)");
        }
        
        if (!path_.empty()) {
            auto marker = createPathMarker();
            marker_array.markers.push_back(marker);
            // RCLCPP_INFO_STREAM(node_->get_logger(), "OSM: Added lidar path with " << path_.size() << " points");
        }
        
        if (marker_array.markers.empty()) {
            // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: No markers to publish! (buildings=" << buildings_.size() << ", roads=" << roads_.size() << ", grasslands=" << grasslands_.size() << ", trees=" << trees_.size() << ", tree_points=" << tree_points_.size() << ", path=" << path_.size() << ")");
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
        
        int total_points = 0;
        for (const auto& m : marker_array.markers) {
            total_points += m.points.size();
        }
        // RCLCPP_INFO_STREAM(node_->get_logger(), "OSMVisualizer: Publishing " << marker_array.markers.size() << " marker(s) (buildings + roads) with " << total_points << " total line points to topic " << topic_ << " in frame " << frame_id_);
        
        // Log marker details and bounding box for debugging
        if (!marker_array.markers.empty()) {
            const auto& first_marker = marker_array.markers[0];
            // RCLCPP_INFO_STREAM(node_->get_logger(), "First marker: ns=" << first_marker.ns << ", id=" << first_marker.id << ", type=" << first_marker.type << ", points=" << first_marker.points.size() << ", frame=" << first_marker.header.frame_id);
            
            // Calculate bounding box of marker points
            if (!first_marker.points.empty()) {
                float min_x = std::numeric_limits<float>::max();
                float max_x = std::numeric_limits<float>::lowest();
                float min_y = std::numeric_limits<float>::max();
                float max_y = std::numeric_limits<float>::lowest();
                for (const auto& pt : first_marker.points) {
                    min_x = std::min(min_x, static_cast<float>(pt.x));
                    max_x = std::max(max_x, static_cast<float>(pt.x));
                    min_y = std::min(min_y, static_cast<float>(pt.y));
                    max_y = std::max(max_y, static_cast<float>(pt.y));
                }
                // RCLCPP_INFO_STREAM(node_->get_logger(), "Marker bounds: X=[" << min_x << ", " << max_x << "], Y=[" << min_y << ", " << max_y << "]");
                // RCLCPP_INFO_STREAM(node_->get_logger(), "First point: [" << first_marker.points[0].x << ", " << first_marker.points[0].y << ", " << first_marker.points[0].z << "]");
            }
        }
        
        // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: About to publish MarkerArray with " << marker_array.markers.size() << " markers");
        // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Publisher valid: " << (pub_ ? "YES" : "NO"));
        // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Topic: " << topic_);
        
        try {
            pub_->publish(marker_array);
            // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Successfully published MarkerArray to " << topic_ << " with " << marker_array.markers.size() << " markers");
        } catch (const std::exception& e) {
            RCLCPP_ERROR_STREAM(node_->get_logger(), "OSMVisualizer: Exception while publishing: " << e.what());
        }
    }

    void OSMVisualizer::startPeriodicPublishing(double rate) {
        if (!pub_) {
            RCLCPP_WARN(node_->get_logger(), "OSMVisualizer: Cannot start periodic publishing - no publisher (topic was empty)");
            return;
        }
        if (rate <= 0.0) {
            // RCLCPP_WARN(node_->get_logger(), "OSMVisualizer: Invalid publishing rate, using default 1.0 Hz");
            rate = 1.0;
        }
        // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Creating timer for periodic publishing at " << rate << " Hz");
        // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Node pointer: " << node_.get() << ", this pointer: " << this);
        // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Publisher pointer: " << pub_.get() << ", Publisher count: " << pub_.use_count());
        
        publish_timer_ = node_->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / rate)),
            [this]() {
                this->timerCallback();
            });
        if (!publish_timer_) {
            // RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Failed to create publishing timer!");
        } else {
            // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Timer created successfully, periodic publishing started at " << rate << " Hz");
            // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Timer will fire every " << (1000.0 / rate) << " ms");
            // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Timer pointer: " << publish_timer_.get() << ", Timer count: " << publish_timer_.use_count());
        }
    }

    void OSMVisualizer::timerCallback() {
        try {
            // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Timer callback triggered, republishing markers (buildings=" << buildings_.size() << ", roads=" << roads_.size() << ", grasslands=" << grasslands_.size() << ", trees=" << trees_.size() << ", tree_points=" << tree_points_.size() << ")");
            if (!node_) {
                RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Timer callback - node_ is null!");
                return;
            }
            if (!pub_) {
                RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Timer callback - pub_ is null!");
                return;
            }
            if (!publish_timer_) {
                RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Timer callback - publish_timer_ is null!");
                return;
            }
            // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: About to call publish(), pub_ count: " << pub_.use_count());
            publish();
            // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Timer callback completed successfully, markers published");
        } catch (const std::exception& e) {
            RCLCPP_ERROR_STREAM(node_->get_logger(), "OSMVisualizer: Exception in timer callback: " << e.what());
        } catch (...) {
            RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Unknown exception in timer callback!");
        }
    }

    void OSMVisualizer::transformToFirstPoseOrigin(const Eigen::Matrix4d& first_pose) {
        // Prevent multiple transformations
        if (transformed_) {
            // RCLCPP_WARN(node_->get_logger(), "OSM data has already been transformed. Skipping additional transformation to prevent double transformation.");
            return;
        }
        
        // Same as BKISemanticMapping_ROS1_ORIG: OSM coordinates are "relative to origin_latlon" (East, North in meters).
        // Binary / Python use: world = first_pose_position + relative; then map = first_pose_inverse * world.
        // So we first add first_pose translation to get world coords, then apply first_pose_inverse.
        Eigen::Matrix4d first_pose_inverse = first_pose.inverse();
        
        // RCLCPP_INFO_STREAM(node_->get_logger(), "Transforming OSM geometries to be relative to first pose (same as ROS1_ORIG)...");
        // RCLCPP_INFO_STREAM(node_->get_logger(), "First pose translation: [" << first_pose(0,3) << ", " << first_pose(1,3) << ", " << first_pose(2,3) << "]");
        // RCLCPP_INFO_STREAM(node_->get_logger(), "Step 1: local_to_world = point + first_pose_translation; Step 2: map = first_pose_inverse * world");
        
        auto transformPoint = [&first_pose, &first_pose_inverse](float& x, float& y) {
            // Local (relative to origin_latlon) -> world: add first pose translation (as in create_scan_osm_topdown.py)
            double world_x = x + first_pose(0, 3);
            double world_y = y + first_pose(1, 3);
            Eigen::Vector4d point(world_x, world_y, 0.0, 1.0);
            Eigen::Vector4d transformed = first_pose_inverse * point;
            x = static_cast<float>(transformed(0));
            y = static_cast<float>(transformed(1));
        };
        
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
        
        // Transform grasslands and trees
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
        for (auto& pt : tree_points_) {
            transformPoint(pt.first, pt.second);
        }
        
        // Log bounding box after transformation
        if (!buildings_.empty() || !roads_.empty()) {
            float min_x_after = std::numeric_limits<float>::max();
            float max_x_after = std::numeric_limits<float>::lowest();
            float min_y_after = std::numeric_limits<float>::max();
            float max_y_after = std::numeric_limits<float>::lowest();
            for (const auto& building : buildings_) {
                for (const auto& coord : building.coords) {
                    min_x_after = std::min(min_x_after, coord.first);
                    max_x_after = std::max(max_x_after, coord.first);
                    min_y_after = std::min(min_y_after, coord.second);
                    max_y_after = std::max(max_y_after, coord.second);
                }
            }
            for (const auto& road : roads_) {
                for (const auto& coord : road.coords) {
                    min_x_after = std::min(min_x_after, coord.first);
                    max_x_after = std::max(max_x_after, coord.first);
                    min_y_after = std::min(min_y_after, coord.second);
                    max_y_after = std::max(max_y_after, coord.second);
                }
            }
            for (const auto& g : grasslands_) {
                for (const auto& coord : g.coords) {
                    min_x_after = std::min(min_x_after, coord.first);
                    max_x_after = std::max(max_x_after, coord.first);
                    min_y_after = std::min(min_y_after, coord.second);
                    max_y_after = std::max(max_y_after, coord.second);
                }
            }
            for (const auto& t : trees_) {
                for (const auto& coord : t.coords) {
                    min_x_after = std::min(min_x_after, coord.first);
                    max_x_after = std::max(max_x_after, coord.first);
                    min_y_after = std::min(min_y_after, coord.second);
                    max_y_after = std::max(max_y_after, coord.second);
                }
            }
            // RCLCPP_INFO_STREAM(node_->get_logger(), "OSM geometries AFTER transform - Bounds: [" << min_x_after << ", " << min_y_after << "] to [" << max_x_after << ", " << max_y_after << "]");
        }
        
        transformed_ = true;
        
        // RCLCPP_INFO_STREAM(node_->get_logger(), "OSM geometries (buildings, roads, grasslands, trees) transformed to first pose origin frame.");
    }

    bool OSMVisualizer::saveAsPNG(const std::string& output_path, int image_width, int image_height, int margin_pixels) {
        if (buildings_.empty() && roads_.empty() && grasslands_.empty() && trees_.empty() && tree_points_.empty() && path_.empty()) {
            // RCLCPP_WARN(node_->get_logger(), "No buildings, roads, grasslands, trees, tree points, or path to render in PNG");
            return false;
        }

        // Find bounding box of all buildings and roads
        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float min_y = std::numeric_limits<float>::max();
        float max_y = std::numeric_limits<float>::lowest();

        for (const auto& building : buildings_) {
            for (const auto& coord : building.coords) {
                min_x = std::min(min_x, coord.first);
                max_x = std::max(max_x, coord.first);
                min_y = std::min(min_y, coord.second);
                max_y = std::max(max_y, coord.second);
            }
        }
        
        for (const auto& road : roads_) {
            for (const auto& coord : road.coords) {
                min_x = std::min(min_x, coord.first);
                max_x = std::max(max_x, coord.first);
                min_y = std::min(min_y, coord.second);
                max_y = std::max(max_y, coord.second);
            }
        }
        for (const auto& g : grasslands_) {
            for (const auto& coord : g.coords) {
                min_x = std::min(min_x, coord.first);
                max_x = std::max(max_x, coord.first);
                min_y = std::min(min_y, coord.second);
                max_y = std::max(max_y, coord.second);
            }
        }
        for (const auto& t : trees_) {
            for (const auto& coord : t.coords) {
                min_x = std::min(min_x, coord.first);
                max_x = std::max(max_x, coord.first);
                min_y = std::min(min_y, coord.second);
                max_y = std::max(max_y, coord.second);
            }
        }
        for (const auto& pt : tree_points_) {
            min_x = std::min(min_x, pt.first);
            max_x = std::max(max_x, pt.first);
            min_y = std::min(min_y, pt.second);
            max_y = std::max(max_y, pt.second);
        }
        for (const auto& pt : path_) {
            min_x = std::min(min_x, pt.first);
            max_x = std::max(max_x, pt.first);
            min_y = std::min(min_y, pt.second);
            max_y = std::max(max_y, pt.second);
        }

        if (min_x >= max_x || min_y >= max_y) {
            RCLCPP_ERROR(node_->get_logger(), "Invalid bounding box for OSM geometries");
            return false;
        }

        // Calculate scale and offset to fit all geometries in image
        float range_x = max_x - min_x;
        float range_y = max_y - min_y;
        float scale = std::min(
            (image_width - 2 * margin_pixels) / range_x,
            (image_height - 2 * margin_pixels) / range_y
        );

        float offset_x = margin_pixels - min_x * scale;
        float offset_y = margin_pixels - min_y * scale;

        // Create white background image
        cv::Mat image = cv::Mat::ones(image_height, image_width, CV_8UC3) * 255;

        // Draw lidar path (green) first
        if (path_.size() >= 2) {
            cv::Scalar path_color(0, 255, 0); // Green (BGR)
            for (size_t i = 0; i < path_.size() - 1; ++i) {
                int px1 = static_cast<int>(path_[i].first * scale + offset_x);
                int py1 = static_cast<int>(path_[i].second * scale + offset_y);
                py1 = image_height - py1;
                int px2 = static_cast<int>(path_[i + 1].first * scale + offset_x);
                int py2 = static_cast<int>(path_[i + 1].second * scale + offset_y);
                py2 = image_height - py2;
                cv::line(image, cv::Point(px1, py1), cv::Point(px2, py2), path_color, 2);
            }
        }

        // Draw grasslands (light green outlines)
        cv::Scalar grassland_color(90, 217, 90); // BGR light green
        for (const auto& grassland : grasslands_) {
            if (grassland.coords.size() < 3) continue;
            std::vector<cv::Point> points;
            for (const auto& coord : grassland.coords) {
                int px = static_cast<int>(coord.first * scale + offset_x);
                int py = static_cast<int>(coord.second * scale + offset_y);
                py = image_height - py;
                points.push_back(cv::Point(px, py));
            }
            for (size_t i = 0; i < points.size(); ++i) {
                size_t next_i = (i + 1) % points.size();
                cv::line(image, points[i], points[next_i], grassland_color, 2);
            }
        }

        // Draw trees/forest (dark green outlines)
        cv::Scalar tree_color(51, 128, 26); // BGR dark green
        for (const auto& tree : trees_) {
            if (tree.coords.size() < 3) continue;
            std::vector<cv::Point> points;
            for (const auto& coord : tree.coords) {
                int px = static_cast<int>(coord.first * scale + offset_x);
                int py = static_cast<int>(coord.second * scale + offset_y);
                py = image_height - py;
                points.push_back(cv::Point(px, py));
            }
            for (size_t i = 0; i < points.size(); ++i) {
                size_t next_i = (i + 1) % points.size();
                cv::line(image, points[i], points[next_i], tree_color, 2);
            }
        }
        // Draw single-point trees (natural=tree nodes) as small circles
        const int tree_point_radius = std::max(2, static_cast<int>(30.0 * scale));  // ~30m in world (10x), at least 2px
        for (const auto& pt : tree_points_) {
            if (std::isnan(pt.first) || std::isnan(pt.second)) continue;
            int px = static_cast<int>(pt.first * scale + offset_x);
            int py = static_cast<int>(pt.second * scale + offset_y);
            py = image_height - py;
            cv::circle(image, cv::Point(px, py), tree_point_radius, tree_color, -1);
        }

        // Draw roads (red)
        cv::Scalar road_color(0, 0, 255); // Red color (BGR format)
        for (const auto& road : roads_) {
            if (road.coords.size() < 2) continue;
            
            // Convert road coordinates to image coordinates and draw polyline
            for (size_t i = 0; i < road.coords.size() - 1; ++i) {
                int px1 = static_cast<int>(road.coords[i].first * scale + offset_x);
                int py1 = static_cast<int>(road.coords[i].second * scale + offset_y);
                py1 = image_height - py1; // Flip Y axis
                
                int px2 = static_cast<int>(road.coords[i + 1].first * scale + offset_x);
                int py2 = static_cast<int>(road.coords[i + 1].second * scale + offset_y);
                py2 = image_height - py2; // Flip Y axis
                
                cv::line(image, cv::Point(px1, py1), cv::Point(px2, py2), road_color, 2);
            }
        }

        // Draw buildings
        cv::Scalar building_color(77, 77, 204); // Blue color (BGR format)
        cv::Scalar building_outline(0, 0, 255); // Red outline for visibility

        for (const auto& building : buildings_) {
            if (building.coords.size() < 3) continue;

            // Convert building coordinates to image coordinates
            std::vector<cv::Point> points;
            for (const auto& coord : building.coords) {
                int px = static_cast<int>(coord.first * scale + offset_x);
                int py = static_cast<int>(coord.second * scale + offset_y);
                // Flip Y axis (image coordinates have origin at top-left)
                py = image_height - py;
                points.push_back(cv::Point(px, py));
            }

            // Fill polygon
            if (points.size() >= 3) {
                const cv::Point* pts = &points[0];
                int npts = static_cast<int>(points.size());
                cv::fillPoly(image, &pts, &npts, 1, building_color);

                // Draw outline
                for (size_t i = 0; i < points.size(); ++i) {
                    size_t next_i = (i + 1) % points.size();
                    cv::line(image, points[i], points[next_i], building_outline, 2);
                }
            }
        }

        // Add coordinate info as text
        std::stringstream info;
        info << "B:" << buildings_.size() << " R:" << roads_.size() << " G:" << grasslands_.size() << " T:" << trees_.size() << " TP:" << tree_points_.size() << " P:" << path_.size() << " | ";
        info << "Bounds: [" << min_x << ", " << min_y << "] to [" << max_x << ", " << max_y << "]";
        cv::putText(image, info.str(), cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

        // Save image
        bool success = cv::imwrite(output_path, image);
        if (success) {
            // RCLCPP_INFO_STREAM(node_->get_logger(), "Saved OSM visualization to: " << output_path);
            // RCLCPP_INFO_STREAM(node_->get_logger(), "  Image size: " << image_width << "x" << image_height);
            // RCLCPP_INFO_STREAM(node_->get_logger(), "  Buildings: " << buildings_.size() << ", Roads: " << roads_.size() << ", Grasslands: " << grasslands_.size() << ", Trees: " << trees_.size());
        } else {
            RCLCPP_ERROR_STREAM(node_->get_logger(), "Failed to save PNG image to: " << output_path);
        }

        return success;
    }

} // namespace semantic_bki
