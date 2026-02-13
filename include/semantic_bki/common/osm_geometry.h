#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace semantic_bki {

    /// 2D polygon (list of (x,y) vertices; same convention as OSMVisualizer).
    struct Geometry2D {
        std::vector<std::pair<float, float>> coords;
    };

    /// Ray-casting point-in-polygon test (returns true if (px,py) is inside poly).
    inline bool point_in_polygon(float px, float py, const Geometry2D& poly) {
        const auto& c = poly.coords;
        if (c.size() < 3) return false;
        int n = static_cast<int>(c.size());
        bool inside = false;
        for (int i = 0, j = n - 1; i < n; j = i++) {
            float xi = c[i].first, yi = c[i].second;
            float xj = c[j].first, yj = c[j].second;
            if (((yi > py) != (yj > py)) &&
                (px < (xj - xi) * (py - yi) / (yj - yi) + xi))
                inside = !inside;
        }
        return inside;
    }

    /// Squared distance from point (px,py) to segment (ax,ay)-(bx,by).
    inline float segment_distance_sq(float px, float py, float ax, float ay, float bx, float by) {
        float dx = bx - ax, dy = by - ay;
        float len_sq = dx * dx + dy * dy;
        if (len_sq < 1e-12f) {
            float dx2 = px - ax, dy2 = py - ay;
            return dx2 * dx2 + dy2 * dy2;
        }
        float t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
        t = std::max(0.f, std::min(1.f, t));
        float qx = ax + t * dx, qy = ay + t * dy;
        float ddx = px - qx, ddy = py - qy;
        return ddx * ddx + ddy * ddy;
    }

    /// Minimum distance from (px,py) to polygon boundary. Returns 0 if inside polygon.
    inline float distance_to_polygon_boundary(float px, float py, const Geometry2D& poly) {
        const auto& c = poly.coords;
        if (c.size() < 2) return std::numeric_limits<float>::max();
        bool inside = point_in_polygon(px, py, poly);
        float min_d_sq = std::numeric_limits<float>::max();
        int n = static_cast<int>(c.size());
        for (int i = 0, j = n - 1; i < n; j = i++) {
            float d_sq = segment_distance_sq(px, py,
                c[j].first, c[j].second,
                c[i].first, c[i].second);
            if (d_sq < min_d_sq) min_d_sq = d_sq;
        }
        float d = std::sqrt(min_d_sq);
        return inside ? -d : d;  // negative = inside (signed distance)
    }

    /// Minimum distance from (px,py) to polyline (open path, not closed polygon).
    /// Returns distance to nearest segment.
    inline float distance_to_polyline(float px, float py, const Geometry2D& polyline) {
        const auto& c = polyline.coords;
        if (c.size() < 2) return std::numeric_limits<float>::max();
        float min_d_sq = std::numeric_limits<float>::max();
        for (size_t i = 0; i < c.size() - 1; ++i) {
            float d_sq = segment_distance_sq(px, py,
                c[i].first, c[i].second,
                c[i+1].first, c[i+1].second);
            if (d_sq < min_d_sq) min_d_sq = d_sq;
        }
        return std::sqrt(min_d_sq);
    }

    /// Minimum distance from (px,py) to nearest point in point list.
    inline float distance_to_points(float px, float py, const std::vector<std::pair<float, float>>& points) {
        if (points.empty()) return std::numeric_limits<float>::max();
        float min_d_sq = std::numeric_limits<float>::max();
        for (const auto& pt : points) {
            float dx = px - pt.first, dy = py - pt.second;
            float d_sq = dx * dx + dy * dy;
            if (d_sq < min_d_sq) min_d_sq = d_sq;
        }
        return std::sqrt(min_d_sq);
    }

    /// OSM prior value: inside -> 1.0, outside -> decay by distance (linear dropoff).
    /// signed_d: result of distance_to_polygon_boundary (negative inside, positive outside).
    /// decay_m: distance in meters over which prior drops from 1 to 0.
    inline float osm_prior_from_signed_distance(float signed_d, float decay_m) {
        if (decay_m <= 0.f) return signed_d <= 0.f ? 1.f : 0.f;
        if (signed_d <= 0.f) return 1.f;
        return std::max(0.f, 1.f - signed_d / decay_m);
    }

    /// OSM prior value for polylines/points: decay by distance (always positive distance).
    /// d: distance to nearest polyline segment or point.
    /// decay_m: distance in meters over which prior drops from 1 to 0.
    inline float osm_prior_from_distance(float d, float decay_m) {
        if (decay_m <= 0.f) return 0.f;
        return std::max(0.f, 1.f - d / decay_m);
    }
}
