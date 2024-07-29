#include <iostream>
#include <vector>
#include <cmath>
#include <queue>
#include <set>
#include <map>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/distances.h>
#include <omp.h>

using namespace std;
using namespace pcl;

struct Point3D {
    double x, y, z;
    int clusterID;
    Point3D(double x, double y, double z) : x(x), y(y), z(z), clusterID(0) {}
};

void expandCluster(vector<Point3D> &points, vector<vector<int>> &neighborhoods, int idx, int clusterID, int minPts) {
    queue<int> seeds;
    seeds.push(idx);
    while (!seeds.empty()) {
        int currentPoint = seeds.front();
        seeds.pop();
        vector<int> &currentNeighbors = neighborhoods[currentPoint];
        if (currentNeighbors.size() >= minPts) {
            for (int i : currentNeighbors) {
                if (points[i].clusterID == 0) {
                    points[i].clusterID = clusterID;
                    seeds.push(i);
                }
            }
        }
    }
}

void dbscan(vector<Point3D> &points, double eps, int minPts) {
    int clusterID = 0;
    double epsSquared = eps * eps;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclPoints(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &point : points) {
        pclPoints->push_back(pcl::PointXYZ(point.x, point.y, point.z));
    }
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(pclPoints);
    vector<vector<int>> neighborhoods(points.size());
    #pragma omp parallel for
    for (int i = 0; i < points.size(); ++i) {
        if (points[i].clusterID == 0) {
            vector<int> neighbors;
            vector<float> distances;
            kdtree.radiusSearch(pclPoints->points[i], eps, neighbors, distances);
            for (size_t j = 0; j < neighbors.size(); ++j) {
                if (distances[j] <= epsSquared) {
                    neighborhoods[i].push_back(neighbors[j]);
                }
            }
            #pragma omp critical
            {
                if (neighborhoods[i].size() >= minPts) {
                    expandCluster(points, neighborhoods, i, ++clusterID, minPts);
                } else {
                    points[i].clusterID = -1;
                }
            }
        }
    }
}

class DbscanNode : public rclcpp::Node {
public:
    DbscanNode() : Node("dbscan_node") {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/carla/ego_vehicle/lidar", 10, std::bind(&DbscanNode::pointCloudCallback, this, std::placeholders::_1));
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("cluster_markers", 10);
        cluster_ids_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("cluster_ids", 10);
        avg_x_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("avg_x", 10);
        avg_y_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("avg_y", 10);
        avg_z_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("avg_z", 10);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromROSMsg(*msg, cloud);
        vector<Point3D> points;
        for (const auto &point : cloud) {
            if (point.x > 0.0 && point.x < 15.0 && point.y > -2.0 && point.y < 2.0 && point.z > -2.0 && point.z < 2.0) {
                points.emplace_back(point.x, point.y, point.z);
            }
        }
        double eps = 0.5;
        int minPts = 5;
        dbscan(points, eps, minPts);
        visualization_msgs::msg::MarkerArray marker_array;
        std::map<int, visualization_msgs::msg::Marker> cluster_markers;
        std::map<int, std::vector<Point3D>> clusters;
        for (const auto &point : points) {
            if (point.clusterID != -1) {
                clusters[point.clusterID].push_back(point);
            }
        }
        clearAllMarkers();

        std_msgs::msg::Float32MultiArray cluster_ids_msg;
        std_msgs::msg::Float32MultiArray avg_x_msg;
        std_msgs::msg::Float32MultiArray avg_y_msg;
        std_msgs::msg::Float32MultiArray avg_z_msg;

        for (const auto &cluster : clusters) {
            int clusterID = cluster.first;
            const auto &cluster_points = cluster.second;
            double avg_x = 0.0, avg_y = 0.0, avg_z = 0.0;
            for (const auto &point : cluster_points) {
                avg_x += point.x;
                avg_y += point.y;
                avg_z += point.z;
            }
            avg_x /= cluster_points.size();
            avg_y /= cluster_points.size();
            avg_z /= cluster_points.size();

            // Create marker for visualization
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = rclcpp::Clock().now();
            marker.ns = "clusters";
            marker.id = clusterID;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = avg_x;
            marker.pose.position.y = avg_y;
            marker.pose.position.z = avg_z + 2.0;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.15;
            marker.scale.y = 0.15;
            marker.scale.z = 0.15;
            marker.color.r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            marker.color.g = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            marker.color.b = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            marker.color.a = 1.0;
            marker_array.markers.push_back(marker);
            RCLCPP_INFO(this->get_logger(), "Cluster ID: %d -> Center(%f, %f, %f)", clusterID, avg_x, avg_y, avg_z);

            // Append data to each message
            cluster_ids_msg.data.push_back(clusterID);
            avg_x_msg.data.push_back(avg_x);
            avg_y_msg.data.push_back(avg_y);
            avg_z_msg.data.push_back(avg_z);
        }

        add2DBoundingBox(marker_array);

        marker_pub_->publish(marker_array);

        // Publish all messages
        cluster_ids_pub_->publish(cluster_ids_msg);
        avg_x_pub_->publish(avg_x_msg);
        avg_y_pub_->publish(avg_y_msg);
        avg_z_pub_->publish(avg_z_msg);

        RCLCPP_INFO(this->get_logger(), "Published %zu markers", marker_array.markers.size());
        RCLCPP_INFO(this->get_logger(), "Published cluster IDs with %zu entries", cluster_ids_msg.data.size());
        RCLCPP_INFO(this->get_logger(), "Published avg_x with %zu entries", avg_x_msg.data.size());
        RCLCPP_INFO(this->get_logger(), "Published avg_y with %zu entries", avg_y_msg.data.size());
        RCLCPP_INFO(this->get_logger(), "Published avg_z with %zu entries", avg_z_msg.data.size());
    }

    void clearAllMarkers() {
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker delete_marker;
        delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);
        marker_pub_->publish(marker_array);
    }

    void add2DBoundingBox(visualization_msgs::msg::MarkerArray &marker_array) {
        visualization_msgs::msg::Marker bbox_marker;
        bbox_marker.header.frame_id = "map";
        bbox_marker.header.stamp = rclcpp::Clock().now();
        bbox_marker.ns = "bounding_box";
        bbox_marker.id = 0;
        bbox_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        bbox_marker.action = visualization_msgs::msg::Marker::ADD;
        bbox_marker.scale.x = 0.1; // Line width

        bbox_marker.color.r = 1.0;
        bbox_marker.color.g = 1.0;
        bbox_marker.color.b = 1.0;
        bbox_marker.color.a = 1.0;

        geometry_msgs::msg::Point p1, p2, p3, p4, p5;

        p1.x = 0.0; p1.y = -2.0; p1.z = 0.0;
        p2.x = 10.0; p2.y = -2.0; p2.z = 0.0;
        p3.x = 10.0; p3.y = 2.0; p3.z = 0.0;
        p4.x = 0.0; p4.y = 2.0; p4.z = 0.0;
        p5 = p1; // Close the loop

        bbox_marker.points.push_back(p1);
        bbox_marker.points.push_back(p2);
        bbox_marker.points.push_back(p3);
        bbox_marker.points.push_back(p4);
        bbox_marker.points.push_back(p5);

        marker_array.markers.push_back(bbox_marker);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr cluster_ids_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr avg_x_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr avg_y_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr avg_z_pub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DbscanNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
