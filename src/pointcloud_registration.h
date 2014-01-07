#ifndef POINTCLOUD_REGISTRATION_H
#define POINTCLOUD_REGISTRATION_H

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>

#include <std_msgs/String.h>

#include <pcl17/point_types.h>
#include <pcl17_ros/point_cloud.h>
#include <pcl17/registration/icp.h>
#include <pcl17/registration/icp_nl.h>

#include <Eigen/SVD>

typedef pcl17::PointXYZ             Point;
typedef pcl17::PointCloud<Point>    PointCloud;
typedef pcl17::PointXYZRGB          PointRGB;
typedef pcl17::PointCloud<PointRGB> PointCloudRGB;

namespace pointcloud_registration
{

class PointCloudRegistration
{
public:
  PointCloudRegistration();
  ~PointCloudRegistration();

protected:
  Eigen::Matrix4f pairAlign(const PointCloudRGB::Ptr cloud_src, const PointCloudRGB::Ptr cloud_tgt);
  void pointCloudCb(const PointCloudRGB::ConstPtr& point_cloud);
  PointCloudRGB::Ptr pointcloudFilter(PointCloudRGB::Ptr cloud, int type);
  void odometryCb(const nav_msgs::OdometryConstPtr& odom);

private:
  // ROS Node handlers
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  // ROS topics
  ros::Subscriber point_cloud_sub_;
  ros::Subscriber odometry_sub_;
  ros::Publisher  point_cloud_merged_pub_;

  // Operational properties
  std::string merged_pointcloud_topic_;
  std::string subscribe_pointcloud_topic_;
  bool first_cloud_received_;
  int input_cue_;
  int counter_;

  // Point clouds
  PointCloudRGB pointcloud_current_;
  PointCloudRGB pointcloud_previous_;
  PointCloudRGB pointcloud_transformed_;
  PointCloudRGB pointcloud_merged_;
  Eigen::Matrix4f final_transformation_;

  // Registration parameters
  std::string registration_method_;
  bool pairwise_registration_;
  int max_number_of_iterations_;
  double epsilon_transformation_;
  double ndt_step_size_;
  double ndt_grid_resolution_;
  double icp_max_correspondence_distance_;
  double icp_euclidean_fitness_epsilon_;

  // Filtering properties
  bool input_filtering_;
  bool output_filtering_;

  // NAN and limits
  double in_filter_x_min_;
  double in_filter_x_max_;
  double in_filter_y_min_;
  double in_filter_y_max_;
  double in_filter_z_min_;
  double in_filter_z_max_;
  double out_filter_x_min_;
  double out_filter_x_max_;
  double out_filter_y_min_;
  double out_filter_y_max_;
  double out_filter_z_min_;
  double out_filter_z_max_;

  // Voxel grid
  double in_filter_voxel_size_;
  double out_filter_voxel_size_;

  // Statistical outlier removal
  int in_filter_mean_k_;
  int out_filter_mean_k_;
  double in_filter_std_dev_thresh_;
  double out_filter_std_dev_thresh_;

  // Odometry callback
  bool use_odometry_;
  std::string odometry_topic_;
  Eigen::Matrix4f odometry_matrix_;

};

} // namespace

#endif // POINTCLOUD_REGISTRATION_H
