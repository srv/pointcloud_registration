#ifndef ICP_REGISTRATION_H
#define ICP_REGISTRATION_H

#include <ros/ros.h>

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

namespace icp_registration
{

class IcpRegistration
{
public:
  IcpRegistration();
  ~IcpRegistration();

protected:
  void pointCloudCb(const PointCloudRGB::ConstPtr& point_cloud);
  void publishPointCloud(PointCloudRGB &point_cloud);
  PointCloudRGB::Ptr downsamplePointCloud(PointCloudRGB::Ptr cloud);

private:
  // ROS Node handlers
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  // ROS topics
  ros::Subscriber point_cloud_sub_;
  ros::Publisher  point_cloud_merged_pub_;

  // Properties
  bool first_cloud_received_;
  bool downsample_pointcloud_after_;
  PointCloudRGB pointcloud_current_;
  PointCloudRGB pointcloud_transformed_;
  PointCloudRGB pointcloud_merged_;
  pcl17::IterativeClosestPoint<PointRGB, PointRGB> icp_;
  Eigen::Matrix4f final_transformation_;
  int input_cue_;
  int max_number_of_iterations_icp_;
  int counter_;
  double downsample_voxel_size_;
  double epsilon_transformation_;
  double max_correspondence_distance_;
  double euclidean_fitness_epsilon_;
  std::string merged_pointcloud_topic_;
  std::string subscribe_pointcloud_topic_;
};

} // namespace

#endif // ICP_REGISTRATION_H
