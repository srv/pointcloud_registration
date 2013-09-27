#include "icp_registration.h"
#include <pcl17/filters/voxel_grid.h>
#include <pcl17/filters/passthrough.h>
#include <pcl17/filters/extract_indices.h>
#include <pcl17/features/normal_3d.h>
#include <pcl17/registration/transforms.h>

icp_registration::IcpRegistration::
IcpRegistration() : nh_private_("~")
{
  // Read the parameters from the parameter server (set defaults)
  nh_private_.param("publish_merged_pointcloud_topic", merged_pointcloud_topic_, std::string("/merged_pointcloud"));
  nh_private_.param("subscribe_pointcloud_topic", subscribe_pointcloud_topic_, std::string("/points2"));
  nh_private_.param("input_cue", input_cue_, 10);
  nh_private_.param("downsample_pointcloud_after", downsample_pointcloud_after_, false);
  nh_private_.param("max_number_of_iterations_icp", max_number_of_iterations_icp_, 100);
  nh_private_.param("epsilon_transformation", epsilon_transformation_, 1e-6);
  nh_private_.param("euclidean_fitness_epsilon", euclidean_fitness_epsilon_, 1.0);
  nh_private_.param("max_correspondence_distance", max_correspondence_distance_, 0.05);
  nh_private_.param("downsample_voxel_size", downsample_voxel_size_, 0.01);

  // Initialize parameters
  counter_ = 0;
  first_cloud_received_ = false;

  // Setup the icp algorithm
  icp_.setMaxCorrespondenceDistance(max_correspondence_distance_);
  icp_.setMaximumIterations(max_number_of_iterations_icp_);
  icp_.setTransformationEpsilon(epsilon_transformation_);
  icp_.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon_);
  icp_.setUseReciprocalCorrespondences(true);
  
  // Subscription to the point cloud result from stereo_image_proc
  point_cloud_sub_ = nh_.subscribe<PointCloudRGB>(
    subscribe_pointcloud_topic_,
    input_cue_,
    &IcpRegistration::pointCloudCb,
    this);

  // Declare the point cloud merget topic
  point_cloud_merged_pub_ = nh_private_.advertise<PointCloudRGB>(merged_pointcloud_topic_, 1); 

  ROS_INFO_STREAM("[IcpRegistration:] icp_registration node is up and running.");
}

icp_registration::IcpRegistration::
~IcpRegistration()
{
  ROS_INFO_STREAM("[IcpRegistration:] Shutting down icp_registration node!");
}

void icp_registration::IcpRegistration::
pointCloudCb(const PointCloudRGB::ConstPtr& point_cloud)
{
  counter_++;

  if( first_cloud_received_ == false)
  {
    first_cloud_received_ = true;
    pointcloud_current_ = *point_cloud;
    ROS_INFO_STREAM("[IcpRegistration:] Received first point cloud with " 
      << pointcloud_current_.points.size() << " points.");
    pointcloud_merged_ = pointcloud_current_;
  }
  else
  {
    pointcloud_current_ = *point_cloud;
    ROS_INFO_STREAM("[IcpRegistration:] Received point cloud number: " 
      << counter_ << " with " << pointcloud_current_.points.size() << " points.");

    // Align point clouds
    Eigen::Matrix4f transformation;
    ROS_INFO_STREAM("[IcpRegistration:] Aligning...");
    icp_.setInputTarget(pointcloud_merged_.makeShared());
    icp_.setInputSource(pointcloud_current_.makeShared());
    icp_.align(pointcloud_transformed_);

    ROS_INFO_STREAM("[IcpRegistration:] Has converged: " << icp_.hasConverged() 
      << " score: " << icp_.getFitnessScore());
    pointcloud_merged_ += pointcloud_transformed_;
  }

  // Publish point cloud
  publishPointCloud(pointcloud_merged_);

}

void icp_registration::IcpRegistration::
publishPointCloud(PointCloudRGB &point_cloud)
{
  PointCloudRGB::Ptr cloud_downsampled;

  if( downsample_pointcloud_after_ == true)
  {
    cloud_downsampled = downsamplePointCloud(point_cloud.makeShared());      
  }
  else
  {
    cloud_downsampled = point_cloud.makeShared();
  }

  point_cloud_merged_pub_.publish(cloud_downsampled);
  ROS_INFO_STREAM("[IcpRegistration:] Merged point cloud published with " 
    << cloud_downsampled->size() << " points.");
}

PointCloudRGB::Ptr icp_registration::IcpRegistration::
downsamplePointCloud(PointCloudRGB::Ptr cloud)
{
  // Downsampling using voxel grid
  pcl17::VoxelGrid<PointRGB> grid_;
  PointCloudRGB::Ptr cloud_downsampled_ptr(new PointCloudRGB);
  grid_.setLeafSize(downsample_voxel_size_,
                    downsample_voxel_size_,
                    downsample_voxel_size_);
  grid_.setDownsampleAllData(true);
  grid_.setInputCloud(cloud);
  grid_.filter(*cloud_downsampled_ptr);

  return cloud_downsampled_ptr;
}
