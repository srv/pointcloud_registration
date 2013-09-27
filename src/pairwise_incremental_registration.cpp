#include "pairwise_incremental_registration.h"
#include <pcl17/filters/voxel_grid.h>
#include <pcl17/filters/passthrough.h>
#include <pcl17/filters/extract_indices.h>
#include <pcl17/filters/statistical_outlier_removal.h>
#include <pcl17/features/normal_3d.h>
#include <pcl17/registration/transforms.h>

/** 
  * \brief Constructor: read node parameters and init the variables
  */
pairwise_incremental_registration::PairwiseIncrementalRegistration::
PairwiseIncrementalRegistration() : nh_private_("~")
{
  // Read the parameters from the parameter server (set defaults)
  nh_private_.param("publish_merged_pointcloud_topic", merged_pointcloud_topic_, std::string("/merged_pointcloud"));
  nh_private_.param("subscribe_pointcloud_topic", subscribe_pointcloud_topic_, std::string("/points2"));
  nh_private_.param("input_cue", input_cue_, 10);
  nh_private_.param("downsample_pointcloud_after", downsample_pointcloud_after_, false);
  nh_private_.param("max_number_of_iterations_icp", max_number_of_iterations_icp_, 2);
  nh_private_.param("epsilon_transformation", epsilon_transformation_, 1e-6);
  nh_private_.param("euclidean_fitness_epsilon", euclidean_fitness_epsilon_, 1.0);
  nh_private_.param("max_correspondence_distance", max_correspondence_distance_, 0.2);
  nh_private_.param("downsample_voxel_size", downsample_voxel_size_, 0.01);

  // Initialize parameters
  counter_ = 0;
  first_cloud_received_ = false;
  final_transformation_ = Eigen::Matrix4f::Identity();
  
  // Subscription to the point cloud result from stereo_image_proc
  point_cloud_sub_ = nh_.subscribe<PointCloudRGB>(
    subscribe_pointcloud_topic_,
    input_cue_,
    &PairwiseIncrementalRegistration::pointCloudCb,
    this);

  // Declare the point cloud merget topic
  point_cloud_merged_pub_ = nh_private_.advertise<PointCloudRGB>(merged_pointcloud_topic_, 1); 

  ROS_INFO_STREAM("[PairwiseIncrementalRegistration:] pairwise_incremental_registration node is up and running.");
}

/** 
  * \brief Destructor
  */
pairwise_incremental_registration::PairwiseIncrementalRegistration::
~PairwiseIncrementalRegistration()
{
  ROS_INFO_STREAM("[PairwiseIncrementalRegistration:] Shutting down pairwise_incremental_registration node!");
}

/** \brief Align a pair of PointCloud datasets and return the result
  * \param cloud_src the source PointCloud
  * \param cloud_tgt the target PointCloud
  * \param output the resultant aligned source PointCloud
  * \param final_transform the resultant transform between source and target
  */
Eigen::Matrix4f pairwise_incremental_registration::PairwiseIncrementalRegistration::
pairAlign(const PointCloudRGB::Ptr cloud_src, const PointCloudRGB::Ptr cloud_tgt)
{
  int iter = 20;
  double sum_scores = 0.0;

  PointCloudRGB::Ptr src(new PointCloudRGB);
  PointCloudRGB::Ptr tgt(new PointCloudRGB);
  src = cloud_src;
  tgt = cloud_tgt;
  
  // Compute surface normals and curvature
  PointCloudWithNormals::Ptr points_with_normals_src(new PointCloudWithNormals);
  PointCloudWithNormals::Ptr points_with_normals_tgt(new PointCloudWithNormals);

  pcl17::NormalEstimation<PointRGB, PointNormal> norm_est;
  pcl17::search::KdTree<PointRGB>::Ptr tree(new pcl17::search::KdTree<PointRGB>());
  norm_est.setSearchMethod(tree);
  norm_est.setKSearch(50);
  
  norm_est.setInputCloud(src);
  norm_est.compute(*points_with_normals_src);
  pcl17::copyPointCloud(*src, *points_with_normals_src);

  norm_est.setInputCloud(tgt);
  norm_est.compute(*points_with_normals_tgt);
  pcl17::copyPointCloud(*tgt, *points_with_normals_tgt);

  // Align
  pcl17::IterativeClosestPoint<PointNormal, PointNormal> reg;
  reg.setTransformationEpsilon(epsilon_transformation_);

  // Set the maximum distance between two correspondences (src<->tgt)
  // Note: adjust this based on the size of your datasets
  reg.setMaxCorrespondenceDistance(max_correspondence_distance_); 

  // Set the point representation
  reg.setInputSource(points_with_normals_src);
  reg.setInputTarget(points_with_normals_tgt);

  // Run the same optimization in a loop and visualize the results
  Eigen::Matrix4f ti = Eigen::Matrix4f::Identity(), prev;
  PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
  reg.setMaximumIterations(max_number_of_iterations_icp_);

  /*//////////////////////////////
  reg.align(*reg_result);

  // Accumulate transformation between each Iteration
  ti = reg.getFinalTransformation() * ti;

  mean_icp_score_ = reg.getFitnessScore();
  /////////////////////////////*/

  for (int i = 0; i < iter; ++i)
  {
    // save cloud for visualization purpose
    points_with_normals_src = reg_result;

    // Estimate
    reg.setInputSource(points_with_normals_src);
    reg.align(*reg_result);

    // Accumulate transformation between each Iteration
    ti = reg.getFinalTransformation() * ti;

    // If the difference between this transformation and the previous one
    // is smaller than the threshold, refine the process by reducing
    // the maximal correspondence distance
    if (fabs((reg.getLastIncrementalTransformation() - prev).sum()) < reg.getTransformationEpsilon())
    {
      reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance() - 0.001);
      ROS_INFO_STREAM("[PairwiseIncrementalRegistration:] Reducing the maximal correspondence distance to " <<
        reg.getMaxCorrespondenceDistance());
    }
    
    prev = reg.getLastIncrementalTransformation();

    if (reg.hasConverged())
    {
      ROS_INFO_STREAM("[PairwiseIncrementalRegistration:] Iteration Nr. " << i <<
      " has converged with score of: " << reg.getFitnessScore());
    }
    else
    {
      ROS_INFO_STREAM("[PairwiseIncrementalRegistration:] Iteration Nr. " << i <<
      " has NOT converged!");
    }

    sum_scores += reg.getFitnessScore();    
  }

  mean_icp_score_ = sum_scores/iter;

  return ti;
}

/** \brief Point cloud callback
  * \param point_cloud the input point cloud received
  */
void pairwise_incremental_registration::PairwiseIncrementalRegistration::
pointCloudCb(const PointCloudRGB::ConstPtr& point_cloud)
{
  counter_++;

  if( first_cloud_received_ == false)
  {
    first_cloud_received_ = true;
    pointcloud_current_ = *point_cloud;
    ROS_INFO_STREAM("[PointCloudRegistration:] Received first point cloud with " 
      << pointcloud_current_.points.size() << " points.");

    // Initialize the point cloud merged
    pointcloud_merged_ = pointcloud_current_;

    // Update for next callback
    pointcloud_previous_ = pointcloud_current_;
  }
  else
  {
    pointcloud_current_ = *point_cloud;
    ROS_INFO_STREAM("[PointCloudRegistration:] Received point cloud number: " 
      << counter_ << " with " << pointcloud_current_.points.size() << " points.");

    // Align point clouds
    // Eigen::Matrix4f pair_transform;
    final_transformation_ = pairAlign(pointcloud_current_.makeShared(), pointcloud_merged_.makeShared());

    // Merge pointcloud with the accumulated one if score is good enough
    if (mean_icp_score_ < 0.02)
    {
      // Update the global transform
      //final_transformation_ *= pair_transform;

      // Transform current pair into the global transform
      pcl17::transformPointCloud(pointcloud_current_, pointcloud_transformed_, final_transformation_);

      // Accumulate pointclouds
      pointcloud_merged_ += pointcloud_transformed_;

      // Update for next callback
      pointcloud_previous_ = pointcloud_current_;
    }
    else
    {
      ROS_INFO_STREAM("[PointCloudRegistration:] Point cloud not merged due to its final poor score: " << mean_icp_score_);
    }
  }

  // Filter the merged pointcloud
  PointCloudRGB::Ptr temp;
  temp = outlierRemovalPointCloud(pointcloud_merged_.makeShared());
  temp = downsamplePointCloud(temp);
  pointcloud_merged_ = *temp;

  // Publish point cloud
  publishPointCloud(pointcloud_merged_);
}

/** \brief Publishes the resulting point cloud
  * \param point_cloud point cloud to be published
  */
void pairwise_incremental_registration::PairwiseIncrementalRegistration::
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
  ROS_INFO_STREAM("[PairwiseIncrementalRegistration:] Merged point cloud published with " 
    << cloud_downsampled->size() << " points.");
}

/** \brief Downsample the point cloud
  * \param cloud point cloud to be downsampled
  */
PointCloudRGB::Ptr pairwise_incremental_registration::PairwiseIncrementalRegistration::
downsamplePointCloud(PointCloudRGB::Ptr cloud)
{
  // Downsampling using voxel grid
  pcl17::VoxelGrid<PointRGB> vg;
  PointCloudRGB::Ptr cloud_downsampled_ptr(new PointCloudRGB);
  vg.setLeafSize(downsample_voxel_size_,
                  downsample_voxel_size_,
                  downsample_voxel_size_);
  vg.setDownsampleAllData(true);
  vg.setInputCloud(cloud);
  vg.filter(*cloud_downsampled_ptr);

  return cloud_downsampled_ptr;
}

/** \brief Downsample the point cloud
  * \param cloud point cloud to be downsampled
  */
PointCloudRGB::Ptr pairwise_incremental_registration::PairwiseIncrementalRegistration::
outlierRemovalPointCloud(PointCloudRGB::Ptr cloud)
{
  // Statistical outlier removal
  PointCloudRGB::Ptr cloud_filtered_ptr(new PointCloudRGB);
  pcl17::StatisticalOutlierRemoval<PointRGB> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(50);
  sor.setStddevMulThresh(1.0);
  sor.filter(*cloud_filtered_ptr);

  return cloud_filtered_ptr;
}
