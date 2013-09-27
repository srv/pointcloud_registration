#include "pointcloud_registration.h"
#include <pcl17/filters/voxel_grid.h>
#include <pcl17/filters/passthrough.h>
#include <pcl17/filters/extract_indices.h>
#include <pcl17/filters/statistical_outlier_removal.h>
#include <pcl17/features/normal_3d.h>
#include <pcl17/registration/transforms.h>
#include <pcl17/registration/ndt.h>

/** 
  * \brief Constructor: read node parameters and init the variables
  */
pointcloud_registration::PointCloudRegistration::
PointCloudRegistration() : nh_private_("~")
{
  // Read the parameters from the parameter server (set defaults)
  nh_private_.param("publish_merged_pointcloud_topic", merged_pointcloud_topic_, std::string("/merged_pointcloud"));
  nh_private_.param("subscribe_pointcloud_topic", subscribe_pointcloud_topic_, std::string("/points2"));
  nh_private_.param("input_cue", input_cue_, 10);
  nh_private_.param("input_filtering", input_filtering_, true);
  nh_private_.param("output_filtering", output_filtering_, true);
  nh_private_.param("in_filter_x_min", in_filter_x_min_, -10.0);
  nh_private_.param("in_filter_x_max", in_filter_x_max_, 10.0);
  nh_private_.param("in_filter_y_min", in_filter_y_min_, -10.0);
  nh_private_.param("in_filter_y_max", in_filter_y_max_, 10.0);
  nh_private_.param("in_filter_z_min", in_filter_z_min_, -10.0);
  nh_private_.param("in_filter_z_max", in_filter_z_max_, 10.0);
  nh_private_.param("out_filter_x_min", out_filter_x_min_, -10.0);
  nh_private_.param("out_filter_x_max", out_filter_x_max_, 10.0);
  nh_private_.param("out_filter_y_min", out_filter_y_min_, -10.0);
  nh_private_.param("out_filter_y_max", out_filter_y_max_, 10.0);
  nh_private_.param("out_filter_z_min", out_filter_z_min_, -10.0);
  nh_private_.param("out_filter_z_max", out_filter_z_max_, 10.0);
  nh_private_.param("in_filter_voxel_size", in_filter_voxel_size_, 0.01);
  nh_private_.param("out_filter_voxel_size", out_filter_voxel_size_, 0.01);
  nh_private_.param("in_filter_mean_k", in_filter_mean_k_, 50);
  nh_private_.param("out_filter_mean_k", out_filter_mean_k_, 50);
  nh_private_.param("in_filter_std_dev_thresh", in_filter_std_dev_thresh_, 1.0);
  nh_private_.param("out_filter_std_dev_thresh", out_filter_std_dev_thresh_, 1.0);
  nh_private_.param("max_number_of_iterations", max_number_of_iterations_, 30);
  nh_private_.param("epsilon_transformation", epsilon_transformation_, 0.01);
  nh_private_.param("nt_step_size", nt_step_size_, 0.1);
  nh_private_.param("ndt_grid_resolution", ndt_grid_resolution_, 1.0);

  // Initialize parameters
  counter_ = 0;
  first_cloud_received_ = false;
  final_transformation_ = Eigen::Matrix4f::Identity();
  
  // Subscription to the point cloud result from stereo_image_proc
  point_cloud_sub_ = nh_.subscribe<PointCloudRGB>(
    subscribe_pointcloud_topic_,
    input_cue_,
    &PointCloudRegistration::pointCloudCb,
    this);

  // Declare the point cloud merget topic
  point_cloud_merged_pub_ = nh_private_.advertise<PointCloudRGB>(merged_pointcloud_topic_, 1); 

  ROS_INFO_STREAM("[PointCloudRegistration:] pointcloud_registration node is up and running.");
}

/** 
  * \brief Destructor
  */
pointcloud_registration::PointCloudRegistration::
~PointCloudRegistration()
{
  ROS_INFO_STREAM("[PointCloudRegistration:] Shutting down pointcloud_registration node!");
}

/** \brief Align a pair of PointCloud datasets and return the result
  * \param cloud_src the source PointCloud
  * \param cloud_tgt the target PointCloud
  * \param output the resultant aligned source PointCloud
  * \param final_transform the resultant transform between source and target
  */
Eigen::Matrix4f pointcloud_registration::PointCloudRegistration::
pairAlign(const PointCloudRGB::Ptr cloud_src, const PointCloudRGB::Ptr cloud_tgt)
{
  PointCloudRGB::Ptr src(new PointCloudRGB);
  PointCloudRGB::Ptr tgt(new PointCloudRGB);
  src = cloud_src;
  tgt = cloud_tgt;
  
  // Initializing Normal Distributions Transform (NDT).
  pcl17::NormalDistributionsTransform<PointRGB, PointRGB> ndt;

  // Setting scale dependent NDT parameters
  // Setting minimum transformation difference for termination condition.
  ndt.setTransformationEpsilon(epsilon_transformation_);
  // Setting maximum step size for More-Thuente line search.
  ndt.setStepSize(nt_step_size_);
  //Setting Resolution of NDT grid structure (VoxelGridCovariance).
  ndt.setResolution(ndt_grid_resolution_);

  // Setting max number of registration iterations.
  ndt.setMaximumIterations(max_number_of_iterations_);
  // Setting point cloud to be aligned.
  ndt.setInputSource(src);
  // Setting point cloud to be aligned to.
  ndt.setInputTarget(tgt);

  // Set initial alignment estimate found using robot odometry.
  //Eigen::AngleAxisf init_rotation (0.6931, Eigen::Vector3f::UnitZ ());
  //Eigen::Translation3f init_translation (1.79387, 0.720047, 0);
  //Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();

  // Calculating required rigid transform to align the input cloud to the target cloud.
  PointCloudRGB::Ptr output_cloud (new PointCloudRGB);
  //ndt.align (*output_cloud, init_guess);
  ndt.align (*output_cloud);

  ROS_INFO_STREAM("[PointCloudRegistration:] has converged with score of: " << ndt.getFitnessScore());

  return ndt.getFinalTransformation();
}

/** \brief Point cloud callback
  * \param point_cloud the input point cloud received
  */
void pointcloud_registration::PointCloudRegistration::
pointCloudCb(const PointCloudRGB::ConstPtr& point_cloud)
{
  PointCloudRGB::Ptr temp(new PointCloudRGB);
  counter_++;

  if(first_cloud_received_ == false)
  {
    first_cloud_received_ = true;
    pointcloud_current_ = *point_cloud;
    ROS_INFO_STREAM("[PointCloudRegistration:] Received first point cloud with " 
      << pointcloud_current_.points.size() << " points.");

    if(input_filtering_)
    {
      temp = pointcloudFilter(pointcloud_current_.makeShared(), 0);
      pointcloud_current_ = *temp;
    }

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

    if(input_filtering_)
    {
      temp = pointcloudFilter(pointcloud_current_.makeShared(), 0);
      pointcloud_current_ = *temp;
    }

    // Align point clouds
    Eigen::Matrix4f pair_transform;
    pair_transform = pairAlign(pointcloud_current_.makeShared(), pointcloud_previous_.makeShared());

    // Update the global transform
    final_transformation_ *= pair_transform;

    // Transform current pair into the global transform
    pcl17::transformPointCloud(pointcloud_current_, pointcloud_transformed_, final_transformation_);

    // Accumulate pointclouds
    pointcloud_merged_ += pointcloud_transformed_;

    // Update for next callback
    pointcloud_previous_ = pointcloud_current_;
  }

  // Filter the merged pointcloud
  if(output_filtering_)
  {
    temp = pointcloudFilter(pointcloud_merged_.makeShared(), 1);
    pointcloud_merged_ = *temp;
  }  

  // Publish point cloud
  point_cloud_merged_pub_.publish(pointcloud_merged_);
  ROS_INFO_STREAM("[PointCloudRegistration:] Merged point cloud published with " 
    << pointcloud_merged_.size() << " points.");
}

/** \brief Filter the point cloud using voxel grid and statistical
  * outlier removal.
  * \param cloud point cloud to be filtered
  */
PointCloudRGB::Ptr pointcloud_registration::PointCloudRegistration::
pointcloudFilter(PointCloudRGB::Ptr cloud, int type)
{
  // Filter parameters
  double filter_x_min;
  double filter_x_max;
  double filter_y_min;
  double filter_y_max;
  double filter_z_min;
  double filter_z_max;
  double filter_voxel_size;
  int filter_mean_k;
  double filter_std_dev_thresh;

  if (type == 0)
  {
    // Input filter
    filter_x_min = in_filter_x_min_;
    filter_x_max = in_filter_x_max_;
    filter_y_min = in_filter_y_min_;
    filter_y_max = in_filter_y_max_;
    filter_z_min = in_filter_z_min_;
    filter_z_max = in_filter_z_max_;
    filter_voxel_size = in_filter_voxel_size_;
    filter_mean_k = in_filter_mean_k_;
    filter_std_dev_thresh = in_filter_std_dev_thresh_;
  }
  else
  {
    // Output filter
    filter_x_min = out_filter_x_min_;
    filter_x_max = out_filter_x_max_;
    filter_y_min = out_filter_y_min_;
    filter_y_max = out_filter_y_max_;
    filter_z_min = out_filter_z_min_;
    filter_z_max = out_filter_z_max_;
    filter_voxel_size = out_filter_voxel_size_;
    filter_mean_k = out_filter_mean_k_;
    filter_std_dev_thresh = out_filter_std_dev_thresh_;
  }

  // Copy the point cloud
  PointCloudRGB::Ptr cloud_ptr(new PointCloudRGB);

  // NAN and limit filtering
  PointCloudRGB::Ptr cloud_filtered_ptr(new PointCloudRGB);
  pcl17::PassThrough<PointRGB> pass;

  // X-filtering
  pass.setFilterFieldName("x");
  pass.setFilterLimits(filter_x_min, filter_x_max);
  pass.setInputCloud(cloud);
  pass.filter(*cloud_filtered_ptr);

  // Y-filtering
  pass.setFilterFieldName("y");
  pass.setFilterLimits(filter_y_min, filter_y_max);
  pass.setInputCloud(cloud_filtered_ptr);
  pass.filter(*cloud_filtered_ptr);

  // Z-filtering
  pass.setFilterFieldName("z");
  pass.setFilterLimits(filter_z_min, filter_z_max);
  pass.setInputCloud(cloud_filtered_ptr);
  pass.filter(*cloud_filtered_ptr);

  // Downsampling using voxel grid  
  PointCloudRGB::Ptr cloud_downsampled_ptr(new PointCloudRGB);
  pcl17::VoxelGrid<PointRGB> grid;
  grid.setLeafSize(filter_voxel_size, filter_voxel_size, filter_voxel_size);
  grid.setDownsampleAllData(true);
  grid.setInputCloud(cloud_filtered_ptr);
  grid.filter(*cloud_downsampled_ptr);
  
  // Statistical outlier removal
  PointCloudRGB::Ptr cloud_outlier_ptr(new PointCloudRGB);
  pcl17::StatisticalOutlierRemoval<PointRGB> sor;
  sor.setInputCloud(cloud_downsampled_ptr);
  sor.setMeanK(filter_mean_k);
  sor.setStddevMulThresh(filter_std_dev_thresh);
  sor.filter(*cloud_outlier_ptr);  

  return cloud_outlier_ptr;
}
