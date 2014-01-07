#include <pcl17/filters/voxel_grid.h>
#include <pcl17/filters/passthrough.h>
#include <pcl17/filters/extract_indices.h>
#include <pcl17/filters/statistical_outlier_removal.h>
#include <pcl17/point_cloud.h>
#include <pcl17/correspondence.h>
#include <pcl17/features/normal_3d_omp.h>
#include <pcl17/features/normal_3d.h>
#include <pcl17/features/shot_omp.h>
#include <pcl17/features/board.h>
#include <pcl17/keypoints/uniform_sampling.h>
#include <pcl17/recognition/cg/hough_3d.h>
#include <pcl17/recognition/cg/geometric_consistency.h>

typedef pcl17::PointXYZ             Point;
typedef pcl17::PointCloud<Point>    PointCloud;
typedef pcl17::PointXYZRGB          PointRGB;
typedef pcl17::PointXYZRGBA 				PointRGBA;
typedef pcl17::PointCloud<PointRGB> PointCloudRGB;
typedef pcl17::Normal 							NormalType;
typedef pcl17::ReferenceFrame 			RFType;
typedef pcl17::SHOT352 							DescriptorType;

namespace pointcloud_registration{
class PointCloudMatcher
{
public:
	PointCloudMatcher();
	~PointCloudMatcher();
protected:
	pointCloudCb(const PointCloudRGB::ConstPtr& point_cloud);
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

  // 
  bool show_keypoints_;
	bool show_correspondences_;
	bool use_cloud_resolution_;
	bool use_hough_;
	float sampling_radius_;
	float ref_frame_rad_;
	float descr_rad_;
	float cluster_size_;
	float cluster_thresh_;

  // Point clouds
  PointCloudRGB pointcloud_current_;
  PointCloudRGB pointcloud_previous_;
  pcl::PointCloud<DescriptorType> keypoints_current_;
  pcl::PointCloud<DescriptorType> keypoints_previous_;
  PointCloudRGB pointcloud_transformed_;
  PointCloudRGB pointcloud_merged_;
  Eigen::Matrix4f final_transformation_;

};
}

/** 
  * \brief Constructor: read node parameters and init the variables
  */
pointcloud_registration::PointCloudMatcher::
PointCloudMatcher() : nh_private_("~")
{
	// Read the parameters from the parameter server (set defaults)
  nh_private_.param("publish_merged_pointcloud_topic", merged_pointcloud_topic_, std::string("/output"));
  nh_private_.param("subscribe_pointcloud_topic", subscribe_pointcloud_topic_, std::string("/input"));
  nh_private_.param("input_cue", input_cue_, 10);
 	
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

  ROS_INFO_STREAM("[PointCloudMatcher:] pointcloud_registration node is up and running.");
}

/** 
  * \brief Destructor
  */
pointcloud_registration::PointCloudMatcher::
~PointCloudMatcher()
{
  ROS_INFO_STREAM("[PointCloudMatcher:] Shutting down pointcloud_registration node!");
}

/** \brief Point cloud callback
  * \param point_cloud the input point cloud received
  */
void pointcloud_registration::PointCloudMatcher::
pointCloudCb(const PointCloudRGB::ConstPtr& point_cloud)
{
  counter_++;

  if(first_cloud_received_ == false)
  {
    first_cloud_received_ = true;
    pointcloud_previous_ = *point_cloud;
    extractKeypoints(*pointcloud_previous_, *keypoints_previous_);

    ROS_INFO_STREAM("[PointCloudMatcher:] Received first point cloud with " 
      << pointcloud_current_.points.size() << " points.");

    // Initialize the point cloud merged
    pointcloud_merged_ = pointcloud_previous_;
  }
  else
  {
    pointcloud_current_ = *point_cloud;
    ROS_INFO_STREAM("[PointCloudMatcher:] Received point cloud number: " 
      << counter_ << " with " << pointcloud_current_.points.size() << " points.");

    extractKeypoints(*pointcloud_current_, *keypoints_current_);
    matchKeypoints(*keypoints_current_,*keypoints_previous_);


    pointcloud_previous_ = pointcloud_current_;
  } 

  // Publish point cloud
  point_cloud_merged_pub_.publish(pointcloud_merged_);
  ROS_INFO_STREAM("[PointCloudMatcher:] Merged point cloud published with " 
    << pointcloud_merged_.size() << " points.");
}

void pointcloud_registration::PointCloudMatcher::
extractKeypoints(const PointCloudRGB::Ptr cloud, const pcl::PointCloud<DescriptorType>::Ptr descriptors)
{
  //  Compute Normals
  pcl::NormalEstimationOMP<PointRGBA, NormalType> norm_est;
  norm_est.setKSearch (10);
  norm_est.setInputCloud (cloud);
  norm_est.compute (*normals);

  //  Downsample Clouds to Extract keypoints
  pcl::PointCloud<int> sampled_indices;

  pcl::UniformSampling<PointRGBA> uniform_sampling;
  uniform_sampling.setInputCloud (cloud);
  uniform_sampling.setRadiusSearch (sampling_radius_);
  uniform_sampling.compute (sampled_indices);
  pcl::copyPointCloud (*cloud, sampled_indices.points, *keypoints);
  std::cout << "Model total points: " << cloud->size() << "; Selected Keypoints: " << keypoints->size() << std::endl;

  //  Compute Descriptor for keypoints
  pcl::SHOTEstimationOMP<PointRGBA, NormalType, DescriptorType> descr_est;
  descr_est.setRadiusSearch (descr_rad_);

  descr_est.setInputCloud (keypoints);
  descr_est.setInputNormals (normals);
  descr_est.setSearchSurface (cloud);
  descr_est.compute (*descriptors);

}

void pointcloud_registration::PointCloudMatcher::
matchKeypoints(const pcl::PointCloud<DescriptorType>::Ptr kp1, const pcl::PointCloud<DescriptorType>::Ptr kp2)
{
  //  Find Model-Scene Correspondences with KdTree
  pcl::CorrespondencesPtr correspondences (new pcl::Correspondences ());

  pcl::KdTreeFLANN<DescriptorType> match_search;
  match_search.setInputCloud(kp1);

  // For each scene keypoint descriptor, find nearest neighbor into 
  // the model keypoints descriptor cloud and add it to the correspondences vector.
  for (size_t i = 0; i < kp2->size (); ++i)
  {
    std::vector<int> neigh_indices (1);
    std::vector<float> neigh_sqr_dists (1);
    if (!pcl_isfinite (kp2->at (i).descriptor[0])) //skipping NaNs
    {
      continue;
    }
    int found_neighs = match_search.nearestKSearch(kp2->at (i), 1, neigh_indices, neigh_sqr_dists);
    // add match only if the squared descriptor distance is 
    // less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
    if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) 
    {
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      correspondences->push_back (corr);
    }
  }
  std::cout << "Correspondences found: " << correspondences->size () << std::endl;

}


void a(){
	

  //
  //  Actual Clustering
  //
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
  std::vector<pcl::Correspondences> clustered_corrs;

  //  Using Hough3D
  if (use_hough_)
  {
    //
    //  Compute (Keypoints) Reference Frames only for Hough
    //
    pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
    pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch (rf_rad_);

    rf_est.setInputCloud (model_keypoints);
    rf_est.setInputNormals (model_normals);
    rf_est.setSearchSurface (model);
    rf_est.compute (*model_rf);

    rf_est.setInputCloud (scene_keypoints);
    rf_est.setInputNormals (scene_normals);
    rf_est.setSearchSurface (scene);
    rf_est.compute (*scene_rf);

    //  Clustering
    pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
    clusterer.setHoughBinSize (cg_size_);
    clusterer.setHoughThreshold (cg_thresh_);
    clusterer.setUseInterpolation (true);
    clusterer.setUseDistanceWeight (false);

    clusterer.setInputCloud (model_keypoints);
    clusterer.setInputRf (model_rf);
    clusterer.setSceneCloud (scene_keypoints);
    clusterer.setSceneRf (scene_rf);
    clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //clusterer.cluster (clustered_corrs);
    clusterer.recognize (rototranslations, clustered_corrs);
  }
  else // Using GeometricConsistency
  {
    pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
    gc_clusterer.setGCSize (cg_size_);
    gc_clusterer.setGCThreshold (cg_thresh_);

    gc_clusterer.setInputCloud (model_keypoints);
    gc_clusterer.setSceneCloud (scene_keypoints);
    gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //gc_clusterer.cluster (clustered_corrs);
    gc_clusterer.recognize (rototranslations, clustered_corrs);
  }

  //
  //  Output results
  //
  std::cout << "Model instances found: " << rototranslations.size () << std::endl;
  for (size_t i = 0; i < rototranslations.size (); ++i)
  {
    std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
    std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

    // Print the rotation matrix and translation vector
    Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
    Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

    printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
    printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
    printf ("\n");
    printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
  }

  //
  //  Visualization
  //
  pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
  viewer.addPointCloud (scene, "scene_cloud");

  pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

  if (show_correspondences_ || show_keypoints_)
  {
    //  We are translating the model so that it doesn't end in the middle of the scene representation
    pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));

    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
    viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
  }

  if (show_keypoints_)
  {
    pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
    viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
    viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
  }

  for (size_t i = 0; i < rototranslations.size (); ++i)
  {
    pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
    pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);

    std::stringstream ss_cloud;
    ss_cloud << "instance" << i;

    pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
    viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());

    if (show_correspondences_)
    {
      for (size_t j = 0; j < clustered_corrs[i].size (); ++j)
      {
        std::stringstream ss_line;
        ss_line << "correspondence_line" << i << "_" << j;
        PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);
        PointType& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);

        //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
        viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
      }
    }
  }

}