#include <ros/ros.h>
#include "icp_registration.h"
#include "pairwise_incremental_registration.h"
#include "normal_distribution_registration.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "point_cloud_registration");
  //icp_registration::IcpRegistration icp_registration;
  //pairwise_incremental_registration::PairwiseIncrementalRegistration pairwise_incremental_registration;
  normal_distribution_registration::NormalDistributionRegistration normal_distribution_registration;
  ros::spin();
  return 0;
}