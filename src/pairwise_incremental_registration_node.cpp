#include <ros/ros.h>
#include "pairwise_incremental_registration.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "point_cloud_registration");
  pairwise_incremental_registration::PairwiseIncrementalRegistration pairwise_incremental_registration;
  ros::spin();
  return 0;
}