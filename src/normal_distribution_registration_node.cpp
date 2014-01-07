#include <ros/ros.h>
#include "normal_distribution_registration.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "point_cloud_registration");
  normal_distribution_registration::NormalDistributionRegistration normal_distribution_registration;
  ros::spin();
  return 0;
}