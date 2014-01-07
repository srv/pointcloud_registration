#include <ros/ros.h>
#include "icp_registration.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "point_cloud_registration");
  icp_registration::IcpRegistration icp_registration;
  ros::spin();
  return 0;
}