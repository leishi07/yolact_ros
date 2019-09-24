cmake_minimum_required(VERSION 2.8.3)
project(yolact_ros)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
   FILES
   Object.msg
   yolact_objects.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs sensor_msgs
)

catkin_package(
   CATKIN_DEPENDS message_runtime cv_bridge geometry_msgs roscpp rospy std_msgs sensor_msgs 
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
