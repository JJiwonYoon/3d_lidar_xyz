cmake_minimum_required(VERSION 3.8)
project(lidar_xyz)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io kdtree)
find_package(visualization_msgs REQUIRED)

# Include directories
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${rclcpp_INCLUDE_DIRS})
include_directories(${sensor_msgs_INCLUDE_DIRS})
include_directories(${nav_msgs_INCLUDE_DIRS})
include_directories(${pcl_conversions_INCLUDE_DIRS})
add_executable(main_node src/3d_xyz.cpp)
ament_target_dependencies(main_node rclcpp sensor_msgs pcl_conversions visualization_msgs nav_msgs)
target_link_libraries(main_node ${PCL_LIBRARIES} ${rclcpp_LIBRARIES} ${sensor_msgs_LIBRARIES} ${pcl_conversions_LIBRARIES} ${nav_msgs_LIBRARIES})
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
install(TARGETS main_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
