cmake_minimum_required(VERSION 3.8)
project(semantic_bki)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message("OpenMP enabled")
endif()

find_package(OpenCV REQUIRED)

## Enable OpenMP
add_definitions(-DOPENMP)

## Find ament packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
if(yaml-cpp_FOUND)
  message(STATUS "Found yaml-cpp")
endif()
find_package(PCL REQUIRED)

###########
## Build ##
###########

include_directories(
  include/semantic_bki/common
  include/semantic_bki/mapping
  include/semantic_bki/dataset_devkit
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# Common source files
set(COMMON_SOURCE_FILES src/common/point3f.cpp)

# Files for SEMANTIC BGK source
set(MAPPING_SOURCE_FILES
    src/mapping/bkiblock.cpp
    src/mapping/bkioctomap.cpp
    src/mapping/bkioctree_node.cpp
    src/mapping/bkioctree.cpp)

add_executable(toy_example_node
  ${MAPPING_SOURCE_FILES}
  ${COMMON_SOURCE_FILES}
  src/toy_example_node.cpp
)

add_executable(kitti_node
  ${MAPPING_SOURCE_FILES}
  ${COMMON_SOURCE_FILES}
  src/kitti_node.cpp
)

add_executable(semantickitti_node
  ${MAPPING_SOURCE_FILES}
  ${COMMON_SOURCE_FILES}
  src/semantickitti_node.cpp
)

add_executable(mcd_node
  ${MAPPING_SOURCE_FILES}
  ${COMMON_SOURCE_FILES}
  src/common/osm_visualizer.cpp
  src/mcd_node.cpp
)

ament_target_dependencies(toy_example_node
  rclcpp
  geometry_msgs
  visualization_msgs
  std_msgs
  pcl_conversions
)

ament_target_dependencies(kitti_node
  rclcpp
  geometry_msgs
  visualization_msgs
  std_msgs
  sensor_msgs
  pcl_conversions
  OpenCV
)

ament_target_dependencies(semantickitti_node
  rclcpp
  geometry_msgs
  visualization_msgs
  std_msgs
  sensor_msgs
  pcl_conversions
  OpenCV
)

ament_target_dependencies(mcd_node
  rclcpp
  geometry_msgs
  visualization_msgs
  std_msgs
  sensor_msgs
  nav_msgs
  pcl_conversions
  tf2_eigen
  tf2_ros
  Eigen3
  ament_index_cpp
  OpenCV
  yaml-cpp
)

target_link_libraries(toy_example_node
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

target_link_libraries(kitti_node
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

target_link_libraries(semantickitti_node
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

target_link_libraries(mcd_node
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
)

#############
## Install ##
#############

install(TARGETS
  toy_example_node
  kitti_node
  semantickitti_node
  mcd_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY data/
  DESTINATION share/${PROJECT_NAME}/data
)

install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
