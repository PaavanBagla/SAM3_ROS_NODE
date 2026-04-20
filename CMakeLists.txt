cmake_minimum_required(VERSION 3.8)
project(sam3_ros)

find_package(ament_cmake_auto REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_auto_find_build_dependencies()

ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/${PROJECT_NAME})

install(
  PROGRAMS
    scripts/segmentation_node
    src/sam3_ros/pub_test_image.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package()