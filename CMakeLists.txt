cmake_minimum_required(VERSION 3.8)
project(code_utils)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
)

include_directories("include")

add_library(polynomial STATIC
  src/math_utils/Polynomial.cpp  
)

add_library(pnp
    src/cv_utils/dlt/dlt.cpp
    src/cv_utils/pnp/pnp.cpp
    src/cv_utils/pnp/linearpnp.cpp
    src/cv_utils/pnp/nonlinearpnp.cpp
)

target_link_libraries(
  pnp
  ${OpenCV_LIBRARIES}  
  ${CERES_LIBRARIES}
)

add_library(
  cv_utils STATIC
  src/cv_utils.cc
)

target_link_libraries(
  cv_utils
  ${OpenCV_LIBRARIES}  
)

add_executable(
  matIO_test 
  src/mat_io_test.cpp
)

target_link_libraries(
  matIO_test
  dw
  ${OpenCV_LIBRARIES}  
)

add_executable(
  sumpixel_test
  src/sumpixel_test.cpp
)

target_link_libraries(
  sumpixel_test
  dw
  ${OpenCV_LIBRARIES}
)

install(
  TARGETS
  polynomial
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS
  pnp
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin  
)

install(
  TARGETS
  cv_utils
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin  
)

install(
  TARGETS
  matIO_test
  DESTINATION lib/matIO_test
)

install(
  TARGETS
  sumpixel_test
  DESTINATION lib/sumpixel_test
)

install(DIRECTORY include/
  DESTINATION include/)



ament_package()
