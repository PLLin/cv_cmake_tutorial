# CMake 最小指定版本要求為3.10.2
cmake_minimum_required(VERSION 3.10.2)
# 設定 Project 名稱
project(cpphw1)
# 設定 C++ 版本為 11
set(CMAKE_CXX_STANDARD 11)

## Cmake build 使用額外套件的執行檔
# 設定尋找 header file 的路徑
include_directories(include/)
# 確認是否有裝 Boost library
find_package(OpenCV REQUIRED)
# 設定尋找Boost header file 的路徑
include_directories(${OpenCV_INCLUDE_DIRS})

# 生成執行檔
add_executable(main_train main_train.cpp ${PROJECT_SOURCE_DIR}/src/knn_train.cpp)
# Link Library
target_link_libraries(main_train ${OpenCV_LIBS})

# 生成執行檔
add_executable(main_test main_test.cpp ${PROJECT_SOURCE_DIR}/src/knn_test.cpp)
# Link Library
target_link_libraries(main_test ${OpenCV_LIBS})