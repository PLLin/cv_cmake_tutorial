# CMake 最小指定版本要求為3.10.2
cmake_minimum_required(VERSION 3.10.2)
# 設定 Project 名稱
project(tutorialcvcmake)
# 設定 C++ 版本為 11
set(CMAKE_CXX_STANDARD 11)

## Cmake Helloworld
# 設定尋找 header file 的路徑
include_directories(include/)
# 生成執行檔
add_executable(main_unique_ptr ${PROJECT_SOURCE_DIR}/main_unique_ptr.cpp src/calculator.cpp)
# 生成執行檔
add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp src/calculator.cpp)

# ## Cmake build 使用額外套件的執行檔
# # 設定尋找 header file 的路徑
# include_directories(include/)
# # 確認是否有裝 Boost library
# find_package(Boost REQUIRED)
# # 設定尋找Boost header file 的路徑
# include_directories(${Boost_INCLUDE_DIR})
# # 生成執行檔
# add_executable(main_string ${PROJECT_SOURCE_DIR}/main_string.cpp src/string_calculator.cpp)
# # Link Library
# target_link_libraries(main_string ${Boost_LIBRARIES})

# ## Cmake build libary
# # 設定尋找 header file 的路徑
# include_directories(include/)
# # 設定一個 variable 為該目錄下的所有檔案
# aux_source_directory(./src DIR_SRCS)

# # 生成 static library
# add_library(${PROJECT_NAME} ${DIR_SRCS})
# # 使用生成的 static library
# add_executable(main_static ${PROJECT_SOURCE_DIR}/main.cpp)
# target_link_libraries(main_static tutorialcvcmake)

# # # 生成 shared library
# # add_library(${PROJECT_NAME} SHARED ${DIR_SRCS})
# # # 使用生成的 shared library
# # add_executable(main_shared ${PROJECT_SOURCE_DIR}/main.cpp)
# # target_link_libraries(main tutorialcvcmake)
