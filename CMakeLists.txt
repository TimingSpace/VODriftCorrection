cmake_minimum_required(VERSION 2.8)
project(vo)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3  ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 ")
option (G2O_USE_CHOLMOD "Build g2o with CHOLMOD support" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
find_package(Cholmod)
find_package(CSparse)
find_package(BLAS)
find_package(LAPACK)
find_package(G2O)

if(G2O_USE_CHOLMOD AND CHOLMOD_FOUND AND BLAS_FOUND AND LAPACK_FOUND)
  message(STATUS "Found CHOLMOD and its dependencies")
  set(CHOLMOD_FOUND TRUE)
else()
  set(CHOLMOD_FOUND FALSE)
endif()
if( G2O_FOUND)
  message(STATUS "Found G2O and its dependencies")
  message(STATUS ${G2O_CORE_LIBRARY})
  message(STATUS ${CHOLMOD_LIBRARY})
else()
    message(STATUS "NOT Found G2O and its dependencies")
endif()



# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


# set your eigin include directory 
set(EIGEN3_INCLUDE_DIR /usr/local/include/eigen3)
include_directories(
    ${OpenCV_INCLUDE_DIR}
${EIGEN3_INCLUDE_DIR}
${CSPARSE_INCLUDE_DIR}
${CHOLMOD_INCLUDE_DIR}
${G2O_INCLUDE_DIR}
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/src
)
add_subdirectory(pybind11)

pybind11_add_module(vocorrection
src/vo_correct.cpp
src/edge/edge_unary_pointxyz.cpp
)
target_link_libraries(vocorrection 
    ${EIGEN3_LIBS}
    ${G2O_CORE_LIBRARY}
    ${G2O_SOLVER_CHOLMOD}
    ${G2O_SOLVER_CSPARSE}
    ${G2O_CLI_LIBRARY}
    ${G2O_TYPES_SBA}
    ${G2O_TYPES_SLAM3D}
    ${G2O_STUFF_LIBRARY}
    ${G2O_SOLVER_CSPARSE_EXTENSION}
    ${CSPARSE_LIBRARY}
    ${CHOLMOD_LIBRARY}
    )

