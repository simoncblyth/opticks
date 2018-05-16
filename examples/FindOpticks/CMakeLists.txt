cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(name FindOpticks)
project(${name})

find_package(GLM REQUIRED)    ## hmm not very scalable having to find all deps of your deps ?
find_package(Opticks REQUIRED)  # CMake to Opticks config with : cmake -DOpticks_DIR=/usr/local/opticks/config  

message(STATUS "Opticks_DIR          : ${Opticks_DIR} " )
message(STATUS "Opticks_VERSION      : ${Opticks_VERSION} " )
message(STATUS "Opticks_INCLUDE_DIRS : ${Opticks_INCLUDE_DIRS} " )
message(STATUS "Opticks_LIBRARIES    : ${Opticks_LIBRARIES} " )
message(STATUS "Opticks_LIBDIR       : ${Opticks_LIBDIR} " )
message(STATUS "Opticks_INCDIR       : ${Opticks_INCDIR} " )


SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  # https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling

add_executable(${name} ${name}.cc)

include_directories(
 
  ${Opticks_INCDIR}/SysRap 
  ${Opticks_INCDIR}/NPY

  ${GLM_INCLUDE_DIRS} 
)

link_directories( ${Opticks_LIBDIR} )   # as Opticks_LIBRARIES just lists names not paths

target_link_libraries(${name} ${Opticks_LIBRARIES} )

install(TARGETS ${name} DESTINATION lib)



