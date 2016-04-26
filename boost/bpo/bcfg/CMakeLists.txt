cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name BCfg)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(OpticksBoost REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${OpticksBoost_INCLUDE_DIRS} 
)
add_definitions(
   ${OpticksBoost_DEFINITIONS} 
)

add_library( ${name}  SHARED  Cfg.cc )

target_link_libraries( ${name} 
      ${OpticksBoost_LIBRARIES}
)


SET(hdrs  Cfg.hh)

install(TARGETS ${name}     DESTINATION lib)
install(FILES ${hdrs} DESTINATION include/${name})


