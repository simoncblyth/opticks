cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name BoostRap)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(OpticksBoost REQUIRED)

set(LIBRARIES
   ${OpticksBoost_LIBRARIES}
)
include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${OpticksBoost_INCLUDE_DIRS} 
)
add_definitions(
   ${OpticksBoost_DEFINITIONS} 
)


set(SOURCES
   regexsearch.cc
   fsutil.cc 
   BLog.cc
   BCfg.cc
   BSys.cc
) 

set(HEADERS
   regexsearch.hh
   fsutil.hh 
   dbg.hh
   BLog.hh
   BCfg.hh
   BSys.hh
)


if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
   set_source_files_properties(BLog.cc      PROPERTIES COMPILE_FLAGS -Wno-sign-compare)
endif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")



add_library( ${name}  SHARED ${SOURCES})

target_link_libraries( ${name}  ${LIBRARIES} )


install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})


#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)


