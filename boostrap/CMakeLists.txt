cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name BoostRap)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

set(Boost_USE_STATIC_LIBS 1)

find_package(OpticksBoost REQUIRED)

set(LIBRARIES
   ${OpticksBoost_LIBRARIES}
)


if(UNIX) 
   set(LIBRARIES ${LIBRARIES} ssl crypto)
endif(UNIX)

#message("${name}:LIBRARIES : ${LIBRARIES} ")

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CMAKE_CURRENT_BINARY_DIR}
   ${OpticksBoost_INCLUDE_DIRS} 
)
add_definitions(
   ${OpticksBoost_DEFINITIONS} 
)


# BLog,fsutil,BSys depend on each other and boost 
# BCfg depends only on boost


set(SOURCES
  BDemo.cc 
  BTime.cc
  BLog.cc
  fsutil.cc
  BSys.cc
  BCfg.cc
  BDir.cc
  BTree.cc
  BJson.cc
  BMap.cc
  BList.cc

  regexsearch.cc
  stringutil.cc
  
  md5.cc
  md5digest.cc
) 

set(PROBLEM_SOURCES
  timeutil.cc
)



set(HEADERS
BDemo.hh 
BTime.hh
bffs.hh
BCfg.hh
BLog.hh
BSys.hh
BDir.hh
BJson.hh
BMap.hh
BList.hh

fsutil.hh
md5digest.hh
regexsearch.hh
stringutil.hh
timeutil.hh
dbg.hh
BRAP_API_EXPORT.hh
)

# appends to flags for specific files : trying to quell warnings from underlying boost code
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
   set_source_files_properties(BLog.cc      PROPERTIES COMPILE_FLAGS -Wno-sign-compare)
   set_source_files_properties(jsonutil.cc  PROPERTIES COMPILE_FLAGS -Wno-all)
endif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")


add_library( ${name}  SHARED ${SOURCES})

target_link_libraries( ${name}  ${LIBRARIES} )


install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})


#add_subdirectory(tests EXCLUDE_FROM_ALL)
#add_subdirectory(tests)


