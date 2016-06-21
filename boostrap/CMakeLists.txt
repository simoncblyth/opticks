cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name BoostRap)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

set(Boost_USE_STATIC_LIBS 1)

find_package(OpticksBoost REQUIRED)
find_package(PLog REQUIRED)

set(LIBRARIES
   ${OpticksBoost_LIBRARIES}
   ${PLog_LIBRARIES}
)


if(UNIX) 
   set(LIBRARIES ${LIBRARIES} ssl crypto)
endif(UNIX)

#message("${name}:LIBRARIES : ${LIBRARIES} ")
message("${name}:PLog_INCLUDE_DIRS : ${PLog_INCLUDE_DIRS} ")

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CMAKE_CURRENT_BINARY_DIR}
   ${OpticksBoost_INCLUDE_DIRS} 
   ${PLog_INCLUDE_DIRS} 
)



add_definitions(
   ${OpticksBoost_DEFINITIONS} 
   ${PLog_DEFINITIONS} 
)


# BLog,BFile,BSys depend on each other and boost 
# BCfg depends only on boost


set(SOURCES
  BRAP_LOG.cc 
  PLOG.cc 

  BDemo.cc 
  BFoo.cc 
  BTime.cc
  BTimer.cc
  BSys.cc
  BCfg.cc
  BDir.cc
  BJson.cc
  BMap.cc
  BList.cc
  BStr.cc
  BHex.cc
  BBit.cc
  BFile.cc
  BRegex.cc
  BDigest.cc
  BArrayDigest.cc

  md5.cc
  BTree.cc
) 

set(TOXIC_SOURCES
  BLog.cc
  timeutil.cc
)

set(HEADERS
  BRAP_API_EXPORT.hh
  BRAP_LOG.hh
  PLOG.hh
  PLOG_INIT.hh  
  BDirect.hh 

  BDemo.hh 
  BTime.hh
  BTimer.hh
  BSys.hh
  BCfg.hh
  BDir.hh
  BJson.hh
  BMap.hh
  BList.hh
  BStr.hh
  BHex.hh
  BBit.hh
  BFile.hh
  BRegex.hh
  BDigest.hh
  BArrayDigest.hh
)

set(TOXIC_HEADERS
  timeutil.hh
  BLog.hh
)



# appends to flags for specific files : trying to quell warnings from underlying boost code
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
   set_source_files_properties(BLog.cc      PROPERTIES COMPILE_FLAGS -Wno-sign-compare)
   set_source_files_properties(BMap.cc  PROPERTIES COMPILE_FLAGS -Wno-all)
   set_source_files_properties(BList.cc  PROPERTIES COMPILE_FLAGS -Wno-all)
endif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")


add_library( ${name}  SHARED ${SOURCES})

target_link_libraries( ${name}  ${LIBRARIES} )


install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})


#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)


