cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name BoostRap)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

if(UNIX AND NOT APPLE) 
   set(Boost_USE_STATIC_LIBS 1)
endif(UNIX AND NOT APPLE)

find_package(OpticksBoost REQUIRED)
find_package(PLog REQUIRED)
find_package(SysRap REQUIRED)

set(LIBRARIES
   ${OpticksBoost_LIBRARIES}
   ${SysRap_LIBRARIES}
   ${PLog_LIBRARIES}
)


if(UNIX) 
   set(LIBRARIES ${LIBRARIES} ssl crypto)
endif(UNIX)

#message("${name}:LIBRARIES : ${LIBRARIES} ")
#message("${name}:PLog_INCLUDE_DIRS : ${PLog_INCLUDE_DIRS} ")

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CMAKE_CURRENT_BINARY_DIR}
   ${CMAKE_CURRENT_BINARY_DIR}/inc

   ${OpticksBoost_INCLUDE_DIRS} 
   ${SysRap_INCLUDE_DIRS} 
   ${PLog_INCLUDE_DIRS} 
)


add_definitions(
   ${OpticksBoost_DEFINITIONS} 
   ${SysRap_DEFINITIONS} 
   ${PLog_DEFINITIONS} 
)

if(UNIX AND NOT APPLE) 
add_definitions(
   -DBOOST_LOG_DYN_LINK=1
)
endif(UNIX AND NOT APPLE)


# generate define header for hardcoding in the library 
set(OPTICKS_INSTALL_PREFIX     "${CMAKE_INSTALL_PREFIX}")
configure_file( BOpticksResourceCMakeConfig.hh.in inc/BOpticksResourceCMakeConfig.hh )



set(SOURCES
  BRAP_LOG.cc 

  BBufSpec.cc 
  BDemo.cc 
  BFoo.cc 
  BTime.cc
  BTimer.cc
  BCfg.cc
  BDir.cc
  BMap.cc
  Map.cc
  BList.cc
  BStr.cc
  BHex.cc
  BBit.cc
  BFile.cc
  BRegex.cc
  BTree.cc
  BEnv.cc
  BOpticksResource.cc
  BDynamicDefine.cc
  BOpticksEvent.cc
  BConfig.cc
  BBnd.cc
  BRng.cc

) 

set(HEADERS
  BRAP_API_EXPORT.hh
  BRAP_LOG.hh
  BRAP_HEAD.hh
  BRAP_TAIL.hh
  BRAP_BODY.hh

  BDirect.hh 

  BBufSpec.hh 
  BDemo.hh 
  BTime.hh
  BTimer.hh
  BCfg.hh
  BDir.hh
  BMap.hh
  Map.hh
  BList.hh
  BStr.hh
  BHex.hh
  BBit.hh
  BFile.hh
  BRegex.hh
  BEnv.hh
  BOpticksResource.hh
  BDynamicDefine.hh
  BOpticksEvent.hh
  BConfig.hh

)


set(TOXIC_SOURCES
  BLog.cc
  timeutil.cc
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


