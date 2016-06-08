cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name BoostRap)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(OpticksBoost REQUIRED)

set(LIBRARIES
   ${OpticksBoost_LIBRARIES}

   ssl
   crypto
)
include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${OpticksBoost_INCLUDE_DIRS} 
)
add_definitions(
   ${OpticksBoost_DEFINITIONS} 
)

set(SOURCES

BCfg.cc
BLog.cc
BSys.cc
dirutil.cc
fsutil.cc
jsonutil.cc
md5digest.cc
regexsearch.cc
stringutil.cc
timeutil.cc

) 

set(HEADERS

BCfg.hh
BLog.hh
BSys.hh
dirutil.hh
fsutil.hh
jsonutil.hh
md5digest.hh
regexsearch.hh
stringutil.hh
timeutil.hh

dbg.hh
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
add_subdirectory(tests)


