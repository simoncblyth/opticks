cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name BoostRap)
project(${name})

include(GenerateExportHeader)
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

message("${name}:LIBRARIES : ${LIBRARIES} ")

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CMAKE_CURRENT_BINARY_DIR}
   ${OpticksBoost_INCLUDE_DIRS} 
)
add_definitions(
   ${OpticksBoost_DEFINITIONS} 
)


# BLog fsutil BSys depend on each other and boost 
# BCfg depends only on boost




set(SOURCES
  BLog.cc
  fsutil.cc
  BSys.cc
  BCfg.cc
  dirutil.cc
  jsonutil.cc

  regexsearch.cc
  stringutil.cc
  
  md5.c
  md5digest.cc

) 

set(SOURCES_FAILING

  timeutil.cc

)


set(HEADERS
)


set(HEADERS_ALL

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


set(EXPORT_HDR BRAP_API_EXPORT.h)
generate_export_header(${name}
                        EXPORT_MACRO_NAME BRAP_API
                         EXPORT_FILE_NAME ${EXPORT_HDR}
                      )

target_link_libraries( ${name}  ${LIBRARIES} )

set(HEADERS 
        ${HEADERS} 
        ${CMAKE_CURRENT_BINARY_DIR}/${EXPORT_HDR}
)



install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})


#add_subdirectory(tests EXCLUDE_FROM_ALL)
#add_subdirectory(tests)


