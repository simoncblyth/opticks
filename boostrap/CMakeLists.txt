cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name BoostRap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

#set(Boost_DEBUG ON)
#set(Boost_NO_BOOST_CMAKE ON)
set(COMPONENTS system program_options filesystem regex )
find_package(Boost REQUIRED COMPONENTS ${COMPONENTS} )
set_target_properties(Boost::system  PROPERTIES INTERFACE_FIND_PACKAGE_NAME "Boost REQUIRED COMPONENTS ${COMPONENTS}")

## Above target_properties kludge tees up arguments to find_dependency in BCM generated export 
## useboost-config.cmake so downstream targets will automatically do the required find_dependency, 
## see examples/UseBoost examples/UseUseBoost for development of this kludge
##
## NB INTERFACE_FIND_PACKAGE_NAME is a BCM defined property, not a standard one, see bcm-

find_package(PLog REQUIRED MODULE)     # for integrated-building ?
find_package(SysRap REQUIRED CONFIG)


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
  BPath.cc
  BRegex.cc
  BTree.cc
  BEnv.cc
  BResource.cc
  BOpticksResource.cc
  BDynamicDefine.cc
  BOpticksEvent.cc
  BConfig.cc
  BBnd.cc
  BRng.cc
  BConverter.cc
  BLocSeq.cc
  BLocSeqDigest.cc
) 

set(HEADERS
  BRAP_API_EXPORT.hh
  BRAP_LOG.hh
  BRAP_HEAD.hh
  BRAP_TAIL.hh
  BRAP_BODY.hh

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
  BRegex.hpp
  BEnv.hh
  BResource.hh
  BOpticksResource.hh
  BDynamicDefine.hh
  BOpticksEvent.hh
  BConfig.hh
  BBnd.hh
  BRng.hh
  BConverter.hh
  BLocSeq.hh
  BLocSeqDigest.hh
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

target_include_directories( ${name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/inc>
)
target_compile_definitions( ${name} PUBLIC OPTICKS_BRAP )


target_link_libraries( ${name} PUBLIC 
   Opticks::PLog
   Opticks::SysRap 
   Boost::system
   Boost::program_options
   Boost::filesystem
   Boost::regex
)
if(UNIX)
target_link_libraries( ${name} PUBLIC ssl crypto )
endif()

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)


#[=[
Have been getting this warning for years::

    ld: warning: direct access in function 'boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::bad_function_call> >::rethrow() const' from file 'CMakeFiles/BoostRap.dir/BStr.cc.o' to global weak symbol 'typeinfo for boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::bad_function_call> >' from file 'CMakeFiles/BoostRap.dir/BCfg.cc.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    [ 34%] Built target BoostRap

Avoided this warning by adding to OpticksCompilationFlags.cmake::

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden") ## avoid boostrap visibility warning at link

::
#]=]


