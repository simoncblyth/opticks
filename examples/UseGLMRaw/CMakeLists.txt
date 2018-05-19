cmake_minimum_required(VERSION 3.0)
set(name UseGLMRaw)
project(${name} VERSION 0.1.0)

#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

find_package(GLM REQUIRED)

set(HEADERS  ${name}.hh )

add_library(${name} SHARED ${name}.cc)

target_include_directories( ${name}  
    PUBLIC 
        $<INSTALL_INTERFACE:include/${name}>    
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries( ${name}   
    PUBLIC
        Opticks::GLM
)

install(TARGETS ${name}      
   EXPORT ${name}ConfigTargets
   LIBRARY DESTINATION lib
   RUNTIME DESTINATION lib        ## for windows
   INCLUDES DESTINATION include/${name}
)

install(EXPORT ${name}ConfigTargets
          FILE ${name}ConfigTargets.cmake
   NAMESPACE Opticks::
   DESTINATION lib/cmake/${name}
)

install(FILES ${HEADERS}   DESTINATION include/${name})


file(WRITE "${PROJECT_BINARY_DIR}/${name}Config.cmake" "
include(CMakeFindDependencyMacro)
find_dependency(GLM)
include(\"\${CMAKE_CURRENT_LIST_DIR}/${name}ConfigTargets.cmake\")
")


set(TARGET_VERSION ${PROJECT_VERSION})

include(WriteBasicConfigVersionFile)
write_basic_config_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/${name}ConfigVersion.cmake
        VERSION ${TARGET_VERSION}
        COMPATIBILITY SameMajorVersion
)


#include(CMakePackageConfigHelpers)
#write_basic_package_version_file("${PROJECT_BINARY_DIR}/${name}ConfigVersion.cmake"
#    VERSION ${PROJECT_VERSION}
#    COMPATIBILITY AnyNewerVersion
#)

install(FILES
    "${PROJECT_BINARY_DIR}/${name}Config.cmake"
    "${PROJECT_BINARY_DIR}/${name}ConfigVersion.cmake"
    DESTINATION lib/cmake/${name}
)


add_executable(Test${name}  Test${name}.cc)
target_link_libraries( Test${name} PUBLIC ${name})
install(TARGETS Test${name}  DESTINATION lib)

