
set(ImGui_MODULE "${CMAKE_CURRENT_LIST_FILE}")
set(ImGui_PREFIX "${CMAKE_INSTALL_PREFIX}/externals")

find_path( ImGui_INCLUDE_DIR
           NAMES "ImGui/imgui.h"
           PATHS "${ImGui_PREFIX}/include"
)

find_library( ImGui_LIBRARY 
              NAMES ImGui 
              PATHS ${ImGui_PREFIX}/lib )


find_package(OpticksGLFW REQUIRED MODULE)


if(ImGui_INCLUDE_DIR AND ImGui_LIBRARY)
   set(ImGui_FOUND "YES")
else()
   set(ImGui_FOUND "NO")
endif()


set(tgt Opticks::ImGui)
if(ImGui_FOUND AND NOT TARGET ${tgt})

    add_library(${tgt} UNKNOWN IMPORTED) 
    set_target_properties(${tgt} PROPERTIES IMPORTED_LOCATION "${ImGui_LIBRARY}")

    if(APPLE)
    else()
       target_link_libraries(${tgt} INTERFACE GL Opticks::OpticksGLFW)
    endif()

    set_target_properties(${tgt} PROPERTIES 
        INTERFACE_INCLUDE_DIRECTORIES "${ImGui_INCLUDE_DIR}" 
        INTERFACE_FIND_PACKAGE_NAME "ImGui MODULE REQUIRED"
    )
    ## Above target_properties INTERFACE_FIND_PACKAGE_NAME kludge tees up the arguments 
    ## to find_dependency in BCM generated exports 
    ## so downstream targets will automatically do the required find_dependency
    ## and call this script again to revive the targets.
    ## NB INTERFACE_FIND_PACKAGE_NAME is a BCM defined property, not a standard one, see bcm-

endif()

if(ImGui_VERBOSE)
  message(STATUS "FindImGui.cmake : ImGui_MODULE      : ${ImGui_MODULE} " )
  message(STATUS "FindImGui.cmake : ImGui_LIBRARY     : ${ImGui_LIBRARY} " )
  message(STATUS "FindImGui.cmake : ImGui_INCLUDE_DIR : ${ImGui_INCLUDE_DIR} " )
endif()


