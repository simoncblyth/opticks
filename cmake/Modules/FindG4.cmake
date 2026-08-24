#[=[
FindG4.cmake
=============

The FindG4.cmake module integrates Geant4 provided CMake variables
with the target-centric approach of Opticks CMake machinery based on BCM,
see bcm-.

0. finds Geant4 via CMAKE_PREFIX_PATH
1. parses Geant4_VERSION_INTEGER from define in G4Version.hh
2. obtains Geant4_DIRDIR from Geant4_DIR
3. if Opticks::G4 imported target does not exist create it from the Geant4 variables

The Opticks::G4 imported target creation attempts to handle multiple Geant4 versions,
tested with 1042 and 1062.

Note that the "INTERFACE_FIND_PACKAGE_NAME" target property used by BCM (see bcm-)
is configured to  "G4 MODULE REQUIRED" which tees up the arguments to find_dependency
in BCM generated exports so downstream targets will automatically do the required find_dependency.
This means that only one find_package call is needed amongst a tree of packages::

     find_package( G4 MODULE REQUIRED )

The automation is achieved by BCM by generation of .cmake config files which run on
finding a dependenct.

For an examples of usage see the below examples which create a Geant4
using library and then links an executable to that::

   examples/UseG4NoOpticks
   examples/UseUseG4NoOpticks

Relevant part of Geant4 CMake config::

   /usr/local/opticks_externals/g4_1042/lib/Geant4-10.4.2/Geant4Config.cmake
   /usr/local/opticks_externals/g4_1062/lib/Geant4-10.6.2/Geant4Config.cmake

#]=]

set(G4_MODULE "${CMAKE_CURRENT_LIST_FILE}")
if(G4_VERBOSE)
    message(STATUS " G4_MODULE : ${G4_MODULE} " )
endif()


find_package(Geant4 CONFIG)

if(Geant4_FOUND)
    set(_dirs ${Geant4_INCLUDE_DIRS})
    list(GET _dirs 0 _firstdir)
    file(READ "${_firstdir}/G4Version.hh" _contents)
    string(REGEX REPLACE "\n" ";" _contents "${_contents}")
    foreach(_line ${_contents})
        if (_line MATCHES "#[ ]*define[ ]+G4VERSION_NUMBER[ ]+([0-9]+)$")
            set(G4_VERSION_INTEGER ${CMAKE_MATCH_1})
        endif()
    endforeach()

    set(G4_DIR ${Geant4_DIR})
    get_filename_component(G4_DIRDIR ${Geant4_DIR} DIRECTORY)

    if(G4_VERBOSE)
        message(STATUS " G4_VERSION_INTEGER : ${G4_VERSION_INTEGER} " )
    endif()

else()
    message(STATUS "\$ENV{CMAKE_PREFIX_PATH}:$ENV{CMAKE_PREFIX_PATH} ")
    message(STATUS "find_package(Geant4 CONFIG) FAILED : check envvar CMAKE_PREFIX_PATH   G4_MODULE : ${G4_MODULE} ")
    # FATAL_ERROR

endif()


set(G4_FOUND ${Geant4_FOUND})

if(G4_VERBOSE)
    message(STATUS)
    message(STATUS "G4_MODULE                : ${G4_MODULE} ")
    message(STATUS "G4_DIR                   : ${G4_DIR} ")
    message(STATUS "G4_DIRDIR                : ${G4_DIRDIR} ")
    message(STATUS "G4_VERSION_INTEGER       : ${G4_VERSION_INTEGER} ")
    message(STATUS)
    message(STATUS "Geant4_DIR               : ${Geant4_DIR} ")
    message(STATUS "Geant4_VERSION           : ${Geant4_VERSION} ")
    message(STATUS "Geant4_LIBRARIES         : ${Geant4_LIBRARIES} ")
    message(STATUS "Geant4_INCLUDE_DIRS      : ${Geant4_INCLUDE_DIRS} ")
    message(STATUS "Geant4_DEFINITIONS       : ${Geant4_DEFINITIONS} ")
    message(STATUS)
    message(STATUS "CMAKE_INSTALL_INCLUDEDIR : ${CMAKE_INSTALL_INCLUDEDIR} ")
    message(STATUS)
endif()


## try delegate to CMake approach suggested by Gemini which purports to working across many Geant4 versions
## so far found to work with 1140

if(G4_FOUND AND NOT TARGET Opticks::G4)

    set(_targets)
    set(G4_targets) # collection used by cmake/Modules/TopMetaTarget.cmake must not be Opticks:: qualified

    foreach(_lib ${Geant4_LIBRARIES})
        string(REGEX REPLACE "^Geant4::" "" _bare_lib "${_lib}")

        # Check if _lib is already a target (e.g. Geant4::G4processes, Geant4::G4geometry)
        if(TARGET ${_lib})
            # Forward the imported target directly to preserve all transitive linking & headers
            list(APPEND _targets ${_lib})
            list(APPEND G4_targets ${_bare_lib})
        else()
            # Legacy fallback for older G4 versions passing raw library names without target namespaces
            set(_loc "${_bare_lib}-NOTFOUND")
            find_library(_loc NAMES ${_bare_lib} PATHS ${G4_DIRDIR})

            if(_loc)
                set(_tgt "Opticks::${_bare_lib}")
                if(NOT TARGET ${_tgt})
                    add_library(${_tgt} UNKNOWN IMPORTED)
                    set_target_properties(${_tgt} PROPERTIES IMPORTED_LOCATION "${_loc}")
                    set_target_properties(${_tgt} PROPERTIES INTERFACE_IMPORTED_LOCATION "${_loc}") # workaround whitelisting restriction
                endif()
                list(APPEND _targets ${_tgt})
                list(APPEND G4_targets ${_bare_lib})
            else()
                message(FATAL_ERROR "failed to locate expected lib ${_bare_lib}")
            endif()
        endif()
    endforeach()

    string(REGEX REPLACE "-D" "" _defs "${Geant4_DEFINITIONS}")

    add_library(Opticks::G4 INTERFACE IMPORTED)

    set_target_properties(Opticks::G4 PROPERTIES
        INTERFACE_FIND_PACKAGE_NAME "G4 MODULE REQUIRED"
        INTERFACE_PKG_CONFIG_NAME "Geant4"
    )

    target_include_directories(Opticks::G4 INTERFACE "${Geant4_INCLUDE_DIRS}")
    target_link_libraries(Opticks::G4 INTERFACE ${_targets})
    target_compile_definitions(Opticks::G4 INTERFACE "${_defs}")

endif()



