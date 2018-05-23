# follow Geant4ConfigureConfigScript.cmake
#
# - Script for configuring and installing opticks-config script
#
# The opticks-config script provides an sh based interface to provide
# information on the Opticks installation, including installation prefix,
# version number, compiler and linker flags.
#
# The script is generated from a template file and then installed to the
# known bindir as an executable.
#
# Paths are always hardcoded in the build tree version as this is never
# intended to be relocatable.
# The Install Tree script uses self-location based on that in
# {root,clehep}-config is the install itself is relocatable, otherwise
# absolute paths are encoded.
#
#

#-----------------------------------------------------------------------
# function get_system_include_dirs
#          return list of directories our C++ compiler searches
#          by default.
#
#          The idea comes from CMake's inbuilt technique to do this
#          for the Eclipse and CodeBlocks generators, but we implement
#          our own function because the CMake functionality is internal
#          so we can't rely on it.
function(get_system_include_dirs _dirs)
  # Only for GCC, Clang and Intel
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES GNU OR "${CMAKE_CXX_COMPILER_ID}" MATCHES Clang OR "${CMAKE_CXX_COMPILER_ID}" MATCHES Intel)
    # Proceed
    file(WRITE "${CMAKE_BINARY_DIR}/CMakeFiles/get_system_include_dirs_dummy" "\n")

    # Save locale, them to "C" english locale so we can parse in English
    set(_orig_lc_all      $ENV{LC_ALL})
    set(_orig_lc_messages $ENV{LC_MESSAGES})
    set(_orig_lang        $ENV{LANG})

    set(ENV{LC_ALL}      C)
    set(ENV{LC_MESSAGES} C)
    set(ENV{LANG}        C)

    execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} -v -E -x c++ -dD g4dummy
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/CMakeFiles
      ERROR_VARIABLE _cxxOutput
      OUTPUT_VARIABLE _cxxStdout
      )

    file(REMOVE "${CMAKE_BINARY_DIR}/CMakeFiles/get_system_include_dirs_dummy")

    # Parse and extract search dirs
    set(_resultIncludeDirs )
    if( "${_cxxOutput}" MATCHES "> search starts here[^\n]+\n *(.+ *\n) *End of (search) list" )
      string(REGEX MATCHALL "[^\n]+\n" _includeLines "${CMAKE_MATCH_1}")
      foreach(nextLine ${_includeLines})
        string(REGEX REPLACE "\\(framework directory\\)" "" nextLineNoFramework "${nextLine}")
        string(STRIP "${nextLineNoFramework}" _includePath)
        list(APPEND _resultIncludeDirs "${_includePath}")
      endforeach()
    endif()

    # Restore original locale
    set(ENV{LC_ALL}      ${_orig_lc_all})
    set(ENV{LC_MESSAGES} ${_orig_lc_messages})
    set(ENV{LANG}        ${_orig_lang})

    set(${_dirs} ${_resultIncludeDirs} PARENT_SCOPE)
  else()
    set(${_dirs} "" PARENT_SCOPE)
  endif()
endfunction()

#-----------------------------------------------------------------------
# Only create script if we have a global library build...
#
if(UNIX)
  # Get implicit search paths
  get_system_include_dirs(_cxx_compiler_dirs)

  # Hardcoded paths
  set(OPTICKS_CONFIG_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
  set(OPTICKS_CONFIG_INSTALL_EXECPREFIX \"\")
  set(OPTICKS_CONFIG_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib")
  set(OPTICKS_CONFIG_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include")


  configure_file(
      ${CMAKE_SOURCE_DIR}/opticks-config.in
      ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/opticks-config
      @ONLY
      )

  file(COPY
      ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/opticks-config
      DESTINATION ${PROJECT_BINARY_DIR}
      FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
      )


  message("CMAKE_INSTALL_PREFIX:${CMAKE_INSTALL_PREFIX} " ) 
  message("CMAKE_INSTALL_BINDIR:${CMAKE_INSTALL_BINDIR} " ) 

  install(FILES ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/opticks-config
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
    COMPONENT Development
    )

endif()

