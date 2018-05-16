# - Find Opticks library
# This module sets up Opticks information
# adapted from the Geant4 equivalent: environments/g4py/cmake/Modules/FindGeant4.cmake
#
# It defines:
# OPTICKS_FOUND               If Opticks is found
# OPTICKS_INCLUDE_DIRS        PATH to the include directory(s)
# OPTICKS_LIBRARY_DIRS        PATH to the library directory(s)
# OPTICKS_LIBRARIES           Compute only libraries
# OPTICKS_LIBRARIES_WITH_VIS  Compute and visualisation libraries 

find_program(OPTICKS_CONFIG NAMES opticks-config
             PATHS $ENV{OPTICKS_INSTALL}/bin
                   ${OPTICKS_INSTALL}/bin
                   /usr/local/bin /opt/local/bin)

if(OPTICKS_CONFIG)
  set(OPTICKS_FOUND TRUE)

  execute_process(COMMAND ${OPTICKS_CONFIG} --prefix
                  OUTPUT_VARIABLE OPTICKS_PREFIX
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND ${OPTICKS_CONFIG} --version
                  OUTPUT_VARIABLE OPTICKS_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND ${OPTICKS_CONFIG} --libs-without-gui
                  OUTPUT_VARIABLE OPTICKS_LIBRARIES
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND ${OPTICKS_CONFIG} --libs
                  OUTPUT_VARIABLE OPTICKS_LIBRARIES_WITH_VIS
                  OUTPUT_STRIP_TRAILING_WHITESPACE)


  set(OPTICKS_INCLUDE_DIRS ${OPTICKS_PREFIX}/include)
  set(OPTICKS_LIBRARY_DIRS ${OPTICKS_PREFIX}/lib)

  message(STATUS "Found Opticks: ${OPTICKS_PREFIX} (${OPTICKS_VERSION})")
  message(STATUS " OPTICKS_PREFIX : ${OPTICKS_PREFIX} " )
  message(STATUS " OPTICKS_LIBRARIES : ${OPTICKS_LIBRARIES} " )
  message(STATUS " OPTICKS_LIBRARIES_WITH_VIS : ${OPTICKS_LIBRARIES_WITH_VIS} " )
  message(STATUS " OPTICKS_LIBRARY_DIRS : ${OPTICKS_LIBRARY_DIRS} " )
  message(STATUS " OPTICKS_INCLUDE_DIRS : ${OPTICKS_INCLUDE_DIRS} " )
 

else()
  set(OPTICKS_FOUND FALSE)
  message(SEND_ERROR "NOT Found Opticks: set OPTICKS_INSTALL env.")

endif()

