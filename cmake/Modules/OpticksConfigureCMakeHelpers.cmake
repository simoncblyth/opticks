#
# Initially based on 
#      /usr/local/opticks/externals/g4/geant4_10_04_p01/cmake/Modules/G4ConfigureCMakeHelpers.cmake
# but that is OTT for whats needed here 


set(OPTICKS_LIBRARIES
   SysRap
   BoostRap
   NPY
   OpticksCore
   AssimpRap
   OpenMeshRap
   GGeo
   OpticksGeometry
   OptiXRap
   CUDARap
   ThrustRap
   OKOP
)



if(UNIX)

  set(OPTICKS_CONFIG_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
  set(OPTICKS_CONFIG_INSTALL_EXECPREFIX \"\")
  set(OPTICKS_CONFIG_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib")
  set(OPTICKS_CONFIG_INCDIR "${CMAKE_INSTALL_PREFIX}/include")

  set(OPTICKS_CONFIG_LIBRARIES    "${OPTICKS_LIBRARIES}")
  set(OPTICKS_CONFIG_DEFINITIONS "")
  set(OPTICKS_CONFIG_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include")
  set(OPTICKS_CONFIG_VERSION      "${OPTICKS_VERSION}")

  configure_file(
      ${CMAKE_SOURCE_DIR}/cmake/Templates/OpticksConfig.cmake.in
      ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/OpticksConfig.cmake
      @ONLY
      )

  file(COPY
      ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/OpticksConfig.cmake
      DESTINATION ${PROJECT_BINARY_DIR}
      FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
      )

  message("CMAKE_INSTALL_PREFIX:${CMAKE_INSTALL_PREFIX} " ) 
  message("CMAKE_INSTALL_BINDIR:${CMAKE_INSTALL_BINDIR} " ) 

  install(FILES ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/OpticksConfig.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/config
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
    COMPONENT Development
    )

endif()



