#[=[
FindJPhysiSim.cmake
====================

HMM: need to find CustomG4OpBoundaryProcess from install dirs (not source dirs). 

Need to arrange a target with the lib 

#]=]

set(JPhysiSim_MODULE  "${CMAKE_CURRENT_LIST_FILE}")


find_path(JPhysiSim_INCLUDE
  NAMES OK_PHYSISIM_LOG.hh   
  PATHS "$ENV{JUNOTOP}/junosw/InstallArea/include/PhysiSim"
  NO_DEFAULT_PATH
  )

if(JPhysiSim_VERBOSE)
    message(STATUS "${CMAKE_CURRENT_LIST_FILE} : JPhysiSim_VERBOSE : ${JPhysiSim_VERBOSE} ")
    message(STATUS "${CMAKE_CURRENT_LIST_FILE} : JPhysiSim_MODULE  : ${JPhysiSim_MODULE} ")
    message(STATUS "${CMAKE_CURRENT_LIST_FILE} : JPhysiSim_INCLUDE : ${JPhysiSim_INCLUDE} ")
endif()


if(JPhysiSim_INCLUDE)
   set(JPhysiSim_FOUND "YES")
else()
   set(JPhysiSim_FOUND "NO")
endif()



