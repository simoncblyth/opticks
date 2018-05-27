#pragma once

#include "OKCONF_API_EXPORT.hh"

class OKCONF_API OKConf 
{
    public:
       static int Check(); 
       static void Dump(const char* msg="OKConf::Dump"); 
    public:
       static const char* OpticksInstallPrefix();
       static const char* OptiXInstallDir();
       static const char* CUDA_NVCC_FLAGS();
       static const char* CMAKE_CXX_FLAGS();

       static unsigned ComputeCapabilityInteger();
       static unsigned OptiXVersionInteger() ; 
       static unsigned Geant4VersionInteger() ; 
       static unsigned CUDAVersionInteger() ; 
};









