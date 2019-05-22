#pragma once
/**
OKConf
=======

Static methods providing installation constants and 
version number integers of externals. 


**/

#include "OKCONF_API_EXPORT.hh"

#include "OKConf_Config.hh"

#ifdef OKCONF_OPTIX_VERSION_INTEGER

#define OKCONF_OPTIX_VERSION_MAJOR (OKCONF_OPTIX_VERSION_INTEGER / 10000)
#define OKCONF_OPTIX_VERSION_MINOR ((OKCONF_OPTIX_VERSION_INTEGER % 10000) / 100)
#define OKCONF_OPTIX_VERSION_MICRO (OKCONF_OPTIX_VERSION_INTEGER % 100)

#else

#define OKCONF_OPTIX_VERSION_INTEGER 0
#define OKCONF_OPTIX_VERSION_MAJOR 0
#define OKCONF_OPTIX_VERSION_MINOR 0
#define OKCONF_OPTIX_VERSION_MICRO 0

#endif


class OKCONF_API OKConf 
{
    public:
       static int Check(); 
       static void Dump(const char* msg="OKConf::Dump"); 
    public:
       static const char* OpticksInstallPrefix();
       static const char* OptiXInstallDir();
       //static const char* CUDA_NVCC_FLAGS();
       static const char* CMAKE_CXX_FLAGS();

       static unsigned ComputeCapabilityInteger();

       static int OptiXVersionInteger() ; 
       static int OptiXVersionMajor() ; 
       static int OptiXVersionMinor() ; 
       static int OptiXVersionMicro() ; 


       static unsigned Geant4VersionInteger() ; 
       static unsigned CUDAVersionInteger() ; 

       static const char* PTXPath( const char* cmake_target, const char* cu_name );
       static const char* ShaderDir();

};









