#include <iostream>
#include <iomanip>

#include "OKConf.hh"
#include "OKCONF_OpticksCMakeConfig.hh"


int OKConf::Check()
{
   int rc = 0 ;  

   if(CUDAVersionInteger() == 0)
   {
       rc += 1 ; 
   }
   if(OptiXVersionInteger() == 0)
   {
       rc += 1 ; 
   }
   if(ComputeCapabilityInteger() == 0)
   {
       rc += 1 ; 
   }
   if(CUDA_NVCC_FLAGS() == 0)
   {
       rc += 1 ; 
   }
   if(CMAKE_CXX_FLAGS() == 0)
   {
       rc += 1 ; 
   }
   if(OptiXInstallDir() == 0)
   {
       rc += 1 ; 
   }
   if(Geant4VersionInteger() == 0)
   {
       rc += 1 ; 
   }
   return rc ; 
}


void OKConf::Dump(const char* msg)
{
    std::cout << msg << std::endl ; 
    std::cout << std::setw(50) << "OKConf::OpticksInstallPrefix() " << OKConf::OpticksInstallPrefix() << std::endl ; 
    std::cout << std::setw(50) << "OKConf::OptiXVersionInteger() "  << OKConf::OptiXVersionInteger() << std::endl ; 
    std::cout << std::setw(50) << "OKConf::ComputeCapabilityInteger() " << OKConf::ComputeCapabilityInteger() << std::endl ; 
    std::cout << std::setw(50) << "OKConf::CUDA_NVCC_FLAGS() "         << OKConf::CUDA_NVCC_FLAGS() << std::endl ; 
    std::cout << std::setw(50) << "OKConf::CMAKE_CXX_FLAGS() "         << OKConf::CMAKE_CXX_FLAGS() << std::endl ; 
    std::cout << std::setw(50) << "OKConf::OptiXInstallDir() "      << OKConf::OptiXInstallDir() << std::endl ; 
    std::cout << std::setw(50) << "OKConf::Geant4VersionInteger() " << OKConf::Geant4VersionInteger() << std::endl ; 
    std::cout << std::setw(50) << "OKConf::CUDAVersionInteger() " << OKConf::CUDAVersionInteger() << std::endl ; 
    std::cout << std::endl ; 
}

unsigned OKConf::CUDAVersionInteger()
{
#ifdef OKCONF_CUDA_API_VERSION_INTEGER
   return OKCONF_CUDA_API_VERSION_INTEGER ;
#else 
   return 0 ; 
#endif    
}

unsigned OKConf::OptiXVersionInteger()
{
#ifdef OKCONF_OPTIX_VERSION_INTEGER
   return OKCONF_OPTIX_VERSION_INTEGER ;
#else 
   return 0 ; 
#endif    
}

unsigned OKConf::Geant4VersionInteger()
{
#ifdef OKCONF_GEANT4_VERSION_INTEGER
   return OKCONF_GEANT4_VERSION_INTEGER ;
#else 
   return 0 ; 
#endif    
}

unsigned OKConf::ComputeCapabilityInteger()
{
#ifdef OKCONF_COMPUTE_CAPABILITY_INTEGER
   return OKCONF_COMPUTE_CAPABILITY_INTEGER ;
#else 
   return 0 ; 
#endif    
}

const char* OKConf::OpticksInstallPrefix()
{
#ifdef OKCONF_OPTICKS_INSTALL_PREFIX
   return OKCONF_OPTICKS_INSTALL_PREFIX ;
#else 
   return "MISSING" ; 
#endif    
}

const char* OKConf::OptiXInstallDir()
{
#ifdef OKCONF_OPTIX_INSTALL_DIR
   return OKCONF_OPTIX_INSTALL_DIR ;
#else 
   return "MISSING" ; 
#endif    
}

const char* OKConf::CUDA_NVCC_FLAGS()
{
#ifdef OKCONF_CUDA_NVCC_FLAGS
   return OKCONF_CUDA_NVCC_FLAGS ;
#else 
   return "MISSING" ; 
#endif    
}

const char* OKConf::CMAKE_CXX_FLAGS()
{
#ifdef OKCONF_CMAKE_CXX_FLAGS
   return OKCONF_CMAKE_CXX_FLAGS ;
#else 
   return "MISSING" ; 
#endif    
}








