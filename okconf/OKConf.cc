#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cstdlib>


#include "OKConf.hh"
#include "OKConf_Config.hh"


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
   /*
   // this setup is now downstream of OKConf, actually from OKConf TOPMATTER
   if(CUDA_NVCC_FLAGS() == 0)
   {
       rc += 1 ; 
   }
   */ 
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
    std::cout << std::setw(50) << "OKConf::OpticksInstallPrefix() "    << OKConf::OpticksInstallPrefix() << std::endl ; 
    std::cout << std::setw(50) << "OKConf::CMAKE_CXX_FLAGS() "         << OKConf::CMAKE_CXX_FLAGS() << std::endl ; 
    std::cout << std::setw(50) << "OKConf::CUDAVersionInteger() "      << OKConf::CUDAVersionInteger() << std::endl ; 
    std::cout << std::setw(50) << "OKConf::ComputeCapabilityInteger() "<< OKConf::ComputeCapabilityInteger() << std::endl ; 
    //std::cout << std::setw(50) << "OKConf::CUDA_NVCC_FLAGS() "         << OKConf::CUDA_NVCC_FLAGS() << std::endl ; 

    std::cout << std::setw(50) << "OKConf::OptiXInstallDir() "         << OKConf::OptiXInstallDir() << std::endl ; 
    std::cout << std::setw(50) << "OKCONF_OPTIX_VERSION_INTEGER "      << OKCONF_OPTIX_VERSION_INTEGER << std::endl ; 
    std::cout << std::setw(50) << "OKConf::OptiXVersionInteger() "     << OKConf::OptiXVersionInteger() << std::endl ; 
    std::cout << std::setw(50) << "OKCONF_OPTIX_VERSION_MAJOR   "      << OKCONF_OPTIX_VERSION_MAJOR << std::endl ; 
    std::cout << std::setw(50) << "OKConf::OptiXVersionMajor() "       << OKConf::OptiXVersionMajor() << std::endl ; 
    std::cout << std::setw(50) << "OKCONF_OPTIX_VERSION_MINOR   "      << OKCONF_OPTIX_VERSION_MINOR << std::endl ; 
    std::cout << std::setw(50) << "OKConf::OptiXVersionMinor() "       << OKConf::OptiXVersionMinor() << std::endl ; 
    std::cout << std::setw(50) << "OKCONF_OPTIX_VERSION_MICRO   "      << OKCONF_OPTIX_VERSION_MICRO << std::endl ; 
    std::cout << std::setw(50) << "OKConf::OptiXVersionMicro() "       << OKConf::OptiXVersionMicro() << std::endl ; 


    std::cout << std::setw(50) << "OKConf::Geant4VersionInteger() "    << OKConf::Geant4VersionInteger() << std::endl ; 
    std::cout << std::setw(50) << "OKConf::ShaderDir()            "    << OKConf::ShaderDir() << std::endl ; 
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

int OKConf::OptiXVersionInteger()
{
#ifdef OKCONF_OPTIX_VERSION_INTEGER
   return OKCONF_OPTIX_VERSION_INTEGER ;
#else 
   return 0 ; 
#endif    
}
int OKConf::OptiXVersionMajor()
{
#ifdef OKCONF_OPTIX_VERSION_MAJOR
   return OKCONF_OPTIX_VERSION_MAJOR ;
#else 
   return 0 ; 
#endif    
}
int OKConf::OptiXVersionMinor()
{
#ifdef OKCONF_OPTIX_VERSION_MINOR
   return OKCONF_OPTIX_VERSION_MINOR ;
#else 
   return 0 ; 
#endif    
}
int OKConf::OptiXVersionMicro()
{
#ifdef OKCONF_OPTIX_VERSION_MICRO
   return OKCONF_OPTIX_VERSION_MICRO ;
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
   const char* evalue = getenv("OPTICKS_INSTALL_PREFIX") ;  
   return evalue ? evalue : OKCONF_OPTICKS_INSTALL_PREFIX ;
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

/*
const char* OKConf::CUDA_NVCC_FLAGS()
{
#ifdef OKCONF_CUDA_NVCC_FLAGS
   return OKCONF_CUDA_NVCC_FLAGS ;
#else 
   return "MISSING" ; 
#endif    
}
*/

const char* OKConf::CMAKE_CXX_FLAGS()
{
#ifdef OKCONF_CMAKE_CXX_FLAGS
   return OKCONF_CMAKE_CXX_FLAGS ;
#else 
   return "MISSING" ; 
#endif    
}


/**
OKConf::PTXPath
-----------------

The path elements configured here must match those from the CMakeLists.txt 
that compiles the <name>.cu to <target>_generated_<name>.cu.ptx eg in optixrap/CMakeLists.txt::

    091 set(CU_SOURCES
    092 
    093     cu/pinhole_camera.cu
    094     cu/constantbg.cu
    ...
    120     cu/intersect_analytic_test.cu
    121     cu/Roots3And4Test.cu
    122 )
    ...
    131 CUDA_WRAP_SRCS( ${name} PTX _generated_PTX_files ${CU_SOURCES} )
    132 CUDA_WRAP_SRCS( ${name} OBJ _generated_OBJ_files ${SOURCES} )
    133 
    134 
    135 add_library( ${name} SHARED ${_generated_OBJ_files} ${_generated_PTX_files} ${SOURCES} )
    136 #[=[
    137 The PTX are not archived in the lib, it is just expedient to list them as sources
    138 of the lib target so they get hooked up as dependencies, and thus are generated before
    139 they need to be installed
    140 #]=]
    ...
    170 install(FILES ${_generated_PTX_files} DESTINATION installcache/PTX)

The form of the PTX filename comes from the FindCUDA.cmake file for example at
/usr/share/cmake3/Modules/FindCUDA.cmake 

**/
const char* OKConf::PTXPath( const char* cmake_target, const char* cu_name, const char* ptxrel )
{
    std::stringstream ss ; 
    ss << OKConf::OpticksInstallPrefix()
       << "/installcache/PTX/"
       ;

    if(ptxrel) ss << ptxrel << "/" ;   

    ss
       << cmake_target
       << "_generated_"
       << cu_name
       << ".ptx" 
       ;
    std::string ptxpath = ss.str();
    return strdup(ptxpath.c_str()); 
}


const char* OKConf::ShaderDir()
{
    std::stringstream ss ; 
    ss << OKConf::OpticksInstallPrefix()
       << "/gl"
       ;
    std::string shaderdir = ss.str();
    return strdup(shaderdir.c_str()); 
}








