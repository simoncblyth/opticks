#include <cassert>
#include <cstring>
#include <iostream>

#include "SSys.hh"
#include "BOpticksResource.hh"

// CMake generated defines from binary_dir/inc
#include "BOpticksResourceCMakeConfig.hh"  

#include "PLOG.hh"


BOpticksResource::BOpticksResource(const char* envprefix)
   :
     m_envprefix(strdup(envprefix)),
     m_install_prefix(NULL)
{
    init();
}

BOpticksResource::~BOpticksResource()
{
}

void BOpticksResource::init()
{
    adoptInstallPrefix() ;
}

void BOpticksResource::adoptInstallPrefix()
{
   m_install_prefix = strdup(OPTICKS_INSTALL_PREFIX) ; 
   
   const char* key = "INSTALL_PREFIX" ; 

   int rc = SSys::setenvvar(m_envprefix, key, m_install_prefix, true );  

   LOG(trace) << "OpticksResource::adoptInstallPrefix " 
              << " install_prefix " << m_install_prefix  
              << " envprefix " << m_envprefix  
              << " key " << key 
              << " rc " << rc
              ;   
 
   assert(rc==0); 

   // The CMAKE_INSTALL_PREFIX from opticks-;opticks-cmake 
   // is set to the result of the opticks-prefix bash function 
   // at configure time.
   // This is recorded into a config file by okc-/CMakeLists.txt 
   // and gets compiled into the OpticksCore library.
   //  
   // Canonically it is :  /usr/local/opticks 
}

void BOpticksResource::Summary(const char* msg)
{
    std::cerr << msg << std::endl ; 
    const char* prefix = m_install_prefix ; 

    std::cerr << "prefix   : " <<  (prefix ? prefix : "NULL" ) << std::endl ; 
    std::cerr << "envprefix: " <<  (m_envprefix?m_envprefix:"NULL") << std::endl; 
}

