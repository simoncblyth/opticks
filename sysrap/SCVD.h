#pragma once
/**
SCVD.hh : promotes CVD envvar into CUDA_VISIBLE_DEVICES
===========================================================

The advantage of using the short CVD envvar instead of the underlying CUDA_VISIBLE_DEVICES
is that Opticks then manages its use and can overrride and incorporate results in bookkeeping. 

The intention is for CVD envvar to be used and not CUDA_VISIBLE_DEVICES, 
although there is currently no enforcement of this. 

Usage, early in main prior to any CUDA context setup::

   int main(int argc, char** argv))
   {
       OPTICKS_LOG(argc, argv); 
       SCVD::ConfigureVisibleDevices();      
       ...
       return 0 ; 
   }   

TODO: perhaps incorporate into OPTICKS_LOG ?

**/

#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <sstream>

struct SCVD 
{
    static const char* CVD_DEFAULT ;      
    static const char* CVD ;      
    static const char* Get() ;      

    static const char* LABEL ;      
    static const char* Label() ;      

    static void ConfigureVisibleDevices(); 
};


const char* SCVD::CVD_DEFAULT = "0" ; 
const char* SCVD::CVD = getenv("CVD"); 
inline const char* SCVD::Get(){ return CVD == nullptr ? CVD_DEFAULT : CVD ; }


const char* SCVD::LABEL = nullptr ; 

inline const char* SCVD::Label()
{  
    if(LABEL == nullptr) 
    {
        std::stringstream ss ; 
        ss << "SCVD" << Get() ;  
        std::string s = ss.str(); 
        LABEL = strdup(s.c_str()) ; 
    }
    return LABEL ; 
}

inline void SCVD::ConfigureVisibleDevices()  // static
{
    std::stringstream ss ; 
    ss << "CUDA_VISIBLE_DEVICES=" << Get() ; 
    std::string s = ss.str(); 
    char* ekv = const_cast<char*>(s.c_str()); 
    putenv(ekv); 
    std::cout << "SCVD::ConfigureVisibleDevices putenv " << ekv << std::endl ; 
}


