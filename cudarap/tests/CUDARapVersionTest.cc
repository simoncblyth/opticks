#include "cuda.h"

#include "OKConf_Config.hh"

#include "PLOG.hh"
#include "SSys.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

#if CUDA_VERSION != OKCONF_CUDA_API_VERSION_INTEGER
   #error FATAL CUDA VERSION INCONSISTENCY 
#endif

    LOG(info) 
        << " CUDA_VERSION " << CUDA_VERSION 
        << " OKCONF_CUDA_API_VERSION_INTEGER " << OKCONF_CUDA_API_VERSION_INTEGER 
       ; 

    return 0 ; 
}



