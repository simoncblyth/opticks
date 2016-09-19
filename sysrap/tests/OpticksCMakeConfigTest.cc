#include "OpticksCMakeConfig.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);


#if OXRAP_OPTIX_VERSION >= 3080
    LOG(info) << " OXRAP_OPTIX_VERSION >= 3080 : " << OXRAP_OPTIX_VERSION  ;
#else
    LOG(info) << " (NOT) OXRAP_OPTIX_VERSION > 3080 : " << OXRAP_OPTIX_VERSION  ;
#endif



    return 0 ; 
}
