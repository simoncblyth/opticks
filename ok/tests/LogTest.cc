#include "PLOG.hh"

#include "SYSRAP_LOG.hh"
#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "ASIRAP_LOG.hh"
#include "MESHRAP_LOG.hh"
#include "OKGEO_LOG.hh"
#include "OGLRAP_LOG.hh"

#ifdef WITH_OPTIX
#include "CUDARAP_LOG.hh"
#include "THRAP_LOG.hh"
#include "OXRAP_LOG.hh"
#include "OKOP_LOG.hh"
#include "OKGL_LOG.hh"
#endif

#include "OK_LOG.hh"


int main(int argc, char** argv)
{
    //PLOG_(argc, argv);
    PLOG_COLOR(argc, argv);

    SYSRAP_LOG__ ;
    BRAP_LOG__ ;
    NPY_LOG__ ;
    OKCORE_LOG__ ;
    GGEO_LOG__ ;
    ASIRAP_LOG__ ;
    MESHRAP_LOG__ ;
    OKGEO_LOG__ ;
    OGLRAP_LOG__ ;

#ifdef WITH_OPTIX
    CUDARAP_LOG__ ;
    THRAP_LOG__ ;
    OXRAP_LOG__ ;
    OKOP_LOG__ ;
    OKGL_LOG__ ;
#endif
    OK_LOG__ ;


    const char* msg = argv[0] ;

    SYSRAP_LOG::Check(msg) ;
    BRAP_LOG::Check(msg) ;
    NPY_LOG::Check(msg) ;
    OKCORE_LOG::Check(msg) ;
    GGEO_LOG::Check(msg) ;
    ASIRAP_LOG::Check(msg) ;
    MESHRAP_LOG::Check(msg) ;
    OKGEO_LOG::Check(msg) ;
    OGLRAP_LOG::Check(msg) ;

#ifdef WITH_OPTIX
    CUDARAP_LOG::Check(msg) ;
    THRAP_LOG::Check(msg) ;
    OXRAP_LOG::Check(msg) ;
    OKOP_LOG::Check(msg) ;
    OKGL_LOG::Check(msg) ;
#endif
    OK_LOG::Check(msg) ;


    return 0 ;
} 

/*

   Seems cannot turn up the loglevel in the projects, all are stuck at the fatal set in main.
   (chaining effect >?)

   LogTest --fatal --asirap trace
   LogTest --fatal --okop warn 
 
   However the converse does work. Can turn down project log level, but only down so far as the primary level.

   LogTest --trace --okop fatal --ggv fatal --okgl fatal --oxrap fatal --thrap fatal --cudarap fatal --oglrap fatal --okgeo fatal --meshrap fatal --asirap fatal --ggeo fatal --okcore fatal --npy fatal --sysrap fatal --brap fatal


   Generally there is very little need for logging control in the main, so can leave that 
   at trace and default the projects to info



*/


