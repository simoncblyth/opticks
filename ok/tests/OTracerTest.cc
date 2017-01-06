#include "OKMgr.hh"

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

/**


OTracerTest
================

Expedient separate executable. Equivalent to running::

   OKTest --nopropagate 
   OKTest -P


**/

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
 
    OKMgr ok(argc, argv, "--trace" );

    ok.visualize();

    exit(EXIT_SUCCESS);
}

