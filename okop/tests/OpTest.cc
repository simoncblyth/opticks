#include "OpMgr.hh"

#include "PLOG.hh"

#include "SYSRAP_LOG.hh"
#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "OKGEO_LOG.hh"


#include "CUDARAP_LOG.hh"
#include "THRAP_LOG.hh"
#include "OXRAP_LOG.hh"
#include "OKOP_LOG.hh"


/**

OpTest
================
**/

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    //PLOG_COLOR(argc, argv);

    SYSRAP_LOG__ ;
    BRAP_LOG__ ;
    NPY_LOG__ ;
    OKCORE_LOG__ ;
    GGEO_LOG__ ;
    OKGEO_LOG__ ;

    CUDARAP_LOG__ ;
    THRAP_LOG__ ;
    OXRAP_LOG__ ;
    OKOP_LOG__ ;


 
    OpMgr op(argc, argv);
    op.snap();
    exit(EXIT_SUCCESS);
}

