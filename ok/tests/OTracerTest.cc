#include "OKMgr.hh"
#include "OPTICKS_LOG.hh"

/**
OTracerTest
================

Expedient separate executable. Equivalent to running::

   OKTest --nopropagate 
   OKTest -P

**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    OKMgr ok(argc, argv, "--tracer" );

    ok.visualize();

    //  exit(EXIT_SUCCESS);   // dont do this, as it prevents cleanup being called
    return ok.rc();  
}

