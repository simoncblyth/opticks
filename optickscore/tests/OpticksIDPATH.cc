// op --idpath

#include <iostream>
#include "Opticks.hh"

#ifdef DO_LOG
#include "SYSRAP_LOG.hh"
#include "BRAP_LOG.hh"
#include "OKCORE_LOG.hh"
#include "PLOG.hh"
#endif

/**
OpticksIDPATH 
===============

Simple executable that dumps the directory of the 
Opticks geocache to stderr. The directory returned 
depends on the arguments provided that select a detector
DAE file and also select subsets of the geometry. 

After setting PATH use::

   args="" # potentially select 

   IDPATH="$(op --idpath 2>&1 > /dev/null)"  # capture only stderr


**/


int main(int argc, char** argv)
{
#ifdef DO_LOG
    PLOG_(argc, argv);
    SYSRAP_LOG__ ;
    BRAP_LOG__ ;
    OKCORE_LOG__ ;
#endif

    Opticks ok(argc, argv) ;
    ok.configure();

    std::cerr << ok.getIdPath() << std::endl ; 

    return 0 ; 
}
