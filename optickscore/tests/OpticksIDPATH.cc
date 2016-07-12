// op --idpath

#include <iostream>
#include "Opticks.hh"

#include "SYSRAP_LOG.hh"
#include "BRAP_LOG.hh"
#include "OKCORE_LOG.hh"
#include "PLOG.hh"

/**
OpticksIDPATH 
===============

Simple executable that dumps the directory of the 
Opticks geocache to stderr. The directory returned 
depends on the arguments provided that select a detector
DAE file and also select subsets of the geometry. 

After setting PATH use::

   args="" # potentially select 
   export IDPATH=$(OpticksIDPATH $args 1>/dev/null)  # ignore stdout logging 


**/


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    SYSRAP_LOG__ ;
    BRAP_LOG__ ;
    OKCORE_LOG__ ;

    Opticks ok(argc, argv) ;
    ok.configure();

    std::cerr << ok.getIdPath() << std::endl ; 

    return 0 ; 
}
