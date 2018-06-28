// op --idpath

#include <iostream>
#include "Opticks.hh"

#include "OPTICKS_LOG.hh"

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
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv) ;
    ok.configure();

    std::cerr << ok.getIdPath() << std::endl ; 

    return 0 ; 
}
