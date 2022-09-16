/**
CSGSimtraceSampleTest.cc : running intersects of small simtrace arrays, eg saved from python 
=============================================================================================

**/

#include "OPTICKS_LOG.hh"
#include "CSGSimtraceSample.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGSimtraceSample t ; 
    t.intersect(); 

    return 0 ;
}

 
