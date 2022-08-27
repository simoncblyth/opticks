/**
CSGSimtraceRerunTest.cc : repeating simtrace GPU intersects on the CPU with the csg_intersect CUDA code
========================================================================================================

Roughly based on CSGQueryTest.cc but updated for simtrace array reruns

For full detail reporting suitable only for single (or few) intersections 
it is necessary to recompile with DEBUG and DEBUG_RECORD. 

**/

#include "OPTICKS_LOG.hh"
#include "CSGSimtraceRerun.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGSimtraceRerun t ; 
    t.intersect_again(); 
    t.save(); 
    t.report(); 

    return 0 ;
}

 
