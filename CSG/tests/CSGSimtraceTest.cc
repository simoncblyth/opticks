/**
CSGSimtraceTest : CPU testing of CUDA capable csg intersection headers
========================================================================

Used from script CSG/ct.sh

**/
#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "CSGSimtrace.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    SEventConfig::SetRGModeSimtrace();

    CSGSimtrace t ;
    t.simtrace();

    return 0 ;
}

