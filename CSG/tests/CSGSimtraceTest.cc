/**
CSGSimtraceTest
======================

Used from script CSG/ct.sh 

**/
#include "OPTICKS_LOG.hh"
#include "CSGSimtrace.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGSimtrace t ;
    t.simtrace(); 
    t.saveEvent() ; 

    return 0 ; 
}

