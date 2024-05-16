/**
U4TreeCreateSSimLoadTest.cc
==============================


1. SSim::Load from $BASE (the directory BASE should contain the "SSim" sub-dir) 
2. emit SSim::brief 


**/

#include "OPTICKS_LOG.hh"
#include "SSim.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    SSim* sim = SSim::Load("$FOLD") ;
    assert(sim && "FOLD directory in filesystem should contain \"SSim\" subfold"); 
    LOG(info) << "SSim::Load(\"$FOLD\") " << ( sim ? sim->brief() : "-" ) ;  
    return 0 ;  
}
