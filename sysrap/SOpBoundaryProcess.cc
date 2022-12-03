#include <cstring>
#include "SOpBoundaryProcess.hh"

SOpBoundaryProcess* SOpBoundaryProcess::INSTANCE = nullptr ; 
SOpBoundaryProcess* SOpBoundaryProcess::Get(){ return INSTANCE ; 
}
SOpBoundaryProcess::SOpBoundaryProcess(const char* name_)
    :
    name(strdup(name_))
{
    INSTANCE = this ; 
}



