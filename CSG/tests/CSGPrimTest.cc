#include "SSys.hh"
#include "scuda.h"
#include "CSGFoundry.h"
#include "CSGName.h"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGFoundry* fd = CSGFoundry::Load(); 

    LOG(info) << "fd.desc" << fd->desc() ; 
    LOG(info) << "fd.summary" ; 
    fd->summary(); 

    LOG(info) << "fd.detailPrim" << std::endl << fd->detailPrim() ; 

    return 0 ; 
}

