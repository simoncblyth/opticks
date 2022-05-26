
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "sframe.h"

#include "SPath.hh"
#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "CSGFoundry.h"


void test_getFrame_MOI(const CSGFoundry* fd )
{
    sframe fr = fd->getFrame() ;  // depends on MOI 

    std::cout << " fr " << std::endl << fr << std::endl ; 

    //const char* dir = SPath::Resolve("$TMP/CSG/CSGFoundry_getFrame_Test", DIRPATH); 
    //std::cout << dir << std::endl ; 
    //fr.save(dir) ; 
}

void test_getFrame_inst(const CSGFoundry* fd )
{
    int inst_idx = SSys::getenvint("INST", 0); 

    sframe fr ; 
    fd->getFrame(fr, inst_idx) ;  



    //const std::string& label = fd->getSolidLabel(sidx); 


    std::cout
        << " INST " << inst_idx << std::endl 
        << " fr " << std::endl 
        << fr << std::endl 
        << std::endl
        << "descInstance"
        << std::endl
        << fd->descInstance(inst_idx)
        << std::endl
        ; 

    //const char* dir = SPath::Resolve("$TMP/CSG/CSGFoundry_getFrame_Test", DIRPATH); 
    //std::cout << dir << std::endl ; 
    //fr.save(dir) ; 
}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGFoundry* fd = CSGFoundry::Load();

    //test_getFrame_MOI(fd); 

    test_getFrame_inst(fd); 

    return 0 ; 
}

