#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "GPts.hh"
#include "GParts.hh"
#include "NCSG.hpp"
#include "GMeshLib.hh"

int main(int argc, char** argv)
{
    unsigned ridx = argc > 1 ? std::atoi(argv[1]) : 0 ; 

    OPTICKS_LOG(argc, argv); 
    Opticks ok(argc, argv); 
    ok.configure(); 

    std::string objpath = ok.getObjectPath("GPts", ridx ); 
    LOG(info) 
        << std::endl 
        << " objpath " << objpath << std::endl 
        ; 

    GPts* pts = GPts::Load(objpath.c_str()); 
    pts->dump("GPartsCreateTest.main"); 

    GMeshLib* mlib = GMeshLib::Load(&ok); 
    const std::vector<const NCSG*>& solids = mlib->getSolids(); 

    GParts* com = GParts::Create(&ok, pts, solids); 
    assert(com); 

    unsigned num_idx = com->getNumIdx() ; 

    LOG(info) 
        << " ridx " << ridx 
        << " num_idx " << num_idx 
        ; 
      

    for(unsigned i=0 ; i < num_idx ; i++)
    {
        std::cout << std::setw(4) << i << " : " ; 
        for(unsigned j=0 ; j < 4 ; j++)  std::cout << std::setw(6) << com->getUIntIdx(i,j) << " " ; 

        unsigned nidx = com->getVolumeIndex(i); 
        unsigned midx = com->getMeshIndex(i) ; 
        const char* mname = mlib->getMeshName(midx); 

        std::cout 
            << " : "
            << " nidx " << std::setw(6) << nidx 
            << " midx " << std::setw(6) << midx 
            << " mname " << ( mname ? mname : "-" ) 
            << std::endl
            ;
    }


    return 0 ; 
}

