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


    return 0 ; 
}

