#include "ssys.h"
#include "sframe.h"
#include "SSim.hh"
#include "OPTICKS_LOG.hh"
#include "CSGFoundry.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    bool VERBOSE=ssys::getenvbool("VERBOSE");

    SSim::Create();
    const CSGFoundry* fd = CSGFoundry::Load();

    if(VERBOSE) std::cout
        << " fd.brief " << fd->brief() << "\n"
        << " fd.desc  " << fd->desc() << "\n"
        ;

    if(VERBOSE) std::cout << "[ fd->getFrameE " << std::endl ;
    sframe fr = fd->getFrameE() ;  // via INST, MOI, OPTICKS_INPUT_PHOTON_FRAME
    if(VERBOSE) std::cout << "] fd->getFrameE " << std::endl ;

    if(VERBOSE) std::cout << " [ fr.save " << std::endl ;
    fr.save("$FOLD");
    if(VERBOSE) std::cout << " ] fr.save " << std::endl ;
    if(VERBOSE) std::cout << " [ fr.save_extras " << std::endl ;
    fr.save_extras("$FOLD");
    if(VERBOSE) std::cout << " ] fr.save_extras " << std::endl ;


    std::cout << fr << std::endl ;


    return 0 ;
}

