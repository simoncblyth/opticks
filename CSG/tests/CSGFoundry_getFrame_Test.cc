/**
CSG/tests/CSGFoundry_getFrame_Test.cc
======================================

::

   TEST=getFrameE ~/o/CSG/tests/CSGFoundry_getFrame_Test.sh


**/
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "sfr.h"
#include "stree.h"
#include "ssys.h"
#include "SSim.hh"
#include "SEvt.hh"
#include "OPTICKS_LOG.hh"
#include "CSGFoundry.h"

struct CSGFoundry_getFrame_Test
{
    const CSGFoundry* fd ;
    const stree* tree ;
    sfr fr = {} ;

    CSGFoundry_getFrame_Test();

    int getFrameE();
    int save();
    int InputPhoton();
    int main();
};

inline CSGFoundry_getFrame_Test::CSGFoundry_getFrame_Test()
    :
    fd(CSGFoundry::Load()),
    tree(fd->getTree())
{
    std::cout << " fd.brief " << fd->brief() << std::endl ;
    std::cout << " fd.desc  " << fd->desc() << std::endl ;
}

inline int CSGFoundry_getFrame_Test::getFrameE()
{
    std::cout << "[ tree->get_frame_moi " << std::endl ;
    fr = tree->get_frame_moi();
    std::cout << "] fd->get_frame_moi " << std::endl ;

    int INST = ssys::getenvint("INST",-1) ;
    if(INST > -1) std::cout <<  "INST" << INST << std::endl << fd->descInstance(INST) << std::endl ;

    std::cout << "[ fr " << std::endl << fr << std::endl << " ] fr " << std::endl ;

    return 0 ;
}

inline int CSGFoundry_getFrame_Test::save()
{
    std::cout << " [ fr.save " << std::endl ;
    fr.save("$FOLD");
    std::cout << " ] fr.save " << std::endl ;
    return 0 ;
}


inline int CSGFoundry_getFrame_Test::InputPhoton()
{
    SEvt* evt = SEvt::Create(0) ;

    NP* ip = evt->getInputPhoton_();
    if(ip == nullptr) return 0 ;

    NP* ipt0 = fr.transform_photon_m2w(ip, false)  ; // normalize:false
    NP* ipt1 = fr.transform_photon_m2w(ip, true )  ; // normalize:true

    std::cout << " ip   " << ( ip  ? ip->sstr() : "-" )  << std::endl ;
    std::cout << " ipt0  " << ( ipt0 ? ipt0->sstr() : "-" )  << std::endl ;
    std::cout << " ipt1  " << ( ipt1 ? ipt1->sstr() : "-" )  << std::endl ;

    ip->save("$FOLD/ip.npy");
    ipt0->save("$FOLD/ipt0.npy");
    ipt1->save("$FOLD/ipt1.npy");

    evt->setFr(fr);

    NP* ipt2 = evt->getInputPhoton() ;
    ipt2->save("$FOLD/ipt2.npy");

    return 0 ;
}

inline int CSGFoundry_getFrame_Test::main()
{
    const char* TEST = ssys::getenvvar("TEST", "getFrameE");
    int rc = 0 ;
    if(strcmp(TEST,"getFrameE")==0)
    {
        rc += getFrameE();
    }
    else if(strcmp(TEST,"save")==0)
    {
        rc += getFrameE();
        rc += save();
    }
    else if(strcmp(TEST,"InputPhoton")==0)
    {
        rc += getFrameE();
        rc += InputPhoton();
    }
    return rc ;
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    CSGFoundry_getFrame_Test test ;
    return test.main() ;
}

