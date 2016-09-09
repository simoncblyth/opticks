#include "OpticksGun.hh"
#include "OpticksHub.hh"
#include "Opticks.hh"

#include "PLOG.hh"

OpticksGun::OpticksGun(OpticksHub* hub)
   :
   m_hub(hub),
   m_ok(hub->getOpticks())
{
    init();
}

void OpticksGun::init()
{
}

std::string OpticksGun::getConfig()
{
    std::string config ; 
    int itag = m_ok->getEventITag();

    if( itag == 1 )
         config.assign(
    "comment=default-config-comment-without-spaces-_"
    "particle=mu-_"
    "frame=3153_"
    "position=0,0,-1_"
    "direction=0,0,1_"
    "polarization=1,0,0_"
    "time=0.1_"
    "energy=1000.0_"
    "number=1_")
    ;  // mm,ns,MeV 

    else if(itag == 100)
         config.assign(
    "comment=default-config-comment-without-spaces-_"
    "particle=mu-_"
    "frame=3153_"
    "position=0,0,-1_"
    "direction=0,0,1_"
    "polarization=1,0,0_"
    "time=0.1_"
    "energy=100000.0_"
    "number=1_")
    ;  // mm,ns,MeV 


    LOG(info) << "OpticksGun::getConfig"
              << " itag : " << itag 
              << " config : " << config 
              ; 

    return config ; 
}

   
