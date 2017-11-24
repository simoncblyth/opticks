#include <sstream>

#include "BConfig.hh"
#include "PLOG.hh"
#include "NEmitConfig.hpp"


//const char* NEmitConfig::DEFAULT = "photons=100000,wavelength=480,time=0.1,weight=1.0,posdelta=0.0,sheetmask=0" ; 
const char* NEmitConfig::DEFAULT = "photons:100000,wavelength:480,time:0.1,weight:1.0,posdelta:0.0,sheetmask:0x3f,umin:0,umax:1,vmin:0,vmax:1" ; 

NEmitConfig::NEmitConfig(const char* cfg)  
    :
    bconfig(new BConfig(cfg ? cfg : DEFAULT,',',":")),
    verbosity(0),
    photons(100),
    wavelength(400),
    time(0.01f),
    weight(1.0f),
    posdelta(0.f),
    sheetmask("0xffff"),
    umin(0.f),
    umax(1.f),
    vmin(0.f),
    vmax(1.f)
{
    LOG(debug) << "NEmitConfig::NEmitConfig"
              << " cfg [" << ( cfg ? cfg : "NULL" ) << "]"
              ;

    bconfig->addInt("verbosity", &verbosity );
    bconfig->addInt("photons", &photons );
    bconfig->addInt("wavelength", &wavelength );
    bconfig->addFloat("time", &time );
    bconfig->addFloat("weight", &weight );
    bconfig->addFloat("posdelta", &posdelta );
    bconfig->addString("sheetmask", &sheetmask );
    bconfig->addFloat("umin", &umin );
    bconfig->addFloat("umax", &umax );
    bconfig->addFloat("vmin", &vmin );
    bconfig->addFloat("vmax", &vmax );

    bconfig->parse();


    assert( umin >= 0 && umin <= 1.) ;
    assert( umax >= 0 && umax <= 1.) ;
    assert( vmin >= 0 && umin <= 1.) ;
    assert( vmax >= 0 && vmax <= 1.) ;

    assert( umax >= umin );
    assert( vmax >= vmin );

}


std::string NEmitConfig::desc() const 
{
    std::stringstream ss ;
    ss << bconfig->desc() ; 
    return ss.str();
}


void NEmitConfig::dump(const char* msg) const
{
    LOG(info) << bconfig->cfg ; 
    bconfig->dump(msg);
}


