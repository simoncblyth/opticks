#include <sstream>

#include "BConfig.hh"
#include "PLOG.hh"
#include "NEmitConfig.hpp"


//const char* NEmitConfig::DEFAULT = "photons=100000,wavelength=480,time=0.1,weight=1.0,posdelta=0.0,sheetmask=0" ; 
const char* NEmitConfig::DEFAULT = "photons:100000,wavelength:480,time:0.1,weight:1.0,posdelta:0.0,sheetmask:0x3f" ; 

NEmitConfig::NEmitConfig(const char* cfg)  
    :
    bconfig(new BConfig(cfg ? cfg : DEFAULT,',',":")),
    verbosity(0),
    photons(100),
    wavelength(400),
    time(0.01f),
    weight(1.0f),
    posdelta(0.f),
    sheetmask("0xffff")
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

    bconfig->parse();
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


