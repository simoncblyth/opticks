#include "BConfig.hh"
#include "PLOG.hh"
#include "NLODConfig.hpp"


NLODConfig::NLODConfig(const char* cfg)  
    :
    bconfig(new BConfig(cfg)),
    verbosity(0),
    levels(1)
{
    LOG(info) << "NLODConfig::NLODConfig"
              << " cfg [" << ( cfg ? cfg : "NULL" ) << "]"
              ;

    bconfig->addInt("verbosity", &verbosity );
    bconfig->addInt("levels", &levels );
    bconfig->parse();
}

void NLODConfig::dump(const char* msg) const
{
    bconfig->dump(msg);
}

