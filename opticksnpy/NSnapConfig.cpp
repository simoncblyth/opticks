#include <sstream>

#include "BConfig.hh"
#include "PLOG.hh"
#include "NSnapConfig.hpp"




NSnapConfig::NSnapConfig(const char* cfg)  
    :
    bconfig(new BConfig(cfg)),
    verbosity(0),
    steps(10),
    eyestartz(0.85),
    eyestopz(0.75),
    prefix("/tmp/snap"),
    postfix(".ppm")
{
    LOG(info) << "NSnapConfig::NSnapConfig"
              << " cfg [" << ( cfg ? cfg : "NULL" ) << "]"
              ;

    // TODO: incorp the help strings into the machinery and include in dumping 

    bconfig->addInt("verbosity", &verbosity );
    bconfig->addInt("steps", &steps );
    bconfig->addFloat("eyestartz", &eyestartz );
    bconfig->addFloat("eyestopz", &eyestopz );
    bconfig->addString("prefix", &prefix );
    bconfig->addString("postfix", &postfix );

    bconfig->parse();
}

void NSnapConfig::dump(const char* msg) const
{
    bconfig->dump(msg);
}


std::string NSnapConfig::getSnapPath(unsigned index)
{
    std::stringstream ss ;
    ss <<  prefix << index << postfix  ; 
    return ss.str();
}


