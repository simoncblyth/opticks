#include <sstream>

#include "BConfig.hh"
#include "PLOG.hh"
#include "NSnapConfig.hpp"

const plog::Severity NSnapConfig::LEVEL = debug ; 


NSnapConfig::NSnapConfig(const char* cfg)  
    :
    bconfig(new BConfig(cfg)),
    verbosity(0),
    steps(10),
    fmtwidth(5),
    eyestartx(-0.f),   // -ve zero on startx,y,z indicates leave asis, see OpTracer::snap
    eyestarty(-0.f),
    eyestartz(0.f),
    eyestopx(-0.f),
    eyestopy(-0.f),
    eyestopz(1.f),
    prefix("$TMP/snap"),
    postfix(".ppm")
{
    LOG(LEVEL)
              << " cfg [" << ( cfg ? cfg : "NULL" ) << "]"
              ;

    // TODO: incorp the help strings into the machinery and include in dumping 

    bconfig->addInt("verbosity", &verbosity );
    bconfig->addInt("steps", &steps );
    bconfig->addInt("fmtwidth", &fmtwidth );

    bconfig->addFloat("eyestartx", &eyestartx );
    bconfig->addFloat("eyestopx", &eyestopx );

    bconfig->addFloat("eyestarty", &eyestarty );
    bconfig->addFloat("eyestopy", &eyestopy );

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

std::string NSnapConfig::SnapIndex(unsigned index, unsigned width)
{
    std::stringstream ss ;
    ss 
       << std::setw(width) 
       << std::setfill('0')
       << index 
       ;
    return ss.str();
}


std::string NSnapConfig::getSnapPath(unsigned index)
{
    std::stringstream ss ;
    ss << prefix 
       << SnapIndex(index, fmtwidth)
       << postfix 
       ; 

    return ss.str();
}

std::string NSnapConfig::desc() const 
{
    return bconfig->desc(); 
}
