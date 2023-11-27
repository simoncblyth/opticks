#include <cstring>
#include <sstream>

#include "SGeo.hh"
#include "SEventConfig.hh"
#include "SLOG.hh"

const plog::Severity SGeo::LEVEL = SLOG::EnvLevel("SGeo", "DEBUG"); 


std::string SGeo::Desc() 
{
    const char* outfold = SEventConfig::OutFold() ; 

    std::stringstream ss ; 
    ss << "SGeo::Desc" << std::endl 
       << " SEventConfig::OutFold() " << ( outfold ? outfold : "-" ) << std::endl 
       ;

    std::string s = ss.str(); 
    return s ; 
}


SGeo* SGeo::INSTANCE = nullptr ; 
SGeo* SGeo::Get(){ return INSTANCE ; }
SGeo::SGeo()
{
    INSTANCE = this ; 
} 



