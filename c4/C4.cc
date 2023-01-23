
#include <sstream>

#include "SSim.hh"
#include "CSGFoundry.h"
#include "C4.hh"

CSGFoundry* C4::Translate(const G4VPhysicalVolume* const top )
{
    C4 c4(top) ; 
    return c4.fd ; 
}


C4::C4(const G4VPhysicalVolume* const top)
    :
    si(SSim::Create()),
    fd(new CSGFoundry)
{
    init(); 
}

void C4::init()
{


}


std::string C4::desc() const 
{
    std::stringstream ss ; 
    ss << "C4::desc" ; 
    std::string str = ss.str(); 
    return str ; 
}


