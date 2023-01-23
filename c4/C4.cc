
#include <sstream>

#include "SLOG.hh"
#include "CSGFoundry.h"
#include "C4.hh"
#include "SSim.hh"
#include "U4Tree.h"


const U4SensorIdentifier* C4::SensorIdentifier = nullptr ; 
void C4::SetSensorIdentifier( const U4SensorIdentifier* sid ){ SensorIdentifier = sid ; }  // static 


CSGFoundry* C4::Translate(const G4VPhysicalVolume* const top )
{
    C4 c4(top) ; 
    std::cout << c4.desc() ; 
    return c4.fd ; 
}

C4::C4(const G4VPhysicalVolume* const top_)
    :
    top(top_),
    sim(SSim::Get()),
    st( sim ? sim->get_tree() : nullptr),
    tr(U4Tree::Create(st, top, SensorIdentifier)),
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
    ss << "C4::desc"   
       << " top " << ( top ? "Y" : "N" ) 
       << " sim " << ( sim ? "Y" : "N" )
       << " st " << ( st ? "Y" : "N" )
       << " tr " << ( tr ? "Y" : "N" )
       << " fd " << ( fd ? "Y" : "N" )
       << std::endl 
       ;

    std::string str = ss.str(); 
    return str ; 
}


