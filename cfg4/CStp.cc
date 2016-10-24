#include <sstream>

#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "CStp.hh"

#include "Format.hh"


CStp::CStp(const G4Step* step, int step_id, const G4ThreeVector& origin) 
   :
   m_step(step),
   m_step_id(step_id),
   m_origin(origin)
{
}




std::string CStp::description()
{
    std::stringstream ss ; 

    ss << "[" 
       << std::setw(4) << m_step_id 
       << "]"       
       << ::Format(m_step, m_origin, "Stp" ) 
       ;

    return ss.str(); 
}

