#include "X4ThreeVector.hh"

std::string X4ThreeVector::Code( const G4ThreeVector& v, const char* identifier )
{
    std::stringstream ss ; 
    ss << "G4ThreeVector"
       << ( identifier == NULL ? "" : " " ) 
       << ( identifier == NULL ? "" : identifier ) 
       << std::fixed
       << "("
       << v.x()
       << ","
       << v.y()
       << ","
       << v.z()
       << ")"
       << ( identifier == NULL ? "" : ";" )
       ; 

    return ss.str(); 
}


