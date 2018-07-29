#pragma once

#include <string>
#include "X4_API_EXPORT.hh"

class G4VSolid ; 
struct nnode ; 
class NCSG ; 


struct X4_API X4SolidRec
{
     X4SolidRec( const G4VSolid* solid_,  const nnode* node_, const NCSG* csg_, unsigned soIdx_, unsigned lvIdx_ ); 
     std::string desc() const ;

     const G4VSolid* solid ; 
     const nnode* node ;
     const NCSG* csg ;
     unsigned soIdx ;
     unsigned lvIdx ; 

};
