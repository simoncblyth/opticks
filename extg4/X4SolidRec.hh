#pragma once

#include <string>
#include "X4_API_EXPORT.hh"

class G4VSolid ; 
struct nnode ; 
class NCSG ; 


struct X4_API X4SolidRec
{
     X4SolidRec( const G4VSolid* solid_,  const nnode* raw_, const nnode* balanced_,  const NCSG* csg_, unsigned soIdx_, unsigned lvIdx_ ); 
     ~X4SolidRec();

     std::string desc() const ;

     const G4VSolid* solid ; 
     const nnode* raw ;
     const nnode* balanced ;

     const NCSG* csg ;
     unsigned soIdx ;
     unsigned lvIdx ; 

};
