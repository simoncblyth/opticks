#pragma once

#include <string>
#include "GGEO_API_EXPORT.hh"

struct nnode ; 
class NCSG ; 


struct GGEO_API GSolidRec
{
     GSolidRec( const nnode* raw_, const nnode* balanced_,  const NCSG* csg_, unsigned soIdx_, unsigned lvIdx_ ); 
     ~GSolidRec();

     std::string desc() const ;

     const nnode* raw ;
     const nnode* balanced ;
     const NCSG* csg ;
     unsigned soIdx ;
     unsigned lvIdx ; 

};
