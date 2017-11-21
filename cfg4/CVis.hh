#pragma once

class G4VisAttributes ;

#include "CFG4_API_EXPORT.hh"

class CFG4_API CVis
{
    public:
        static G4VisAttributes* MakeInvisible() ;
        static G4VisAttributes* MakeAtt(float r, float g, float b, bool wire) ;
   
};
 
