#pragma once
/**
U4Solid.h : Convert G4VSolid CSG trees into transient snd<double> trees
========================================================================

Canonical usage from U4Tree::initSolid

Looks like can base U4Solid on X4Solid with fairly minor changes:

1. NNode swapped to snd<double>
2. no equivalent to X4SolidBase consolidate the base
3. header-only U4Entity.h based on X4Entity.{hh,cc}   




X4Solid
    from G4VSolid into NNode 

X4SolidBase
    does little, G4 params, mainly placeholder convert methods that assert

X4Entity
    create headeronly U4Entity.h from this 

NNode
    organic mess that need to leave behind 

U4SolidTree
    some overlap with X4Entity, tree mechanics might be better that X4Solid 

**/


#include "snd.h"

struct U4Solid
{
    static snd<double>* Convert( const G4VSolid* so ) ; 
};


inline snd<double>* U4Solid::Convert( const G4VSolid* so )
{
    return nullptr ; 
}
