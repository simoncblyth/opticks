#pragma once

class G4VSolid ; 
struct nnode ; 

struct convertSphereTest
{
    static nnode* MakePhiMask( float radius, float deltaPhi, float centerPhi );
    nnode* convertSphereLucas() ; 

    G4VSolid* m_solid ; 
    const char*     m_name ; 

};
