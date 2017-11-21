
#include "G4Colour.hh"
#include "G4VisAttributes.hh"
#include "CVis.hh"


G4VisAttributes* CVis::MakeInvisible()
{
    return new G4VisAttributes(G4VisAttributes::Invisible) ;
}

G4VisAttributes* CVis::MakeAtt(float r, float g, float b, bool wire)
{
     // g4-;g4-cls G4VisAttributes

    G4VisAttributes* att = new G4VisAttributes(G4Colour(r,g,b));
    //att->SetVisibility(true);
    if(wire) att->SetForceWireframe(true);

    //World_log->SetVisAttributes (G4VisAttributes::Invisible);

    return att ; 
}

