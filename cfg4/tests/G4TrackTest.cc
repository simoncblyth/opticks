#include "CTrackInfo.hh"
#include "G4Track.hh"


int main(int argc, char** argv)
{
    CTrackInfo* tkui_0 = new CTrackInfo( 42, 'S', false ); 

    G4Track* trk = new G4Track ; 

    trk->SetUserInformation(tkui_0) ; 

    G4VUserTrackInformation* ui = trk->GetUserInformation(); 

    CTrackInfo* tkui_1 = dynamic_cast<CTrackInfo*>(ui); 

    assert( tkui_1 == tkui_0 ); 


    return 0 ; 
}
