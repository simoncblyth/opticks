#include "G4Orb.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"

#include "GMaterialLib.hh"

#include "X4PhysicalVolume.hh"
#include "X4MaterialTable.hh"
#include "OpNoviceDetectorConstruction.hh"
#include "LXe_Materials.hh"
#include "Opticks.hh"
#include "SDirect.hh"
#include "OPTICKS_LOG.hh"





G4VPhysicalVolume* construct(char c)
{
    LXe_Materials lm ; 

    G4VSolid* solid = NULL ; 
    switch(c)
    {
       case 'b': solid = new G4Box("World",100.,100.,100.) ; break ;  
       case 's': solid = new G4Orb("World",100.)           ; break ;  
    }

    G4LogicalVolume* lv = new G4LogicalVolume(solid,lm.fAir,"World",0,0,0);

    G4VPhysicalVolume* pv = new G4PVPlacement(0,G4ThreeVector(),lv, "World",0,false,0);

    return pv ;  
}



G4VPhysicalVolume* construct_OpNovice()
{
    G4VPhysicalVolume* top = NULL ; 

    OpNoviceDetectorConstruction ondc ; 

    // redirect cout and cerr from the Construct
    std::stringstream coutbuf;
    std::stringstream cerrbuf;
    {   
       cout_redirect out(coutbuf.rdbuf());
       cerr_redirect err(cerrbuf.rdbuf());

       top = ondc.Construct() ;     
    }   
    std::string _cout = coutbuf.str() ; 
    std::string _cerr = cerrbuf.str() ; 
 
    //LOG(trace) << " cout " << _cout ;
    LOG(trace) << " cerr " << _cerr ;
    assert(top);  

    return top ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //char c = argc > 1 ? *argv[1] : 'o' ; 

    char c = 's' ; 


    // Note that okc.Opticks instanciation is 
    // handled by X4PhysicalVolume ctor in order to
    // set the static geometry key before the instanciation.
   
    // NB cannot grab top via the navigator singleton as not a full Geant4 environment  

    G4VPhysicalVolume* top = NULL ;  
    switch(c)
    {
        case 'o': top = construct_OpNovice() ; break ;  
        case 's': top = construct(c)         ; break ; 
        case 'b': top = construct(c)         ; break ; 
    }
    assert(top);

    GGeo* ggeo = X4PhysicalVolume::Convert(top) ;   
    assert(ggeo);  

    Opticks* ok = Opticks::GetInstance();
    ok->Summary();

    return 0 ; 
}


