#include "G4Orb.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"

#include "OpNoviceDetectorConstruction.hh"
#include "LXe_Materials.hh"
#include "SDirect.hh"

#include "X4Sample.hh"

#include "PLOG.hh"



G4VPhysicalVolume* X4Sample::Sample(char c)
{
    G4VPhysicalVolume* top = NULL ;  
    switch(c)
    {
        case 'o': top = X4Sample::OpNovice() ; break ;  
        case 's': top = X4Sample::Simple(c)  ; break ; 
        case 'b': top = X4Sample::Simple(c)  ; break ; 
    }
    assert(top);
    return top ; 
}

G4VPhysicalVolume* X4Sample::Simple(char c)
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

G4VPhysicalVolume* X4Sample::OpNovice()
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



