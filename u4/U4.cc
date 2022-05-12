#include "NP.hh"
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4MaterialPropertyVector.hh"
#include "U4.hh" 

G4MaterialPropertyVector* U4::MakeProperty(const NP* a)  // static
{
    std::vector<double> d, v ; 
    a->psplit<double>(d,v);   // split array into domain and values 
    assert(d.size() == v.size() && d.size() > 1 );  

    G4MaterialPropertyVector* mpv = new G4MaterialPropertyVector(d.data(), v.data(), d.size() );  
    return mpv ; 
}


G4MaterialPropertiesTable*  U4::MakeMaterialPropertiesTable( 
     const char* a_key, const G4MaterialPropertyVector* a_prop,
     const char* b_key, const G4MaterialPropertyVector* b_prop,
     const char* c_key, const G4MaterialPropertyVector* c_prop,
     const char* d_key, const G4MaterialPropertyVector* d_prop
)
{
    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    if(a_key && a_prop) mpt->AddProperty(a_key,const_cast<G4MaterialPropertyVector*>(a_prop));    //  HUH: why not const ?
    if(b_key && b_prop) mpt->AddProperty(b_key,const_cast<G4MaterialPropertyVector*>(b_prop));
    if(c_key && c_prop) mpt->AddProperty(c_key,const_cast<G4MaterialPropertyVector*>(c_prop));
    if(d_key && d_prop) mpt->AddProperty(d_key,const_cast<G4MaterialPropertyVector*>(d_prop));
    return mpt ;
}

G4Material* U4::MakeMaterial(const G4MaterialPropertyVector* rindex, const char* name)  // static
{
    // its Water, but that makes no difference for Cerenkov 
    // the only thing that matters us the rindex property
    G4double a, z, density;
    G4int nelements;
    G4Element* O = new G4Element("Oxygen"  , "O", z=8 , a=16.00*CLHEP::g/CLHEP::mole);
    G4Element* H = new G4Element("Hydrogen", "H", z=1 , a=1.01*CLHEP::g/CLHEP::mole);
    G4Material* mat = new G4Material(name, density= 1.0*CLHEP::g/CLHEP::cm3, nelements=2);
    mat->AddElement(H, 2);
    mat->AddElement(O, 1);

    G4MaterialPropertiesTable* mpt = MakeMaterialPropertiesTable("RINDEX", rindex) ;
    mat->SetMaterialPropertiesTable(mpt) ;
    return mat ;
}


