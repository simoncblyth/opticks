#include <iomanip>
#include <iostream>
#include <cassert>

#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4MaterialPropertyVector.hh"

#include "SDir.h"
#include "SOpticksResource.hh"
#include "SPath.hh"
#include "SStr.hh"
#include "NP.hh"

#include "U4.hh" 

G4MaterialPropertyVector* U4::MakeProperty(const NP* a)  // static
{
    std::vector<double> d, v ; 
    a->psplit<double>(d,v);   // split array into domain and values 
    assert(d.size() == v.size() && d.size() > 1 );  

    G4MaterialPropertyVector* mpv = new G4MaterialPropertyVector(d.data(), v.data(), d.size() );  
    return mpv ; 
}


G4MaterialPropertiesTable*  U4::MakeMaterialPropertiesTable_FromProp( 
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


G4MaterialPropertiesTable* U4::MakeMaterialPropertiesTable(const char* reldir, const char* keys_, char delim )
{
    std::vector<std::string> keys ; 
    SStr::Split( keys_, delim, keys ); 

    const char* idpath = SOpticksResource::IDPath(); 

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    for(unsigned i=0 ; i < keys.size() ; i++)
    {
        const std::string& key_ = keys[i]; 
        const char* key = key_.c_str(); 
        std::string name = SStr::Format("%s.npy", key );   
        NP* a = NP::Load(idpath, reldir, name.c_str() );         
        assert(a); 

        std::cout 
             << " key " << std::setw(15) << key 
             << " a.desc " << a->desc() 
             << std::endl 
             ; 

        G4MaterialPropertyVector* v = MakeProperty(a); 
        mpt->AddProperty(key, v);    
    }
    return mpt ; 
} 


G4MaterialPropertiesTable* U4::MakeMaterialPropertiesTable(const char* reldir)
{
    const char* idpath = SOpticksResource::IDPath(); 
    const char* matdir = SPath::Resolve(idpath, reldir, NOOP); 

    std::vector<std::string> names ; 
    SDir::List(names, matdir, ".npy" ); 

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    for(unsigned i=0 ; i < names.size() ; i++)
    {
        const std::string& name_ = names[i]; 
        const char* name = name_.c_str(); 
        const char* key = SStr::HeadFirst(name, '.'); 

        std::cout << " name " << name << std::endl ; 

        NP* a = NP::Load(idpath, reldir, name  );         
        assert(a); 
        assert(a->has_shape(-1,2) && a->ebyte == 8); 
        double* values = a->values<double>() ; 

        bool is_cprop = a->has_shape(2,2) ; 
        bool is_prop = a->shape[0] > 2 ; 
        bool is_skip = is_cprop == false && is_prop == false ; 

        double value = is_cprop ? values[1] : 0. ; 

        std::cout 
             << " name " << std::setw(15) << name 
             << " key " << std::setw(15) << key
             << " is_cprop " << is_cprop
             << " is_prop " << is_prop
             << " is_skip " << is_skip
             << " value " << value 
             << " a.sstr " << a->sstr() 
             << " a.desc " << a->desc() 
             << std::endl 
             ; 

        if(is_cprop)
        {
             assert( values[1] == values[3] );         
             mpt->AddConstProperty(key, value ); 
        }
        else if(is_prop)
        {
            G4MaterialPropertyVector* v = MakeProperty(a); 
            mpt->AddProperty(key, v);    
        }
 
          
    }
    return mpt ; 
} 





G4Material* U4::MakeWater(const char* name)  // static
{
    G4double a, z, density;
    G4int nelements;
    G4Element* O = new G4Element("Oxygen"  , "O", z=8 , a=16.00*CLHEP::g/CLHEP::mole);
    G4Element* H = new G4Element("Hydrogen", "H", z=1 , a=1.01*CLHEP::g/CLHEP::mole);
    G4Material* mat = new G4Material(name, density= 1.0*CLHEP::g/CLHEP::cm3, nelements=2);
    mat->AddElement(H, 2);
    mat->AddElement(O, 1);
    return mat ; 
}

G4Material* U4::MakeMaterial(const G4MaterialPropertyVector* rindex, const char* name)  // static
{
    G4Material* mat = MakeWater(name); 
    G4MaterialPropertiesTable* mpt = MakeMaterialPropertiesTable_FromProp("RINDEX", rindex) ;
    mat->SetMaterialPropertiesTable(mpt) ;
    return mat ;
}

G4Material* U4::MakeMaterial(const char* name, const char* reldir, const char* props )
{
    G4Material* mat = MakeWater(name); 
    G4MaterialPropertiesTable* mpt = MakeMaterialPropertiesTable(reldir, props, ','); 
    mat->SetMaterialPropertiesTable(mpt) ;
    return mat ;
}

G4Material* U4::MakeMaterial(const char* name, const char* reldir )
{
    G4Material* mat = MakeWater(name); 
    G4MaterialPropertiesTable* mpt = MakeMaterialPropertiesTable(reldir); 
    mat->SetMaterialPropertiesTable(mpt) ;
    return mat ;
}

G4Material* U4::MakeScintillator()
{
    //G4Material* mat = MakeMaterial("LS", "GScintillatorLib/LS_ori", "RINDEX,FASTCOMPONENT,SLOWCOMPONENT,REEMISSIONPROB,GammaCONSTANT,OpticalCONSTANT");  
    //G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable(); 
    //mpt->AddConstProperty("SCINTILLATIONYIELD", 1000.f ); 

    G4Material* mat = MakeMaterial("LS", "GScintillatorLib/LS_ori");  

    return mat ; 
}



