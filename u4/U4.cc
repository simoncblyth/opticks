#include <iomanip>
#include <iostream>
#include <cassert>

#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4MaterialPropertyVector.hh"

#include "G4VParticleChange.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "sscint.h"
#include "OpticksGenstep.h"


#include "SDir.h"
#include "SOpticksResource.hh"
#include "SPath.hh"
#include "SStr.hh"
#include "NP.hh"
#include "PLOG.hh"

#include "U4.hh" 

const plog::Severity U4::LEVEL = PLOG::EnvLevel("U4", "DEBUG"); 


G4MaterialPropertyVector* U4::MakeProperty(const NP* a)  // static
{
    std::vector<double> d, v ; 
    a->psplit<double>(d,v);   // split array into domain and values 
    assert(d.size() == v.size());

    //assert(d.size() > 1 );  // OpticalCONSTANT (scint time fraction property misusage) breaks this  

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

/**
U4::Classify
-------------

Heuristic to distinguish ConstProperty from Property based on array size and content
and identify fractional property. 

C
   ConstProperty
P
   Property
F
   Property that looks like a fractional split with a small number of values summing to 1.

**/

char U4::Classify(const NP* a)
{
    assert(a); 
    assert(a->shape.size() == 2); 
    assert(a->has_shape(-1,2) && a->ebyte == 8 && a->uifc == 'f' ); 
    const double* values = a->cvalues<double>() ; 
    char type = a->has_shape(2,2) && values[1] == values[3] ? 'C' : 'P' ; 

    int numval = a->shape[0]*a->shape[1] ; 
    double fractot = 0. ; 
    if(type == 'P' && numval <= 8) 
    {
        for(int i=0 ; i < numval ; i++) if(i%2==1) fractot += values[i] ;   
        if( fractot == 1. ) type = 'F' ;  
    }  
    return type ; 
}


std::string U4::Desc(const char* key, const NP* a )
{
    char type = Classify(a); 
    const double* values = a->cvalues<double>() ; 
    int numval = a->shape[0]*a->shape[1] ; 
    double value = type == 'C' ? values[1] : 0. ; 

    std::stringstream ss ; 

    ss << std::setw(20) << key 
       << " " << std::setw(10) << a->sstr() 
       << " type " << type
       << " value " << std::setw(10) << std::fixed << std::setprecision(3) << value 
       << " : " 
       ; 

    double tot = 0. ; 
    if(numval <= 8) 
    {
        for(int i=0 ; i < numval ; i++) if(i%2==1) tot += values[i] ;   
        for(int i=0 ; i < numval ; i++) ss << std::setw(10) << std::fixed << std::setprecision(3) << values[i] << " " ;
        ss << " tot: " << std::setw(10) << std::fixed << std::setprecision(3) << tot << " " ; 
    } 
    std::string s = ss.str(); 
    return s ; 
}


G4MaterialPropertiesTable* U4::MakeMaterialPropertiesTable(const char* reldir )
{
    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();

    const char* idpath = SOpticksResource::IDPath(); 
    const char* matdir = SPath::Resolve(idpath, reldir, NOOP); 

    std::vector<std::string> names ; 
    SDir::List(names, matdir, ".npy" ); 

    std::stringstream ss ; 
    ss << "reldir " << reldir << " names " << names.size() << std::endl ; 

    for(unsigned i=0 ; i < names.size() ; i++)
    {
        const char* name = names[i].c_str(); 
        const char* key = SStr::HeadFirst(name, '.'); 

        NP* a = NP::Load(idpath, reldir, name  );         

        char type = Classify(a); 
        double* values = a->values<double>() ; 
        ss << Desc(key, a) << std::endl ;   
        
        switch(type)
        {
            case 'C': mpt->AddConstProperty(key, values[1])    ; break ; 
            case 'P': mpt->AddProperty(key, MakeProperty(a))   ; break ; 
            case 'F': mpt->AddProperty(key, MakeProperty(a))   ; break ; 
        }
    }

    std::string s = ss.str(); 
    
    LOG(LEVEL) << s ; 
    std::cout << s << std::endl ; 
    
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

G4Material* U4::MakeMaterial(const char* name, const char* reldir)
{
    G4Material* mat = MakeWater(name); 
    G4MaterialPropertiesTable* mpt = MakeMaterialPropertiesTable(reldir); 
    mat->SetMaterialPropertiesTable(mpt) ;
    return mat ;
}

G4Material* U4::MakeScintillatorOld()
{
    G4Material* mat = MakeMaterial("LS", "GScintillatorLib/LS_ori", "RINDEX,FASTCOMPONENT,SLOWCOMPONENT,REEMISSIONPROB,GammaCONSTANT,OpticalCONSTANT");  
    G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable(); 
    mpt->AddConstProperty("SCINTILLATIONYIELD", 1000.f );   // ACTUALLY BETTER TO LOAD THIS LIKE THE REST 
    return mat ; 
}

/**
U4::MakeScintillator
----------------------

Creates material with Water G4Element and density and then loads the properties of LS into its MPT 

**/

G4Material* U4::MakeScintillator()
{
    G4Material* mat = MakeMaterial("LS", "GScintillatorLib/LS_ori");  
    return mat ; 
}


G4MaterialPropertyVector* U4::GetProperty(const G4Material* mat, const char* name)
{
    G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable();
    return mpt->GetProperty(name) ; 
}


NP* U4::CollectOpticalSecondaries(const G4VParticleChange* pc )
{
    G4int num = pc->GetNumberOfSecondaries();

    std::cout << "U4::CollectOpticalSecondaries num " << num << std::endl ; 

    NP* p = NP::Make<float>(num, 4, 4); 
    sphoton* pp = (sphoton*)p->bytes() ; 

    for(int i=0 ; i < num ; i++)
    {   
        G4Track* track =  pc->GetSecondary(i) ;
        assert( track->GetParticleDefinition() == G4OpticalPhoton::Definition() );
        const G4DynamicParticle* ph = track->GetDynamicParticle() ;
        const G4ThreeVector& pmom = ph->GetMomentumDirection() ;
        const G4ThreeVector& ppol = ph->GetPolarization() ;
        sphoton& sp = pp[i] ; 

        sp.mom.x = pmom.x(); 
        sp.mom.y = pmom.y(); 
        sp.mom.z = pmom.z(); 

        sp.pol.x = ppol.x(); 
        sp.pol.y = ppol.y(); 
        sp.pol.z = ppol.z(); 
    }
    return p ; 
} 


void U4::CollectGenstep_DsG4Scintillation_r4695( 
     const G4Track* aTrack,
     const G4Step* aStep,
     G4int    numPhotons,
     G4int    scnt,        
     G4double ScintillationTime
    )
{
    G4StepPoint* pPreStepPoint  = aStep->GetPreStepPoint();
    G4StepPoint* pPostStepPoint = aStep->GetPostStepPoint();

    G4ThreeVector x0 = pPreStepPoint->GetPosition();
    G4double      t0 = pPreStepPoint->GetGlobalTime();
    G4ThreeVector deltaPosition = aStep->GetDeltaPosition() ;
    G4double meanVelocity = (pPreStepPoint->GetVelocity()+pPostStepPoint->GetVelocity())/2. ;

    const G4DynamicParticle* aParticle = aTrack->GetDynamicParticle();
    //const G4Material* aMaterial = aTrack->GetMaterial();

    sscint gs = {} ; 

    gs.gentype = OpticksGenstep_DsG4Scintillation_r4695 ;
    gs.trackid = aTrack->GetTrackID() ;
    gs.matline = 0u ; //  aMaterial->GetIndex()   // not used for scintillation
    gs.numphoton = numPhotons ;  

    gs.pos.x = x0.x() ; 
    gs.pos.y = x0.y() ; 
    gs.pos.z = x0.z() ; 
    gs.time = t0 ; 

    gs.DeltaPosition.x = deltaPosition.x() ; 
    gs.DeltaPosition.y = deltaPosition.y() ; 
    gs.DeltaPosition.z = deltaPosition.z() ; 
    gs.step_length = aStep->GetStepLength() ;

    gs.code = aParticle->GetDefinition()->GetPDGEncoding() ;
    gs.charge = aParticle->GetDefinition()->GetPDGCharge() ;
    gs.weight = aTrack->GetWeight() ;
    gs.meanVelocity = meanVelocity ; 

    gs.scnt = scnt ; 
    gs.f41 = 0.f ;  
    gs.f42 = 0.f ;  
    gs.f43 = 0.f ; 

    gs.ScintillationTime = ScintillationTime ;
    gs.f51 = 0.f ;
    gs.f52 = 0.f ;
    gs.f53 = 0.f ;
     

}





