

#include "SOpticksResource.hh"
#include "SSys.hh"
#include "SSim.hh"
#include "SPath.hh"
#include "NP.hh"
#include "sdigest.h"


#include "U4Material.hh"
#include "U4PhysicsVector.h"
#include "G4Material.hh"
#include "Randomize.hh"

#include "OPTICKS_LOG.hh"


void test_MakeMaterial()
{
    const NP* a = SOpticksResource::IDLoad("GScintillatorLib/LS_ori/RINDEX.npy"); 
    G4MaterialPropertyVector* v = U4Material::MakeProperty(a) ;  
    G4Material* mat = U4Material::MakeMaterial(v) ;  

    LOG(info) << "mat " << mat ; 
    G4cout << "mat " << *mat << std::endl ; 
}

void test_MakeMaterialPropertiesTable()
{
    G4MaterialPropertiesTable* mpt = U4Material::MakeMaterialPropertiesTable("GScintillatorLib/LS_ori", "FASTCOMPONENT,SLOWCOMPONENT,REEMISSIONPROB", ',' ) ; 
    std::cout << " mpt " << mpt << std::endl ; 
}

void test_MakeScintillator()
{
    G4Material* mat = U4Material::MakeScintillator(); 
    G4cout << "mat " << *mat << std::endl ; 
}

void test_GetProperty()
{
    G4Material* mat = U4Material::MakeScintillator(); 

    std::vector<std::string> names = {"GammaCONSTANT", "AlphaCONSTANT", "NeutronCONSTANT", "OpticalCONSTANT" } ; 

    for(unsigned i=0 ; i < names.size() ; i++)
    {
        const char* name = names[i].c_str() ;
        G4MaterialPropertyVector* prop = U4Material::GetProperty(mat, name) ; 
        if(prop == nullptr) continue ; 

        size_t len = prop->GetVectorLength();      

        LOG(info) << name << " len: " << len << std::endl << "[" << std::endl << *prop << "]" << std::endl ;  
    }
}


void test_NumVec()
{
    G4MaterialPropertyVector* prop = U4Material::GetProperty(U4Material::MakeScintillator(), "GammaCONSTANT") ; 
    if(prop == nullptr) return ; 

    G4int NumTracks = 1000000 ;

    G4int nscnt = prop->GetVectorLength();
    std::vector<G4int> NumVec(nscnt);
    NumVec.clear();
    for(G4int i = 0 ; i < NumTracks ; i++){
       G4double p = G4UniformRand();
       G4double p_count = 0; 
       for(G4int j = 0 ; j < nscnt; j++) 
       {    
            p_count += (*prop)[j] ;
            if( p < p_count ){
               NumVec[j]++ ;
               break;
            }    
        }    
     }    

    int tot_NumVec = 0 ; 
    double tot_prop = 0 ; 

    std::cout << " split the NumTracks:" << NumTracks << " into nscnt:" << nscnt << " groups, with each group having a single scintillationTime param " << std::endl ;  

    for(int i=0 ; i < nscnt ; i++) 
    { 
        tot_NumVec += NumVec[i] ; 
        tot_prop += (*prop)[i] ; 
        std::cout 
            << " i " << std::setw(5) << i 
            << " NumVec[i] "  << std::setw(7) << NumVec[i] 
            << " prop[i] " << std::setw(10) << std::fixed << std::setprecision(4) << (*prop)[i] 
            << " prop.Energy(i) " << std::setw(10) << std::fixed << std::setprecision(4) << prop->Energy(i) << " (scintillationTime) " 
            << std::endl 
            ; 
    }

    std::cout 
        << "tot" << std::setw(5) << ""
        << " NumVec[ ] " << std::setw(7) << tot_NumVec
        << " prop[ ] " << std::setw(10) << std::fixed << std::setprecision(4) << tot_prop
        << std::endl 
        ; 

}




void test_Load_0()
{
    const NP* a = SOpticksResource::IDLoad("GScintillatorLib/LS_ori/RINDEX.npy"); 
    const NP* b = NP::Load(SPath::Resolve("$IDPath/GScintillatorLib/LS_ori/RINDEX.npy", NOOP)); 

    std::cout << " a " << a->sstr() << " digest " << sdigest::Item(a) << std::endl ; 
    std::cout << " b " << b->sstr() << " digest " << sdigest::Item(b) << std::endl ; 

    a->dump(); 

}



void test_LoadOri_name()
{
    G4Material* mat = U4Material::LoadOri("Water"); 
    LOG(info) << " mat " << mat ;  
}

void test_ListOri()
{
    std::vector<std::string> names ; 
    U4Material::ListOri(names); 
}

void test_LoadOri()
{
    U4Material::LoadOri(); 
    std::cout << U4Material::DescMaterialTable() ; 
}

void test_LoadOri_remove_material_property()
{
    U4Material::LoadOri(); 
    std::cout << U4Material::DescMaterialTable() ; 

    G4Material* mat = G4Material::GetMaterial("Rock"); 

    LOG(info) << "before removal " << std::endl << U4Material::DescPropertyNames( mat ) ; 

    U4Material::RemoveProperty( "RINDEX", mat );  
    U4Material::RemoveProperty( "RINDEX", mat );  
    U4Material::RemoveProperty( "RINDEX", mat );  

    LOG(info) << "after removal " << std::endl << U4Material::DescPropertyNames( mat ) ; 
}

void test_LoadBnd()
{
    U4Material::LoadBnd(); 

    //std::cout << U4Material::DescProperty("Water", "GROUPVEL") << std::endl ;  
    //std::cout << U4Material::DescProperty("Air", "GROUPVEL") << std::endl ;  

    //std::cout << U4Material::DescProperty("Water") << std::endl ; 
    std::cout << U4Material::DescProperty() << std::endl ; 
}

void test_BndNames()
{
    const char* path = SPath::Resolve("$CFBASE/CSGFoundry/SSim/bnd_names.txt", NOOP);  
    LOG(info) << " path " << path ; 
}

void test_MakeStandardArray_prop_override()
{
    const char* spec = "Water/RAYLEIGH" ; 

    std::map<std::string, G4PhysicsVector*> prop_override ; 
    prop_override[spec] = U4PhysicsVector::CreateConst(101.); 

    bool do_override = prop_override.count(spec) > 0 ; 

    G4PhysicsVector* prop = do_override ? prop_override[spec] : nullptr ; 
    G4double value = prop ? prop->Value(3.*eV) : 0. ;  

    std::cout 
        << "spec " << spec 
        << " do_override " << (do_override ? "Y" : "N" ) 
        << " value " << value 
        << std::endl 
        ; 

}

void test_MakeStandardArray_override_count()
{
    typedef std::map<std::string,int> SI ; 
    SI oc ; 

    oc["Water/RAYLEIGH"] += 1 ; 
    oc["Water/RAYLEIGH"] += 1 ; 
    oc["vetoWater/RAYLEIGH"] += 1 ; 

    for(SI::const_iterator it=oc.begin() ; it != oc.end() ; it++)
    {
        std::cout << it->first << " : " << it->second << std::endl ; 
    }

}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    
    /*
    test_MakeMaterial(); 
    test_MakeMaterialPropertiesTable(); 
    test_GetProperty(); 
    test_NumVec(); 
    test_MakeScintillator(); 
    test_Load_0(); 
    test_LoadOri_name(); 
    test_ListOri(); 
    test_LoadOri(); 
    test_LoadOri_remove_material_property(); 
    test_LoadBnd(); 
    test_BndNames(); 
    */

    //test_MakeStandardArray_prop_override(); 
    test_MakeStandardArray_override_count(); 


     
    return 0 ; 
}
