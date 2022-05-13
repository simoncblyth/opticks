#include "U4.hh"
#include "SOpticksResource.hh"
#include "G4Material.hh"
#include "Randomize.hh"

#include "OPTICKS_LOG.hh"

void test_MakeMaterial()
{
    const NP* a = SOpticksResource::IDLoad("GScintillatorLib/LS_ori/RINDEX.npy"); 
    G4MaterialPropertyVector* v = U4::MakeProperty(a) ;  
    G4Material* mat = U4::MakeMaterial(v) ;  

    LOG(info) << "mat " << mat ; 
    G4cout << "mat " << *mat << std::endl ; 
}

void test_MakeMaterialPropertiesTable()
{
    G4MaterialPropertiesTable* mpt = U4::MakeMaterialPropertiesTable("GScintillatorLib/LS_ori", "FASTCOMPONENT,SLOWCOMPONENT,REEMISSIONPROB", ',' ) ; 
    std::cout << " mpt " << mpt << std::endl ; 
}

void test_MakeScintillator()
{
    G4Material* mat = U4::MakeScintillator(); 
    G4cout << "mat " << *mat << std::endl ; 
}

void test_GetProperty()
{
    G4Material* mat = U4::MakeScintillator(); 

    std::vector<std::string> names = {"GammaCONSTANT", "AlphaCONSTANT", "NeutronCONSTANT", "OpticalCONSTANT" } ; 

    for(unsigned i=0 ; i < names.size() ; i++)
    {
        const char* name = names[i].c_str() ;
        G4MaterialPropertyVector* prop = U4::GetProperty(mat, name) ; 
        if(prop == nullptr) continue ; 

        size_t len = prop->GetVectorLength();      

        LOG(info) << name << " len: " << len << std::endl << "[" << std::endl << *prop << "]" << std::endl ;  
    }
}


void test_NumVec()
{
    G4MaterialPropertyVector* prop = U4::GetProperty(U4::MakeScintillator(), "GammaCONSTANT") ; 
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






int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    
    /*
    test_MakeMaterial(); 
    test_MakeMaterialPropertiesTable(); 
    test_MakeScintillator(); 
    test_GetProperty(); 
    */
    test_NumVec(); 

     
    return 0 ; 
}
