// ~/opticks/sysrap/tests/SEvt_test.sh 

#include "OpticksGenstep.h"
#include "ssys.h"
#include "SEvt.hh"

struct SEvt_test
{
    static const char* TEST ; 
    static constexpr const int M = 1000000 ; 
    

    static int AddGenstep(); 
    static int GetNumHit(); 
    static int RUN_META(); 
    static int SetRunProf(); 

    static int Main();   
};


const char* SEvt_test::TEST = ssys::getenvvar("TEST", "ALL") ; 

int SEvt_test::AddGenstep()
{
    for(unsigned i=0 ; i < 10 ; i++)
    {
        quad6 q ; 
        q.set_numphoton(1000) ; 
        unsigned gentype = i % 2 == 0 ? OpticksGenstep_SCINTILLATION : OpticksGenstep_CERENKOV ;  
        q.set_gentype(gentype); 

        SEvt::AddGenstep(q);    
    }
    std::cout << SEvt::Get_EGPU()->desc() << std::endl ; 
    return 0 ; 
}

int SEvt_test::GetNumHit()
{
    unsigned num_hit = SEvt::GetNumHit(SEvt::EGPU); 
    unsigned UNDEF = ~0u ; 

    std::cout 
        << " num_hit " << num_hit 
        << " num_hit " << std::hex << num_hit << std::dec 
        << " is_undef " << ( num_hit == UNDEF ? "Y" : "N" )
        << " int(num_hit) " << int(num_hit) 
        << " int(UNDEF) " << int(UNDEF) 
        << std::endl
        ; 

    for(int idx=0 ; idx < int(num_hit) ; idx++) std::cout << idx << std::endl ;  
    // when num_hit is UNDEF the below loop does nothing as int(UNDEF) is -1 
    return 0 ; 
}

int SEvt_test::RUN_META()
{
    NP* m = SEvt::RUN_META ; 

    std::cout 
        << ( m ? m->sstr() : "-" ) 
        << std::endl 
        << ( m ? m->meta : "-" ) 
        << std::endl
        ;   
    return 0 ; 
}

int SEvt_test::SetRunProf()
{
    std::cout << "test_SetRunProf" << std::endl ; 
    SEvt::SetRunProf("test_SetRunProf_0"); 
    SEvt::SetRunProf("test_SetRunProf_1"); 
    std::cout << SEvt::RUN_META->meta ; 
    return 0 ; 
}


int SEvt_test::Main()
{
    SEvt::Create_EGPU() ; 
    bool ALL = strcmp(TEST, "ALL") == 0 ; 

    int rc = 0 ; 
    if(ALL||0==strcmp(TEST,"AddGenstep")) rc+=AddGenstep(); 
    if(ALL||0==strcmp(TEST,"GetNumHit"))  rc+=GetNumHit(); 
    if(ALL||0==strcmp(TEST,"RUN_META"))   rc+=RUN_META(); 
    if(ALL||0==strcmp(TEST,"SetRunProf")) rc+=SetRunProf(); 
    return rc ; 
}

// when tests start to need logging, switch to SEvtTest.cc
int main()
{
    return SEvt_test::Main(); 
}

// ~/opticks/sysrap/tests/SEvt_test.sh 

