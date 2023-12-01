// ~/opticks/sysrap/tests/SEvt_test.sh 

#include "OpticksGenstep.h"
#include "SEvt.hh"

struct SEvt_test
{
    static constexpr const int M = 1000000 ; 

    static void AddGenstep(); 
    static void GetNumHit(); 
    static void RUN_META(); 
    static void SetRunProf(); 

};



void SEvt_test::AddGenstep()
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
}

void SEvt_test::GetNumHit()
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
}

void SEvt_test::RUN_META()
{
    NP* m = SEvt::RUN_META ; 

    std::cout 
        << ( m ? m->sstr() : "-" ) 
        << std::endl 
        << ( m ? m->meta : "-" ) 
        << std::endl
        ;   
}

void SEvt_test::SetRunProf()
{
    std::cout << "test_SetRunProf" << std::endl ; 
    SEvt::SetRunProf("test_SetRunProf_0"); 
    SEvt::SetRunProf("test_SetRunProf_1"); 
    std::cout << SEvt::RUN_META->meta ; 
}



// when tests start to need logging, switch to SEvtTest.cc

int main()
{
    SEvt::Create_EGPU() ; 

    SEvt_test::AddGenstep(); 
    SEvt_test::GetNumHit();  
    SEvt_test::RUN_META(); 
    SEvt_test::SetRunProf(); 

    return 0 ; 
}
// ~/opticks/sysrap/tests/SEvt_test.sh 

