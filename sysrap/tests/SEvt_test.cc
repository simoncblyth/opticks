// ~/opticks/sysrap/tests/SEvt_test.sh 

#include "OpticksGenstep.h"
#include "SEvt.hh"


void test_AddGenstep()
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

void test_GetNumHit()
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

void test_RUN_META()
{
    NP* m = SEvt::RUN_META ; 

    std::cout 
        << ( m ? m->sstr() : "-" ) 
        << std::endl 
        << ( m ? m->meta : "-" ) 
        << std::endl
        ;   
}

void test_SetRunProf()
{
    std::cout << "test_SetRunProf" << std::endl ; 
    SEvt::SetRunProf("test_SetRunProf_0"); 
    SEvt::SetRunProf("test_SetRunProf_1"); 
    std::cout << SEvt::RUN_META->meta ; 
}



int main()
{
    SEvt* evt = SEvt::Create_EGPU() ; 

    /*
    test_AddGenstep(); 
    test_GetNumHit();  
    test_RUN_META(); 
    */
    test_SetRunProf(); 

    return 0 ; 
}
// ~/opticks/sysrap/tests/SEvt_test.sh 

