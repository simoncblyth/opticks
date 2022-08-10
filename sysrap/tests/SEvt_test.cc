// ./SEvt_test.sh 

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
   std::cout << SEvt::Get()->desc() << std::endl ; 
}

void test_GetNumHit()
{
    unsigned num_hit = SEvt::GetNumHit(); 
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


int main()
{
   SEvt evt ; 

   /*
   test_AddGenstep(); 
   */

   test_GetNumHit();  


   return 0 ; 
}

