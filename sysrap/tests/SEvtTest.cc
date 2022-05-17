#include "OpticksGenstep.h"
#include "SEvt.hh"

int main()
{
   SEvt evt ; 

   for(unsigned i=0 ; i < 10 ; i++)
   {
       quad6 q ; 
       q.set_numphoton(1000) ; 
       unsigned gentype = i % 2 == 0 ? OpticksGenstep_SCINTILLATION : OpticksGenstep_CERENKOV ;  
       q.set_gentype(gentype); 

       SEvt::AddGenstep(q);    
   }

   std::cout << SEvt::Get()->desc() << std::endl ; 

   return 0 ; 
}

