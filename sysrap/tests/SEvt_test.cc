// name=SEvt_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name

#include "OpticksGenstep.h"
#include "SEvt.h"

int main()
{
   SEvt evt ; 

   for(unsigned i=0 ; i < 10 ; i++)
   {
       quad6 q ; 
       q.set_numphoton(1000) ; 
       unsigned gentype = i % 2 == 0 ? OpticksGenstep_SCINTILLATION : OpticksGenstep_CERENKOV ;  
       q.set_gentype(gentype); 
       evt.add(q);    
   }

   std::cout << evt.desc() << std::endl ; 

   return 0 ; 
}

