#include <iostream>
#include "SEvt.hh"
#include "NPFold.h"
#include "spath.h"

int main()
{
    SEvt* a = SEvt::LoadAbsolute("$AFOLD") ; 
    NPFold* af = a ? a->fold : nullptr  ;

    SEvt* b = SEvt::LoadAbsolute("$BFOLD") ; 
    NPFold* bf = b ? b->fold : nullptr  ;
 
    /*
    std::cout 
         << "a.desc" 
         << ( a ? a->desc() : "-" )
         << std::endl
         << "b.desc" 
         << ( b ? b->desc() : "-" )
         << std::endl
         ;
    */

    std::cout 
         << "af.desc" 
         << std::endl
         << ( af ? af->desc() : "-" )
         << std::endl
         << "bf.desc" 
         << std::endl
         << ( bf ? bf->desc() : "-" )
         << std::endl
         ; 


    return 0 ; 
}
