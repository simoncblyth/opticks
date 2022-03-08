#include "SEnabled.hh"
#include "OPTICKS_LOG.hh"
#include "CSGSelection.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEnabled<64>*  mm = new SEnabled<64>("10,20,30,-1") ; 
    SEnabled<512>* lv = new SEnabled<512>("100,200,300,-1,-2,-512") ; 

    CSGFoundry* src = nullptr ; 
    CSGFoundry* dst = CSGSelection::Apply(src, mm, lv ); 
    LOG(info) << " dst " << dst ;  

    return 0 ;  
}
