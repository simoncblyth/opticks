#include "CSGSelection.h"

#include "CSGFoundry.h"
#include "SEnabled.hh"
#include "PLOG.hh"

const plog::Severity CSGSelection::LEVEL = PLOG::EnvLevel("CSGSelection", "DEBUG"); 

CSGFoundry* CSGSelection::Apply( const CSGFoundry* src, const SEnabled<64>* mmidx, const SEnabled<512>* lvidx )
{
    CSGFoundry* dst = new CSGFoundry ; 

    for(unsigned mm=0 ; mm < 64  ; mm++) if(mmidx->isEnabled(mm)) std::cout << "mm " << mm << std::endl ; 
    for(unsigned lv=0 ; lv < 512 ; lv++) if(lvidx->isEnabled(lv)) std::cout << "lv " << lv << std::endl ; 


    return dst ; 
}

