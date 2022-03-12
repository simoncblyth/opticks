#include "SEnabled.hh"
#include "SSelector.hh"

SSelector::SSelector()
    :
    emm(new SEnabled<64>()), 
    elv(new SEnabled<512>())
{
}

bool SSelector::isCompoundEnabled( unsigned mmIdx ){ return emm->isEnabled(mmIDx) ; }
bool SSelector::isShapeEnabled(    unsigned lvIdx ){ return elv->isEnabled(lvIdx) ; }


