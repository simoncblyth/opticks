#include "PLOG.hh"
#include "GGeo.hh"
#include "GGeoLib.hh"
#include "GParts.hh"
#include "GGeoDump.hh"

GGeoDump::GGeoDump(const GGeo* ggeo_) 
   :
   ggeo(ggeo_),
   geolib(ggeo->getGeoLib())
{
}

void GGeoDump::dump(int repeatIdx, int primIdx, int partIdxRel )
{
    if( repeatIdx > -1 && primIdx > -1 && partIdxRel > -1 )
    {
        dump_(repeatIdx, primIdx, partIdxRel);
    }
    else if( repeatIdx > -1 && primIdx > -1  )
    {
        dump_(repeatIdx, primIdx);
    }
    else if( repeatIdx > -1 )
    {
        dump_(repeatIdx);
    }
    else
    {
        std::string smry = geolib->summary("GGeoDump::dump"); 
        std::cout << smry << std::endl ; 
    }
}


void GGeoDump::dump_(unsigned repeatIdx)
{
    GParts* comp = geolib->getCompositeParts(repeatIdx); 
    unsigned numPrim = comp->getNumPrim();
    LOG(info)
        << " repeatIdx " << repeatIdx 
        << " numPrim " << numPrim
        ;   

    for(unsigned primIdx=0 ; primIdx < numPrim ; primIdx++)
    {   
        dump_(repeatIdx, primIdx); 
    }   
}


void GGeoDump::dump_(unsigned repeatIdx, unsigned primIdx )
{
    GParts* comp = geolib->getCompositeParts(repeatIdx); 
    unsigned numPrim = comp->getNumPrim();
    unsigned numParts = comp->getNumParts(primIdx) ;
    unsigned tranOffset = comp->getTranOffset(primIdx) ;
    unsigned partOffset = comp->getPartOffset(primIdx) ;
    LOG(info) 
        << " repeatIdx " << repeatIdx 
        << " primIdx/numPrim " << primIdx << "/" << numPrim 
        << " numParts " << numParts
        << " tranOffset " << tranOffset
        << " partOffset " << partOffset
        ;   
    for(unsigned partIdxRel=0 ; partIdxRel < numParts ; partIdxRel++ )
    {   
        dump_(repeatIdx, primIdx, partIdxRel); 
    }   
}
void GGeoDump::dump_(unsigned repeatIdx, unsigned primIdx, unsigned partIdxRel )
{
    GParts* comp = geolib->getCompositeParts(repeatIdx); 
    unsigned partOffset = comp->getPartOffset(primIdx) ; 
    unsigned numParts = comp->getNumParts(primIdx) ;
    assert( partIdxRel < numParts );  
    unsigned partIdx = partOffset + partIdxRel ; 
    comp->dumpPart(partIdx); 
}


