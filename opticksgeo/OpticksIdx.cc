
#include <string>
#include <map>

#include "PhotonsNPY.hpp"
#include "HitsNPY.hpp"
#include "RecordsNPY.hpp"
#include "BoundariesNPY.hpp"
#include "SequenceNPY.hpp"
#include "Types.hpp"
#include "Timer.hpp"

#include "GGeo.hh"
#include "GItemIndex.hh"

#include "Opticks.hh"
#include "OpticksIdx.hh"
#include "OpticksHub.hh"
#include "OpticksEvent.hh"

#include "PLOG.hh"



#define TIMER(s) \
    { \
       if(m_evt)\
       {\
          Timer& t = *(m_evt->getTimer()) ;\
          t((s)) ;\
       }\
       else if(m_opticks) \
       {\
          Timer& t = *(m_opticks->getTimer()) ;\
          t((s)) ;\
       }\
    }




OpticksIdx::OpticksIdx(OpticksHub* hub)
   :
   m_hub(hub), 
   m_opticks(hub->getOpticks()),
   m_evt(hub->getEvent())           // usually starts NULL
{
}

void OpticksIdx::setEvent(OpticksEvent* evt)
{
   m_evt = evt ; 
}


GItemIndex* OpticksIdx::makeHistoryItemIndex()
{
    Index* seqhis_ = m_evt->getHistoryIndex() ;
    if(!seqhis_)
    {
         LOG(warning) << "OpticksIdx::makeHistoryItemIndex NULL seqhis" ;
         return NULL ; 
    }
 
    OpticksAttrSeq* qflg = m_hub->getFlagNames();
    //qflg->dumpTable(seqhis, "OpticksIdx::makeHistoryItemIndex seqhis"); 

    GItemIndex* seqhis = new GItemIndex(seqhis_) ;  
    seqhis->setTitle("Photon Flag Sequence Selection");
    seqhis->setHandler(qflg);
    seqhis->formTable();

    return seqhis ; 
}

GItemIndex* OpticksIdx::makeMaterialItemIndex()
{
    Index* seqmat_ = m_evt->getMaterialIndex() ;
    if(!seqmat_)
    {
         LOG(warning) << "OpticksIdx::makeMaterialItemIndex NULL seqmat" ;
         return NULL ; 
    }
 
    OpticksAttrSeq* qmat = m_hub->getMaterialNames();

    GItemIndex* seqmat = new GItemIndex(seqmat_) ;  
    seqmat->setTitle("Photon Material Sequence Selection");
    seqmat->setHandler(qmat);
    seqmat->formTable();

    return seqmat ; 
}

GItemIndex* OpticksIdx::makeBoundaryItemIndex()
{
    Index* bndidx_ = m_evt->getBoundaryIndex();
    if(!bndidx_)
    {
         LOG(warning) << "OpticksIdx::makeBoundaryItemIndex NULL bndidx" ;
         return NULL ; 
    }
 
    OpticksAttrSeq* qbnd = m_hub->getBoundaryNames();
    //qbnd->dumpTable(bndidx, "OpticksIdx::makeBoundariesItemIndex bndidx"); 

    GItemIndex* boundaries = new GItemIndex(bndidx_) ;  
    boundaries->setTitle("Photon Termination Boundaries");
    boundaries->setHandler(qbnd);
    boundaries->formTable();

    return boundaries ; 
}
 


void OpticksIdx::indexEvtOld()
{
    if(!m_evt) return ; 

    // TODO: wean this off use of Types, for the new way (GFlags..)
    Types* types = m_opticks->getTypes();
    Typ* typ = m_opticks->getTyp();

    NPY<float>* ox = m_evt->getPhotonData();

    if(ox && ox->hasData())
    {
        PhotonsNPY* pho = new PhotonsNPY(ox);   // a detailed photon/record dumper : looks good for photon level debug 
        pho->setTypes(types);
        pho->setTyp(typ);
        m_evt->setPhotonsNPY(pho);

        GGeo* ggeo = m_hub->getGGeo();

        if(!ggeo) LOG(fatal) << "OpticksIdx::indexEvtOld" 
                             << " MUST OpticksHub::loadGeometry before OpticksIdx::indexEvtOld "
                             ;

        assert(ggeo);
        HitsNPY* hit = new HitsNPY(ox, ggeo->getSensorList());
        m_evt->setHitsNPY(hit);
    }

    NPY<short>* rx = m_evt->getRecordData();

    if(rx && rx->hasData())
    {
        RecordsNPY* rec = new RecordsNPY(rx, m_evt->getMaxRec(), m_evt->isFlat());
        rec->setTypes(types);
        rec->setTyp(typ);
        rec->setDomains(m_evt->getFDomain()) ;

        PhotonsNPY* pho = m_evt->getPhotonsNPY();
        if(pho)
        {
            pho->setRecs(rec);
        }
        m_evt->setRecordsNPY(rec);
    }

    TIMER("indexEvtOld"); 
}



void OpticksIdx::indexBoundariesHost()
{
    // Indexing the final signed integer boundary code (p.flags.i.x = prd.boundary) from optixrap-/cu/generate.cu
    // see also opop-/OpIndexer::indexBoundaries for GPU version of this indexing 
    // also see optickscore-/Indexer for another CPU version 

    if(!m_evt) return ;

    NPY<float>* dpho = m_evt->getPhotonData();
    if(dpho && dpho->hasData())
    {
        // host based indexing of unique material codes, requires downloadEvt to pull back the photon data
        LOG(info) << "OpticksIdx::indexBoundaries host based " ;
        std::map<unsigned int, std::string> boundary_names = m_hub->getBoundaryNamesMap();
        BoundariesNPY* bnd = new BoundariesNPY(dpho);
        bnd->setBoundaryNames(boundary_names);
        bnd->indexBoundaries();
        m_evt->setBoundariesNPY(bnd);
    }
    else
    {
        LOG(warning) << "OpticksIdx::indexBoundariesHost dpho NULL or no data " ;
    }

    TIMER("indexBoundariesHost");
}





