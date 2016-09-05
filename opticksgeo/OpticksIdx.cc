
#include <string>
#include <map>

#include "SeqNPY.hpp"
#include "PhotonsNPY.hpp"
#include "HitsNPY.hpp"
#include "RecordsNPY.hpp"
#include "BoundariesNPY.hpp"
#include "SequenceNPY.hpp"
#include "G4StepNPY.hpp"
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
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }



OpticksIdx::OpticksIdx(OpticksHub* hub)
   :
   m_hub(hub), 
   m_opticks(hub->getOpticks()),
   m_seq(NULL)
{
}




GItemIndex* OpticksIdx::makeHistoryItemIndex()
{
    OpticksEvent* evt = m_hub->getEvent();
    Index* seqhis_ = evt->getHistoryIndex() ;
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
    OpticksEvent* evt = m_hub->getEvent();
    Index* seqmat_ = evt->getMaterialIndex() ;
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
    OpticksEvent* evt = m_hub->getEvent();
    Index* bndidx_ = evt->getBoundaryIndex();
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
    OpticksEvent* evt = m_hub->getEvent();
    if(!evt) return ; 

    // TODO: wean this off use of Types, for the new way (GFlags..)
    Types* types = m_opticks->getTypes();
    Typ* typ = m_opticks->getTyp();

    NPY<float>* ox = evt->getPhotonData();

    if(ox && ox->hasData())
    {
        PhotonsNPY* pho = new PhotonsNPY(ox);   // a detailed photon/record dumper : looks good for photon level debug 
        pho->setTypes(types);
        pho->setTyp(typ);
        evt->setPhotonsNPY(pho);

        GGeo* ggeo = m_hub->getGGeo();

        if(!ggeo) LOG(fatal) << "OpticksIdx::indexEvtOld" 
                             << " MUST OpticksHub::loadGeometry before OpticksIdx::indexEvtOld "
                             ;

        assert(ggeo);
        HitsNPY* hit = new HitsNPY(ox, ggeo->getSensorList());
        evt->setHitsNPY(hit);
    }

    NPY<short>* rx = evt->getRecordData();

    if(rx && rx->hasData())
    {
        RecordsNPY* rec = new RecordsNPY(rx, evt->getMaxRec(), evt->isFlat());
        rec->setTypes(types);
        rec->setTyp(typ);
        rec->setDomains(evt->getFDomain()) ;

        PhotonsNPY* pho = evt->getPhotonsNPY();
        if(pho)
        {
            pho->setRecs(rec);
        }
        evt->setRecordsNPY(rec);
    }

    TIMER("indexEvtOld"); 
}




void OpticksIdx::indexSeqHost()
{
    LOG(info) << "OpticksIdx::indexSeqHost" ; 

    OpticksEvent* evt = m_hub->getEvent();
    if(!evt) return ; 

    NPY<unsigned long long>* ph = evt->getSequenceData();

    if(ph && ph->hasData())
    {
        m_seq = new SeqNPY(ph);
        m_seq->dump("OpticksIdx::indexSeqHost");
        std::vector<int> counts = m_seq->getCounts();

        G4StepNPY* g4step = m_hub->getG4Step();
        assert(g4step && "OpticksIdx::indexSeqHost requires G4StepNPY, created in translate"); 
        g4step->checkCounts(counts, "OpticksIdx::indexSeqHost checkCounts"); 

    }
    else
    { 
        LOG(warning) << "OpticksIdx::indexSeqHost requires sequence data hostside " ;      
    }
}




void OpticksIdx::indexBoundariesHost()
{
    // Indexing the final signed integer boundary code (p.flags.i.x = prd.boundary) from optixrap-/cu/generate.cu
    // see also opop-/OpIndexer::indexBoundaries for GPU version of this indexing 
    // also see optickscore-/Indexer for another CPU version 

    OpticksEvent* evt = m_hub->getEvent();
    if(!evt) return ; 

    NPY<float>* dpho = evt->getPhotonData();
    if(dpho && dpho->hasData())
    {
        // host based indexing of unique material codes, requires downloadEvt to pull back the photon data
        LOG(info) << "OpticksIdx::indexBoundaries host based " ;
        std::map<unsigned int, std::string> boundary_names = m_hub->getBoundaryNamesMap();
        BoundariesNPY* bnd = new BoundariesNPY(dpho);
        bnd->setBoundaryNames(boundary_names);
        bnd->indexBoundaries();
        evt->setBoundariesNPY(bnd);
    }
    else
    {
        LOG(warning) << "OpticksIdx::indexBoundariesHost dpho NULL or no data " ;
    }

    TIMER("indexBoundariesHost");
}





