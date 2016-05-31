#include "OpticksEvent.hh"
#include "Indexer.hh"

// npy-
#include "NumpyEvt.hpp"
#include "Timer.hpp"
#include "NLog.hpp"


#define TIMER(s) \
    { \
       if(m_evt)\
       {\
          Timer& t = *(m_evt->getTimer()) ;\
          t((s)) ;\
       }\
    }



void OpticksEvent::init()
{
}

void OpticksEvent::indexPhotonsCPU()
{
    // see tests/IndexerTest

    LOG(info) << "OpticksEvent::indexPhotonsCPU" ; 

    NPY<unsigned long long>* sequence = m_evt->getSequenceData();
    NPY<unsigned char>*        phosel = m_evt->getPhoselData();
    assert(sequence->getShape(0) == phosel->getShape(0));

    Indexer<unsigned long long>* idx = new Indexer<unsigned long long>(sequence) ; 
    idx->indexSequence();
    idx->applyLookup<unsigned char>(phosel->getValues());

    // TODO: phosel->recsel by repeating by maxrec

    m_evt->setHistoryIndex(idx->getHistoryIndex());
    m_evt->setMaterialIndex(idx->getMaterialIndex());

    TIMER("indexPhotonsCPU");    
}


