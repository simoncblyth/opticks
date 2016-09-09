#include <cstddef>
#include <algorithm>
#include <cassert>

// optickscore-
#include "OpticksConst.hh"
#include "OpticksEvent.hh"
#include "OpticksBufferControl.hh"

// opticksgeo-
#include "OpticksHub.hh"

// npy-
#include "PLOG.hh"
#include "Timer.hpp"
#include "NPY.hpp"  

// thrustrap-
#include "TBuf.hh"
#include "TSparse.hh"
#include "TUtil.hh"

// okc-
#include "Opticks.hh"

// optixrap-
#include "OBuf.hh"
#include "OContext.hh"
#include "OPropagator.hh"

// cudawrap-
#include "CResource.hh"
#include "CBufSpec.hh"

// thrust 
#include <thrust/device_vector.h>


// opticksop-
#include "OpIndexer.hh"
#include "OpEngine.hh"



#define TIMER(s) \
    { \
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }


OpIndexer::OpIndexer(OpticksHub* hub, OpEngine* engine)  
   :
     m_hub(hub),
     m_engine(engine),
     m_opticks(hub->getOpticks()),
     m_evt(NULL),
     m_ocontext(engine->getOContext()),
     m_propagator(engine->getOPropagator()),
     m_seq(NULL),
     m_pho(NULL),
     m_verbose(false),
     m_maxrec(0),
     m_num_photons(0)
{
}

void OpIndexer::setSeq(OBuf* seq)
{
    m_seq = seq ; 
}
void OpIndexer::setPho(OBuf* pho)
{
    m_pho = pho ; 
}
void OpIndexer::setVerbose(bool verbose)
{
    m_verbose = verbose ; 
}




void OpIndexer::update()
{
    m_evt = m_hub->getEvent();
    assert(m_evt) ;

    m_maxrec = m_evt->getMaxRec(); 

    setNumPhotons(m_evt->getNumPhotons());

    OBuf* seq = m_propagator ? m_propagator->getSequenceBuf() : NULL ;
    setSeq(seq);

    OBuf* pho = m_propagator ? m_propagator->getPhotonBuf() : NULL ;
    setPho(pho);

}

void OpIndexer::setNumPhotons(unsigned int num_photons)
{

    NPY<unsigned long long>* seq = m_evt->getSequenceData() ;
    unsigned int x_num_photons = seq ? seq->getShape(0) : 0 ; 
    bool expected = num_photons == x_num_photons ;

    if(!expected)
    {
        LOG(fatal) << "OpIndexer::setNumPhotons"
                   << " discrepancy with sequence data length " 
                   << " num_photons " << num_photons
                   << " x_num_photons " << x_num_photons
                   ; 
    }

    assert(expected);
    m_num_photons = num_photons ; 
}





void OpIndexer::indexBoundaries()
{
    update();

    if(!m_pho)
    {
        LOG(warning) << "OpIndexer::indexBoundaries OBuf m_pho is NULL : SKIPPING " ; 
        return ;  
    }


    bool compute = m_ocontext->isCompute() ;
    //NPYBase* npho = m_pho->getNPY();
    NPYBase* npho = m_evt->getData(OpticksEvent::photon_);
    unsigned int buffer_id = npho->getBufferId();
    unsigned long long ctrl = npho->getBufferControl();

    unsigned int stride = 4*4 ; 
    unsigned int begin  = 4*3+0 ;  

    if(compute)
    {
         indexBoundariesFromOptiX(m_pho, stride, begin);
    }
    else if(ctrl & OpticksBufferControl::PTR_FROM_OPTIX )
    {
         indexBoundariesFromOptiX(m_pho, stride, begin);
    } 
    else if(ctrl & OpticksBufferControl::PTR_FROM_OPENGL)
    {
         assert(buffer_id > 0);
         indexBoundariesFromOpenGL(buffer_id, stride, begin);
    }
    else
    {
         assert(0 && "NO BUFFER CONTROL");
    }

    TIMER("indexBoundaries"); 
}












void OpIndexer::indexSequence()
{
    update();

    if(m_evt->isIndexed() && !m_opticks->hasOpt("forceindex"))
    {
        LOG(info) << "OpIndexer::indexSequence"
                  << " already indexed SKIP "
                  ;
        return ;
    }

    bool loaded = m_evt->isLoaded();
    bool compute = m_ocontext ? m_ocontext->isCompute() : false ;

    if(loaded)
    {
        indexSequenceLoaded();
    }
    else if(compute)
    {
        indexSequenceCompute();
    }
    else
    {
        indexSequenceInterop();
    }
}

void OpIndexer::indexSequenceCompute()
{
    if(!m_seq)
        LOG(fatal) << "OpIndexer::indexSequenceCompute"
                   << " m_seq NULL " ; 

    assert(m_seq);

    LOG(info) << "OpIndexer::indexSequenceCompute" ; 
    CBufSlice seqh = m_seq->slice(2,0) ;  // stride, begin
    CBufSlice seqm = m_seq->slice(2,1) ;

    TSparse<unsigned long long> seqhis(OpticksConst::SEQHIS_NAME_, seqh );
    TSparse<unsigned long long> seqmat(OpticksConst::SEQMAT_NAME_, seqm ); 

    m_evt->setHistoryIndex(seqhis.getIndex());
    m_evt->setMaterialIndex(seqmat.getIndex());  // the indices are populated by the make_lookup below

    indexSequenceViaThrust(seqhis, seqmat, m_verbose );

    TIMER("indexSequenceCompute"); 
}


void OpIndexer::indexSequenceInterop()
{
    // used by standard indexing from ggv- ie with OptiX OpenGL and Thrust all in play 
    //
    // OptiX OBuf provides access to the sequence buffer
    // and OpenGL mapping to CUDA gives access to the output recsel/phosel
    // as these are used by OpenGL


    if(!m_seq)
        LOG(fatal) << "OpIndexer::indexSequenceInterop"
                   << " m_seq NULL " ; 

    assert(m_seq);

    LOG(info) << "OpIndexer::indexSequenceInterop slicing (OBufBase*)m_seq " ; 
    CBufSlice seqh = m_seq->slice(2,0) ;  // stride, begin
    CBufSlice seqm = m_seq->slice(2,1) ;

    TSparse<unsigned long long> seqhis(OpticksConst::SEQHIS_NAME_, seqh );
    TSparse<unsigned long long> seqmat(OpticksConst::SEQMAT_NAME_, seqm ); 

    m_evt->setHistoryIndex(seqhis.getIndex());
    m_evt->setMaterialIndex(seqmat.getIndex());  // the indices are populated by the make_lookup below

    indexSequenceViaOpenGL(seqhis, seqmat, m_verbose );

    TIMER("indexSequenceInterop"); 
}


void OpIndexer::indexSequenceLoaded()
{
    // starts from host based index

    LOG(info) << "OpIndexer::indexSequenceLoaded" ; 
    if(m_evt->isIndexed() && !m_opticks->hasOpt("forceindex")) 
    {
        LOG(info) << "OpIndexer::indexSequenceLoaded evt already indexed" ; 
        return ;  
    }

    NPY<unsigned long long>* ph = m_evt->getSequenceData(); 
    if(!ph) LOG(fatal) << "OpIndexer::indexSequenceLoaded" << " ph NULL " ; 
    assert(ph);

    thrust::device_vector<unsigned long long> dph(ph->begin(),ph->end());
    CBufSpec cph = make_bufspec<unsigned long long>(dph); 
    cph.hexdump = true ; 

    TBuf tph("tph", cph);
    tph.dump<unsigned long long>("OpIndexer::indexSequenceLoaded tph dump st/be/en 2/0/20", 2, 0, 20 );  
    tph.dump<unsigned long long>("OpIndexer::indexSequenceLoaded tph dump st/be/en 2/1/20", 2, 1, 20 );  
    
    CBufSlice phh = tph.slice(2,0) ; // stride, begin  
    CBufSlice phm = tph.slice(2,1) ;

    TSparse<unsigned long long> seqhis(OpticksConst::SEQHIS_NAME_, phh );
    TSparse<unsigned long long> seqmat(OpticksConst::SEQMAT_NAME_, phm ); 



    m_evt->setHistoryIndex(seqhis.getIndex());
    m_evt->setMaterialIndex(seqmat.getIndex());  // the indices are populated by the make_lookup below

    prepareTarget("indexSequenceLoaded");

    indexSequenceViaThrust(seqhis, seqmat, m_verbose );

    TIMER("indexSequenceLoaded"); 
}


void OpIndexer::prepareTarget(const char* msg)
{
    NPY<unsigned char>*  phosel = m_evt->getPhoselData();
    NPY<unsigned char>*  recsel = m_evt->getRecselData();

    assert(phosel && "photon index lookups are written to phosel, this must be allocated with num photons length " );
    assert(recsel && "photon index lookups are repeated to into recsel, this must be allocated with num records length " );

    LOG(info) << "OpIndexer::checkTarget"
              << " (" << msg << ") " 
              << " phosel " << phosel->getShapeString() 
              << " recsel " << recsel->getShapeString() 
              ;

    if(phosel->getShape(0) == 0 && recsel->getShape(0) == 0)
    {
        //m_evt->resizeIndices();
    }

}



void OpIndexer::dumpHis(const TBuf& tphosel, const TSparse<unsigned long long>& seqhis)
{
    OBuf* seq = m_seq ; 
    if(seq)
    {
        unsigned int nsqa = seq->getNumAtoms(); 
        unsigned int nsqd = std::min(nsqa,100u); 
        seq->dump<unsigned long long>("OpIndexer::dumpHis seq(2,0)", 2, 0, nsqd);
    }

    unsigned int nphosel = tphosel.getSize() ; 
    unsigned int npsd = std::min(nphosel,100u) ;
    tphosel.dumpint<unsigned char>("tphosel.dumpint<unsigned char>(4,0)", 4,0, npsd) ;
    LOG(info) << seqhis.dump_("OpIndexer::dumpHis seqhis");
}

void OpIndexer::dumpMat(const TBuf& tphosel, const TSparse<unsigned long long>& seqmat)
{
    OBuf* seq = m_seq ; 
    if(seq) 
    {
        unsigned int nsqa = seq->getNumAtoms(); 
        unsigned int nsqd = std::min(nsqa,100u); 
        seq->dump<unsigned long long>("OpIndexer::dumpMat OBuf seq(2,1)", 2, 1, nsqd);
    }

    unsigned int nphosel = tphosel.getSize() ; 
    unsigned int npsd = std::min(nphosel,100u) ;
    tphosel.dumpint<unsigned char>("tphosel.dumpint<unsigned char>(4,1)", 4,1, npsd) ;
    LOG(info) << seqmat.dump_("OpIndexer::dumpMat seqmat");
}

void OpIndexer::dump(const TBuf& tphosel, const TBuf& trecsel)
{
    OBuf* seq = m_seq ; 

    unsigned int nphosel = tphosel.getSize() ; 
    unsigned int nrecsel = trecsel.getSize() ; 

    LOG(info) << "OpIndexer::dump"
              << " nphosel " << nphosel
              << " nrecsel " << nrecsel
              ; 

    if(seq)
    {
        unsigned int nsqa = seq->getNumAtoms(); 
        assert(nphosel == 2*nsqa);
        assert(nrecsel == m_maxrec*2*nsqa);
    } 
}


