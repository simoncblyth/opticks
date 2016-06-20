
#include <algorithm>
#include <cassert>

// opticksop-
#include "OpIndexer.hh"

// optickscore-
#include "OpticksConst.hh"
#include "OpticksEvent.hh"

// npy-
#include "BLog.hh"
#include "Timer.hpp"
#include "NPY.hpp"  

// thrustrap-
#include "TBuf.hh"
#include "TSparse.hh"
#include "TUtil.hh"

// optixrap-
#include "OBuf.hh"
#include "OContext.hh"
#include "OPropagator.hh"

// cudawrap-
#include "CResource.hh"
#include "CBufSpec.hh"

// thrust 
#include <thrust/device_vector.h>


#define TIMER(s) \
    { \
       if(m_evt)\
       {\
          Timer& t = *(m_evt->getTimer()) ;\
          t((s)) ;\
       }\
    }



void OpIndexer::update()
{
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

    bool hexkey = false ; 

    if(!m_pho)
    {
        LOG(warning) << "OpIndexer::indexBoundaries OBuf m_pho is NULL : SKIPPING " ; 
        return ;  
    }

    TSparse<int> boundaries(OpticksConst::BNDIDX_NAME_, m_pho->slice(4*4,4*3+0), hexkey); // stride,begin  hexkey effects Index and dumping only 

    m_evt->setBoundaryIndex(boundaries.getIndex());

    boundaries.make_lookup();

    if(m_verbose)
    boundaries.dump("OpIndexer::indexBoundaries TSparse<int>::dump");

    TIMER("indexBoundaries"); 
}




void OpIndexer::indexSequence()
{
    if(m_evt->isIndexed())
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
    update();

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

    update();
    if(m_evt->isIndexed()) return ;  

    if(!m_seq)
        LOG(fatal) << "OpIndexer::indexSequenceInterop"
                   << " m_seq NULL " ; 

    assert(m_seq);

    LOG(info) << "OpIndexer::indexSequenceInterop" ; 
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
    update();
    if(m_evt->isIndexed()) 
    {
        LOG(info) << "OpIndexer::indexSequenceLoaded evt already indexed" ; 
        return ;  
    }

    NPY<unsigned long long>* ph = m_evt->getSequenceData(); 
    if(!ph) LOG(fatal) << "OpIndexer::indexSequenceLoaded" << " ph NULL " ; 
    assert(ph);

    thrust::device_vector<unsigned long long> dph(ph->begin(),ph->end());
    CBufSpec cph = make_bufspec<unsigned long long>(dph); 

    TBuf tph("tph", cph);
    tph.dump<unsigned long long>("OpIndexer::indexSequenceLoaded tph dump", 2, 0, 10 );  
    
    CBufSlice phh = tph.slice(2,0) ; // stride, begin  
    CBufSlice phm = tph.slice(2,1) ;

    TSparse<unsigned long long> seqhis(OpticksConst::SEQHIS_NAME_,  phh );
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


