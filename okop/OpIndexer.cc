#include <cstddef>
#include <algorithm>
#include <cassert>

// optickscore-
#include "Opticks.hh"
#include "OpticksSwitches.h"
#include "OpticksConst.hh"
#include "OpticksEvent.hh"
#include "OpticksBufferControl.hh"

// npy-
#include "PLOG.hh"
#include "BTimeKeeper.hh"
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
#include "OEvent.hh"

// cudarap-
#include "CResource.hh"
#include "CBufSpec.hh"

// thrust 
#include <thrust/device_vector.h>

// okop-
#include "OpIndexer.hh"



OpIndexer::OpIndexer(Opticks* ok, OEvent* oevt)  
   :
     m_ok(ok),
     m_oevt(oevt),
     m_evt(NULL),
     m_ocontext(oevt->getOContext()),
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
    m_evt = m_ok->getEvent();
    assert(m_evt) ;

    m_maxrec = m_evt->getMaxRec(); 

    setNumPhotons(m_evt->getNumPhotons());

#ifdef WITH_RECORD
    OBuf* seq = m_oevt ? m_oevt->getSequenceBuf() : NULL ;
    setSeq(seq);
#endif

    OBuf* pho = m_oevt ? m_oevt->getPhotonBuf() : NULL ;
    setPho(pho);

}

void OpIndexer::setNumPhotons(unsigned int num_photons)
{

#ifdef WITH_RECORD
    NPY<unsigned long long>* seq = m_evt->getSequenceData() ;
    unsigned int x_num_photons = seq ? seq->getShape(0) : 0 ; 
#else
    NPY<float>* pho = m_evt->getPhotonData();
    unsigned int x_num_photons = pho ? pho->getShape(0) : 0 ; 
#endif
    bool expected = num_photons == x_num_photons ;

    if(!expected)
    {
        LOG(fatal) << "OpIndexer::setNumPhotons"
                   << " DISCREPANCY  " 
                   << " num_photons " << num_photons
                   << " x_num_photons " << x_num_photons
                   ; 
    }

    assert(expected);
    m_num_photons = num_photons ; 
}





void OpIndexer::indexBoundaries()
{
    OK_PROFILE("_OpIndexer::indexBoundaries"); 

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
    else if(ctrl & OpticksBufferControl::INTEROP_PTR_FROM_OPTIX )
    {
         indexBoundariesFromOptiX(m_pho, stride, begin);
    } 
    else if(ctrl & OpticksBufferControl::INTEROP_PTR_FROM_OPENGL)
    {
         assert(buffer_id > 0);
         indexBoundariesFromOpenGL(buffer_id, stride, begin);
    }
    else
    {
         assert(0 && "NO BUFFER CONTROL");
    }


    OK_PROFILE("OpIndexer::indexBoundaries"); 
}










#ifdef WITH_RECORD
void OpIndexer::indexSequence()
{
    OK_PROFILE("_OpIndexer::indexSequence");

    update();

    if(m_evt->isIndexed() && !m_ok->hasOpt("forceindex"))
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
    OK_PROFILE("OpIndexer::indexSequence");
}

void OpIndexer::indexSequenceCompute()
{
    OK_PROFILE("_OpIndexer::indexSequenceCompute"); 

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

    OK_PROFILE("OpIndexer::indexSequenceCompute"); 
}


/**
OpIndexer::indexSequenceInterop
----------------------------------

Used by standard indexing with OptiX OpenGL and Thrust all in play 

OptiX OBuf provides access to the sequence buffer
and OpenGL mapping to CUDA gives access to the output recsel/phosel
as these are used by OpenGL

**/

void OpIndexer::indexSequenceInterop()
{
    OK_PROFILE("_OpIndexer::indexSequenceInterop"); 

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

    OK_PROFILE("OpIndexer::indexSequenceInterop"); 
}


void OpIndexer::indexSequenceLoaded()
{
    OK_PROFILE("_OpIndexer::indexSequenceLoaded"); 
    // starts from host based index

    LOG(info) << "OpIndexer::indexSequenceLoaded" ; 
    if(m_evt->isIndexed() && !m_ok->hasOpt("forceindex")) 
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

    OK_PROFILE("OpIndexer::indexSequenceLoaded"); 

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
#endif


