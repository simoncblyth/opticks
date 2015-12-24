#include "OpIndexer.hh"

#include <algorithm>
#include <cassert>

// npy-
#include "NLog.hpp"
#include "Timer.hpp"
#include "NumpyEvt.hpp"  
#include "NPY.hpp"  

// cudawrap-
#include "CResource.hh"
#include "CBufSpec.hh"

// thrustrap-
#include "TBuf.hh"
#include "TSparse.hh"
#include "TUtil.hh"

// thrust 
#include <thrust/device_vector.h>


// optixrap-
#include "OBuf.hh"


void OpIndexer::init()
{
    LOG(info) << "OpIndexer::init" ;
    m_timer      = new Timer("OpIndexer::");
    m_timer->setVerbose(true);
}

void OpIndexer::setEvt(NumpyEvt* evt)
{
    m_evt = evt ; 
}

void OpIndexer::updateEvt()
{
    m_phosel = m_evt->getPhoselData(); 
    m_recsel = m_evt->getRecselData();
    m_maxrec = m_evt->getMaxRec(); 
    m_sequence = m_evt->getSequenceData();

    setNumPhotons(m_evt->getNumPhotons());
}


void OpIndexer::setNumPhotons(unsigned int num_photons)
{
    assert(num_photons == m_sequence->getShape(0));
    m_num_photons = num_photons ; 
}


void OpIndexer::indexSequence()
{
    if(!m_evt) return ; 
    updateEvt();

    if(m_evt->isIndexed())
    {
        LOG(info) << "OpIndexer::indexSequence evt is indexed already : SKIPPING " ; 
        return ; 
    }

    m_evt->prepareForIndexing();
    updateEvt();

    if(m_seq)
    {
        // used by standard indexing from ggv- ie with OptiX OpenGL and Thrust all in play 
        indexSequenceOptiXGLThrust();
    }
    else
    {
        // used by the opop- indexer with just Thrust in play 
        indexSequenceThrust();
    }
}


void OpIndexer::indexSequenceThrust()
{
    LOG(info) << "OpIndexer::indexSequenceThrust" ; 

    NPY<unsigned long long>* ph = m_evt->getSequenceData(); 

    thrust::device_vector<unsigned long long> dph(ph->begin(),ph->end());

    CBufSpec cph = make_bufspec<unsigned long long>(dph); 

    TBuf tph("tph", cph);

    tph.dump<unsigned long long>("tph dump", 2, 0, 10 );  
    
    CBufSlice phh = tph.slice(2,0) ; // stride, begin  
    CBufSlice phm = tph.slice(2,1) ;

    indexSequenceThrust(phh, phm, true);
}



void OpIndexer::indexSequenceOptiXGLThrust()
{
    LOG(info) << "OpIndexer::indexSequenceOptiXGLThrust" ; 
    CBufSlice seqh = m_seq->slice(2,0) ;  // stride, begin
    CBufSlice seqm = m_seq->slice(2,1) ;
    indexSequenceGLThrust(seqh, seqm, true);
}

void OpIndexer::indexSequenceGLThrust()
{
    // loaded does not involve OptiX, so there is no OBuf 
    // instead grab the sequence 
    LOG(info) << "OpIndexer::indexSequenceGLThrust" ; 

    CResource rsequence( m_sequence->getBufferId(), CResource::W );
    {
        CBufSpec seqbuf = rsequence.mapGLToCUDA<unsigned long long>() ;
        TBuf tsequence("tsequence", seqbuf );
        CBufSlice seqh = tsequence.slice(2,0) ; // stride, begin  
        CBufSlice seqm = tsequence.slice(2,1) ;
        indexSequenceThrust(seqh, seqm, true);
    }
    rsequence.unmapGLToCUDA(); 
}


void OpIndexer::indexSequenceGLThrust(const CBufSlice& seqh, const CBufSlice& seqm, bool verbose )
{
    m_timer->start();

    TSparse<unsigned long long> seqhis("History_Sequence", seqh );
    TSparse<unsigned long long> seqmat("Material_Sequence", seqm ); 
    m_evt->setHistorySeq(seqhis.getIndex());
    m_evt->setMaterialSeq(seqmat.getIndex());  // the indices are populated by the make_lookup below

    CResource rphosel( m_phosel->getBufferId(), CResource::W );
    CResource rrecsel( m_recsel->getBufferId(), CResource::W );
    CBufSpec rps = rphosel.mapGLToCUDA<unsigned char>() ;
    CBufSpec rrs = rrecsel.mapGLToCUDA<unsigned char>() ;
   
    indexSequenceThrust(seqhis, seqmat, rps, rrs, verbose);

    rphosel.unmapGLToCUDA(); 
    rrecsel.unmapGLToCUDA(); 

    m_timer->stop();

    (*m_timer)("indexSequenceGLThrust"); 
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



void OpIndexer::saveSel()
{
    m_phosel->save("/tmp/phosel.npy");  
    m_recsel->save("/tmp/recsel.npy");  
}

