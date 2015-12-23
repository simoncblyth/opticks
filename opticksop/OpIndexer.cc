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
    m_phosel = m_evt->getPhoselData(); 
    m_recsel = m_evt->getRecselData();
    m_maxrec = m_evt->getMaxRec(); 
}


void OpIndexer::indexSequence()
{
    if(!m_evt) return ; 
    if(!m_seq) return ; 

    m_timer->start();

    CBufSlice seqh = m_seq->slice(2,0) ;  // stride, begin
    CBufSlice seqm = m_seq->slice(2,1) ;

    TSparse<unsigned long long> seqhis("History_Sequence", seqh );
    TSparse<unsigned long long> seqmat("Material_Sequence", seqm ); 
    m_evt->setHistorySeq(seqhis.getIndex());
    m_evt->setMaterialSeq(seqmat.getIndex());  // the indices are populated below

    CResource rphosel( m_phosel->getBufferId(), CResource::W );
    CResource rrecsel( m_recsel->getBufferId(), CResource::W );
    {
        TBuf tphosel("tphosel", rphosel.mapGLToCUDA<unsigned char>() );
        tphosel.zero();

        TBuf trecsel("trecsel", rrecsel.mapGLToCUDA<unsigned char>() );
        //dump(tphosel, trecsel);

        seqhis.make_lookup(); 
        seqhis.apply_lookup<unsigned char>( tphosel.slice(4,0));  // stride, begin
        //dumpHis(tphosel, seqhis) ;

        seqmat.make_lookup();
        seqmat.apply_lookup<unsigned char>( tphosel.slice(4,1));
        //dumpMat(tphosel, seqmat) ;

        tphosel.repeat_to<unsigned char>( &trecsel, 4, 0, tphosel.getSize(), m_maxrec );  // other, stride, begin, end, repeats

        tphosel.download<unsigned char>( m_phosel );  // cudaMemcpyDeviceToHost
        trecsel.download<unsigned char>( m_recsel );
    }
    rphosel.unmapGLToCUDA(); 
    rrecsel.unmapGLToCUDA(); 

    m_timer->stop();

    (*m_timer)("indexSequence"); 
}


void OpIndexer::dumpHis(const TBuf& tphosel, const TSparse<unsigned long long>& seqhis)
{
    OBuf* seq = m_seq ; 
    if(!seq) return ; 

    unsigned int nsqa = seq->getNumAtoms(); 
    unsigned int nsqd = std::min(nsqa,100u); 
    unsigned int nphosel = tphosel.getSize() ; 
    unsigned int npsd = std::min(nphosel,100u) ;

    seq->dump<unsigned long long>("OpIndexer::dumpHis seq(2,0)", 2, 0, nsqd);
    tphosel.dumpint<unsigned char>("tphosel.dumpint<unsigned char>(4,0)", 4,0, npsd) ;
    LOG(info) << seqhis.dump_("OpIndexer::dumpHis seqhis");
}

void OpIndexer::dumpMat(const TBuf& tphosel, const TSparse<unsigned long long>& seqmat)
{
    OBuf* seq = m_seq ; 
    if(!seq) return ; 

    unsigned int nsqa = seq->getNumAtoms(); 
    unsigned int nsqd = std::min(nsqa,100u); 
    unsigned int nphosel = tphosel.getSize() ; 
    unsigned int npsd = std::min(nphosel,100u) ;

    seq->dump<unsigned long long>("OpIndexer::dumpMat OBuf seq(2,1)", 2, 1, nsqd);
    tphosel.dumpint<unsigned char>("tphosel.dumpint<unsigned char>(4,1)", 4,1, npsd) ;
    LOG(info) << seqmat.dump_("OpIndexer::dumpMat seqmat");
}

void OpIndexer::dump(const TBuf& tphosel, const TBuf& trecsel)
{
    OBuf* seq = m_seq ; 
    if(!seq) return ; 

    unsigned int nphosel = tphosel.getSize() ; 
    unsigned int nrecsel = trecsel.getSize() ; 

    unsigned int nsqa = seq->getNumAtoms(); 
    assert(nphosel == 2*nsqa);

    unsigned int npsd = std::min(nphosel,100u) ;
    unsigned int nsqd = std::min(nsqa,100u); 

    assert(nrecsel == m_maxrec*2*nsqa);

    LOG(info) << "OpIndexer::dump"
              << " nsqa (2*num_photons)" << nsqa 
              << " nphosel " << nphosel
              << " nrecsel " << nrecsel
              ; 
}



void OpIndexer::saveSel()
{
    m_phosel->save("/tmp/phosel.npy");  
    m_recsel->save("/tmp/recsel.npy");  
}

