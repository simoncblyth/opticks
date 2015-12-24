#include "OpIndexer.hh"

#include <cassert>

// npy-
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



void OpIndexer::indexSequenceThrust(const CBufSlice& seqh, const CBufSlice& seqm, bool verbose )
{
    m_timer->start();
    TSparse<unsigned long long> seqhis("History_Sequence", seqh );
    TSparse<unsigned long long> seqmat("Material_Sequence", seqm ); 
    m_evt->setHistorySeq(seqhis.getIndex());
    m_evt->setMaterialSeq(seqmat.getIndex());  // the indices are populated by the make_lookup below

    assert(m_phosel != 0 && m_recsel != 0);

    thrust::device_vector<unsigned char> dps(m_num_photons);
    thrust::device_vector<unsigned char> drs(m_num_photons*m_maxrec);

    CBufSpec rps = make_bufspec<unsigned char>(dps); 
    CBufSpec rrs = make_bufspec<unsigned char>(drs) ;

    indexSequenceThrust(seqhis, seqmat, rps, rrs, verbose);
}


void OpIndexer::indexSequenceThrust(
   TSparse<unsigned long long>& seqhis, 
   TSparse<unsigned long long>& seqmat, 
   const CBufSpec& rps,
   const CBufSpec& rrs,
   bool verbose 
)
{
    TBuf tphosel("tphosel", rps );
    tphosel.zero();

    TBuf trecsel("trecsel", rrs );

    if(verbose) dump(tphosel, trecsel);

    seqhis.make_lookup(); 

    // phosel buffer is shaped (num_photons, 1, 4)
    CBufSlice tp_his = tphosel.slice(4,0) ; // stride, begin  
    CBufSlice tp_mat = tphosel.slice(4,1) ; 
  
    seqhis.apply_lookup<unsigned char>(tp_his); 
    if(verbose) dumpHis(tphosel, seqhis) ;

    seqmat.make_lookup();
    seqmat.apply_lookup<unsigned char>(tp_mat);
    if(verbose) dumpMat(tphosel, seqmat) ;

    tphosel.repeat_to<unsigned char>( &trecsel, 4, 0, tphosel.getSize(), m_maxrec );  // other, stride, begin, end, repeats

    tphosel.download<unsigned char>( m_phosel );  // cudaMemcpyDeviceToHost
    trecsel.download<unsigned char>( m_recsel );
}



