#include "OpIndexer.hh"

#include <cassert>

// npy-
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

void OpIndexer::indexSequenceViaThrust(
   TSparse<unsigned long long>& seqhis, 
   TSparse<unsigned long long>& seqmat, 
   bool verbose
)
{
    // allocate phosel and recsel GPU buffers
    thrust::device_vector<unsigned char> dps(m_phosel->getNumValues());
    thrust::device_vector<unsigned char> drs(m_recsel->getNumValues());

    // refs to the buffers
    CBufSpec rps = make_bufspec<unsigned char>(dps); 
    CBufSpec rrs = make_bufspec<unsigned char>(drs) ;

    indexSequenceImp(seqhis, seqmat, rps, rrs, verbose);
}

void OpIndexer::indexSequenceViaOpenGL(
   TSparse<unsigned long long>& seqhis, 
   TSparse<unsigned long long>& seqmat, 
   bool verbose
)
{
    CResource rphosel( m_phosel->getBufferId(), CResource::W );
    CResource rrecsel( m_recsel->getBufferId(), CResource::W );

    // grab refs to the OpenGL GPU buffers
    CBufSpec rps = rphosel.mapGLToCUDA<unsigned char>() ;
    CBufSpec rrs = rrecsel.mapGLToCUDA<unsigned char>() ;
   
    indexSequenceImp(seqhis, seqmat, rps, rrs, verbose);

    // hand back to OpenGL
    rphosel.unmapGLToCUDA(); 
    rrecsel.unmapGLToCUDA(); 
}

void OpIndexer::indexSequenceImp(
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


    // hmm: this pull back to host might not be necessary : only used on GPU ?
    tphosel.download<unsigned char>( m_phosel );  // cudaMemcpyDeviceToHost
    trecsel.download<unsigned char>( m_recsel );
}


