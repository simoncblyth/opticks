#include "OpIndexer.hh"

#include <cassert>

// optickscore-
#include "OpticksConst.hh"  
#include "OpticksEvent.hh"  

// npy-
#include "Timer.hpp"  
#include "NPY.hpp"  

// optixrap-
#include "OBuf.hh"

// cudawrap-
#include "CResource.hh"
#include "CBufSpec.hh"
#include "CBufSlice.hh"

// thrustrap-
#include "TBuf.hh"
#include "TSparse.hh"
#include "TUtil.hh"

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


void OpIndexer::indexSequenceViaThrust(
   TSparse<unsigned long long>& seqhis, 
   TSparse<unsigned long long>& seqmat, 
   bool verbose
)
{
    NPY<unsigned char>* phosel = m_evt->getPhoselData() ;
    NPY<unsigned char>* recsel = m_evt->getRecselData() ;

    // allocate phosel and recsel GPU buffers
    thrust::device_vector<unsigned char> dps(phosel->getNumValues());
    thrust::device_vector<unsigned char> drs(recsel->getNumValues());

    // CUDA refs to the buffers obtained from Thrust 
    //     thrustrap-/TUtil.make_bufspec 
    //     does thrust::raw_pointer_cast to get raw CUDA pointer from Thrust 
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
    NPY<unsigned char>* phosel = m_evt->getPhoselData() ;
    NPY<unsigned char>* recsel = m_evt->getRecselData() ;

    unsigned int phosel_id = phosel->getBufferId() ;
    unsigned int recsel_id = recsel->getBufferId() ;
    printf("OpIndexer::indexSequenceViaOpenGL phosel_id %u recsel_id %u \n", phosel_id, recsel_id ); 

    CResource rphosel( phosel_id, CResource::W );
    CResource rrecsel( recsel_id, CResource::W );

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
    // NB the make_lookup writes into constant GPU memory 
    //    so must apply that lookup before doing another 
    //    because of this cannot move the make_lookup prior
    //    to this 

    TBuf tphosel("tphosel", rps );
    tphosel.zero();

    TBuf trecsel("trecsel", rrs );

    if(verbose) dump(tphosel, trecsel);

    // phosel buffer is shaped (num_photons, 1, 4)
    CBufSlice tp_his = tphosel.slice(4,0) ; // stride, begin  
    CBufSlice tp_mat = tphosel.slice(4,1) ; 
 

    TIMER("_seqhisMakeLookup");
    seqhis.make_lookup(); 
    TIMER("seqhisMakeLookup");
    seqhis.apply_lookup<unsigned char>(tp_his); 
    TIMER("seqhisApplyLookup");

    if(verbose) dumpHis(tphosel, seqhis) ;

    TIMER("_seqmatMakeLookup");
    seqmat.make_lookup();
    TIMER("seqmatMakeLookup");
    seqmat.apply_lookup<unsigned char>(tp_mat);
    TIMER("seqmatApplyLookup");

    if(verbose) dumpMat(tphosel, seqmat) ;

    tphosel.repeat_to<unsigned char>( &trecsel, 4, 0, tphosel.getSize(), m_maxrec );  // other, stride, begin, end, repeats



    NPY<unsigned char>* phosel = m_evt->getPhoselData() ;
    NPY<unsigned char>* recsel = m_evt->getRecselData() ;

    // hmm: this pull back to host might not be necessary : only used on GPU ?
    tphosel.download<unsigned char>( phosel );  // cudaMemcpyDeviceToHost
    trecsel.download<unsigned char>( recsel );
}







void OpIndexer::indexBoundariesFromOptiX(OBuf* pho, unsigned int stride, unsigned int begin)
{
     CBufSlice cbnd = pho->slice(stride,begin) ;    // gets CUDA devPtr from OptiX

     TSparse<int> boundaries(OpticksConst::BNDIDX_NAME_, cbnd, false); // hexkey effects Index and dumping only 
    
     m_evt->setBoundaryIndex(boundaries.getIndex());
    
     boundaries.make_lookup();

     if(m_verbose)
        boundaries.dump("OpIndexer::indexBoundariesFromOptiX PTR_FROM_OPTIX TSparse<int>::dump");
}

void OpIndexer::indexBoundariesFromOpenGL(unsigned int photon_id, unsigned int stride, unsigned int begin)
{
    // NB this is not using the OptiX buffer, 
    //    OpenGL buffer is interop to CUDA accessed directly 

    CResource rphoton( photon_id, CResource::R );

    CBufSpec rph = rphoton.mapGLToCUDA<int>();    // gets CUDA devPtr from OpenGL
    {
        CBufSlice cbnd = rph.slice(stride,begin) ; // stride, begin  

        TSparse<int> boundaries(OpticksConst::BNDIDX_NAME_, cbnd, false);
    
        m_evt->setBoundaryIndex(boundaries.getIndex());
    
        boundaries.make_lookup();

        if(m_verbose)
           boundaries.dump("OpIndexer::indexBoundariesFromOpenGL PTR_FROM_OPTIX TSparse<int>::dump");

        rphoton.unmapGLToCUDA(); 
    }
}




