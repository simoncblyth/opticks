#pragma once

/**
OpIndexer
==========

Indexes history and material sequences from the sequence buffer
(each photon has unsigned long long 64-bit ints containg step-by-step flags and material indices)   
The sequences are sparse histogrammed (essentially an index of the popularity of each sequence)
creating a small lookup table that is then applied to all photons to create the phosel and 
recsel arrays containg the popularity index.
This index allows fast selection of all photons from the top 32 categories, this
can be used both graphically and in analysis.
The recsel buffer repeats the phosel values maxrec times to provide fast access to the
selection at record level.

There is no-point in attempting to reuse or optimize OpIndexer 
as indexing is a non-essential activity that will not be done in production
although for debugging/analysis it is highly useful.

Cast
-----

optixrap-/OBuf 
    optix buffer wrapper

thrustrap-/TBuf 
    thrust buffer wrapper, with download to NPY interface 

thrustrap-/TSparse
    GPU Thrust sparse histogramming 

cudawrap-/CBufSpec   
    lightweight struct holding device pointer

cudawrap-/CBufSlice  
    lightweight struct holding device pointer and slice addressing begin/end

cudawrap-/CResource  
    OpenGL buffer made available as a CUDA Resource, 
    mapGLToCUDA returns CBufSpec struct 

OBuf and TBuf have slice methods that return CBufSlice structs 
identifying a view of the GPU buffers 

NB hostside allocation deferred for these
note the layering here, this pattern will hopefully facilitate moving 
from OpenGL backed to OptiX backed for COMPUTE mode
although this isnt a good example as indexSequence is not 
necessary in COMPUTE mode 

CAUTION: the lookup resides in CUDA constant memory, not in the TSparse 
object so must apply the lookup before making another 





Sequence Indexing histograms the per-photon sequence

    sequence buffer -> recsel/phosel buffers

*Interop*
    sequence from OBuf, maps preexisting OpenGL recsel and phosel buffers
    as destination for the indices

*Compute*
    Although indexing is not strictly necessary, it has proved so useful 
    for analysis that it is retained.

*Loaded*
    used for example with Geant4 cfg4- sequence data, 
    which comes from host NPY arrays. 

    The sequence data is uploaded to the GPU using thrust device_vectors
    and output indices are written to thrust created  



**/


class OBuf ; 
class OContext ; 
class OEvent ; 

class TBuf ; 

class Opticks ; 
class OpticksHub ; 
class OpticksEvent ; 
struct CBufSlice ; 
struct CBufSpec ; 
template <typename T> class NPY ;
template <typename T> class TSparse ;

class BTimeKeeper ; 


#include "plog/Severity.h"

#include "OpticksSwitches.h"
#include "OKOP_API_EXPORT.hh"

class OKOP_API OpIndexer {
      static const plog::Severity LEVEL ; 
   public:
      OpIndexer(Opticks* ok, OEvent* oevt);

      void setVerbose(bool verbose=true);
      void setNumPhotons(unsigned int num_photons);
      void setSeq(OBuf* seq);
      void setPho(OBuf* pho);
   public:
      void indexBoundaries(); 
      void prepareTarget(const char* msg="OpIndexer::prepareTarget");
   public:
      void indexSequence(); 
   private:
      void checkTarget(const char* msg);
      void indexSequenceLoaded();  
      void indexSequenceInterop();  
      void indexSequenceCompute();  
   private:
      void init();
      void update();  
   private:
      // implemented in OpIndexer_.cu for nvcc compilation
      // allocates recsel and phosel buffers with Thrust device_vector 
      void indexSequenceViaThrust(         
           TSparse<unsigned long long>& seqhis, 
           TSparse<unsigned long long>& seqmat, 
           bool verbose 
      );
      // maps preexisting OpenGL buffers to CUDA
      void indexSequenceViaOpenGL(
           TSparse<unsigned long long>& seqhis, 
           TSparse<unsigned long long>& seqmat, 
           bool verbose 
      );

      void indexSequenceImp(
           TSparse<unsigned long long>& seqhis, 
           TSparse<unsigned long long>& seqmat, 
           const CBufSpec& rps,
           const CBufSpec& rrs,
           bool verbose 
      );
   private:
      void indexBoundariesFromOpenGL(unsigned int photon_buffer_id, unsigned int stride, unsigned int begin);
      void indexBoundariesFromOptiX(OBuf* pho                     , unsigned int stride, unsigned int begin);
   private:

      void dump(const TBuf& tphosel, const TBuf& trecsel);
      void dumpHis(const TBuf& tphosel, const TSparse<unsigned long long>& seqhis) ;
      void dumpMat(const TBuf& tphosel, const TSparse<unsigned long long>& seqhis) ;
      void saveSel();
   private:
      // resident
      Opticks*                 m_ok ; 
      OEvent*                  m_oevt ; 
      OpticksEvent*            m_evt ; 
      OContext*                m_ocontext ;
   private:
      // externally set 
      OBuf*                    m_seq ; 
      OBuf*                    m_pho ; 
      bool                     m_verbose ; 
   private:
      // transients updated by updateEvt at indexSequence
      unsigned int             m_maxrec ; 
      unsigned int             m_num_photons ; 
};

