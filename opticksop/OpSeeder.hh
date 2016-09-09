#pragma once

//  Distributes unsigned int genstep indices 0:m_num_gensteps-1 into the first 
//  4 bytes of the 4*float4 photon record in the photon buffer 
//  using the number of photons per genstep obtained from the genstep buffer 
//  
//  Note that this is done almost entirely on the GPU, only the num_photons reduction
//  needs to come back to CPU in order to allocate an appropriately sized OptiX photon 
//  buffer on GPU.
//  
//  This per-photon genstep index is used by OptiX photon propagation 
//  program cu/generate.cu to access the appropriate values from the genstep buffer
//
//  TODO: make this operational in COMPUTE as well as INTEROP modes without code duplication ?
//
//
// When operating with OpenGL buffers the buffer_id lodged in NPYBase is 
// all thats needed to reference the GPU buffers.  
//
// When operating without OpenGL need some equivalent way to hold onto the 
// GPU buffers or somehow pass them to OptiX 
// The OptiX buffers live as members of OPropagator in OptiX case, 
// with OBuf jackets providing CBufSlice via slice method.
//


class OpticksHub ; 
class OPropagator ; 
class OpEngine ; 
class OContext ; 

struct CBufSpec ; 

#include "OKOP_API_EXPORT.hh"

class OKOP_API OpSeeder {
   public:
      OpSeeder(OpticksHub* hub, OpEngine* engine);
   public:
      void seedPhotonsFromGensteps();
   private:
      void seedPhotonsFromGenstepsViaOptiX();
      void seedPhotonsFromGenstepsViaOpenGL();
      void seedPhotonsFromGenstepsImp(const CBufSpec& rgs_, const CBufSpec& rox_);
   private:
      OpticksHub*              m_hub ;
      OpEngine*                m_engine ;
      OContext*                m_ocontext ;
      OPropagator*             m_propagator ;
};


