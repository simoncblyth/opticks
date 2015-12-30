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

#include <cstddef>

class NumpyEvt ; 
class OPropagator ; 
class OContext ; 

struct CBufSpec ; 


class OpSeeder {
   public:
      OpSeeder(OContext* ocontext);
   public:
      void setEvt(NumpyEvt* evt);
      void setPropagator(OPropagator* propagator);
   public:
      void seedPhotonsFromGensteps();
   private:
      void seedPhotonsFromGenstepsViaOptiX();
      void seedPhotonsFromGenstepsViaOpenGL();
      void seedPhotonsFromGenstepsImp(const CBufSpec& rgs_, const CBufSpec& rox_);
   private:
      void init();
   private:
      OContext*                m_ocontext ;
      NumpyEvt*                m_evt ;
      OPropagator*             m_propagator ;
};

inline OpSeeder::OpSeeder(OContext* ocontext)  
   :
     m_ocontext(ocontext),
     m_evt(NULL),
     m_propagator(NULL)
{
   init(); 
}

inline void OpSeeder::init()
{
}

inline void OpSeeder::setEvt(NumpyEvt* evt)
{
    m_evt = evt ; 
}  
inline void OpSeeder::setPropagator(OPropagator* propagator)
{
    m_propagator = propagator ; 
}  



