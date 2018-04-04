#include "OpticksEvent.hh"
#include "OpticksBufferSpec.hh"
#include "OpticksCMakeConfig.hh"
#include "OpticksSwitches.h"
/**
OpticksBufferSpec
==================

Accomodating new versions of OptiX
------------------------------------

Compiling with an unhandled version of OptiX gives::

    [ 64%] Linking CXX shared library libOpticksCore.dylib
    Undefined symbols for architecture x86_64:
      "OpticksBufferSpec::photon_compute_", referenced from:
          OpticksBufferSpec::Get(char const*, bool) in OpticksBufferSpec.cc.o
      "OpticksBufferSpec::photon_interop_", referenced from:

To find the version to handle see sysrap-/OpticksCMakeConfigTest 


Meanings of the settings
---------------------------


OPTIX_NON_INTEROP  
    creates OptiX buffer even in INTEROP mode, this is possible for buffers such as "sequence"
    which are not used from OpenGL shaders so there is no need for the INTEROP OpenGL buffer
    instead can profit from speed and simplicity of pure OptiX buffer

 INTEROP_PTR_FROM_OPENGL  
    adopted this with OptiX 4.0, as OpenGL/OptiX/CUDA 3-way interop not working in 400000 
    instead moved to 
           OpenGL/OptiX : to write the photon data
           OpenGL/CUDA  : to index the photons  


NB have just moved to splitting the spec by compute and interop
   so some simplifications should be possible


INTEROP mode GPU buffer access C:create R:read W:write
----------------------------------------------------------

                 OpenGL     OptiX              Thrust 

   gensteps       CR       R (gen/prop)       R (seeding)
   source         CR       R (gen/prop)       -

   photons        CR       W (gen/prop)       W (seeding)
   sequence                W (gen/prop)
   phosel         CR                          W (indexing) 

   records        CR       W  
   recsel         CR                          W (indexing)


OptiX has no business with phosel and recsel 

Photon spec
--------------

* OpIndexer::indexBoundaries demands to know where to get pointer from for interop running

**/

#if OXRAP_OPTIX_VERSION == 3080 || OXRAP_OPTIX_VERSION == 3090 

#ifdef WITH_SEED_BUFFER
const char* OpticksBufferSpec::photon_compute_ = "OPTIX_OUTPUT_ONLY"  ;
const char* OpticksBufferSpec::photon_interop_ = "OPTIX_OUTPUT_ONLY,INTEROP_PTR_FROM_OPENGL"  ;
#else
const char* OpticksBufferSpec::photon_compute_ = "OPTIX_INPUT_OUTPUT,BUFFER_COPY_ON_DIRTY"  ;
const char* OpticksBufferSpec::photon_interop_ = "OPTIX_INPUT_OUTPUT,BUFFER_COPY_ON_DIRTY,INTEROP_PTR_FROM_OPENGL"  ;
#endif

#elif OXRAP_OPTIX_VERSION == 400000 || OXRAP_OPTIX_VERSION == 40000 ||  OXRAP_OPTIX_VERSION == 40101 || OXRAP_OPTIX_VERSION == 50001

#ifdef WITH_SEED_BUFFER
const char* OpticksBufferSpec::photon_compute_ = "OPTIX_OUTPUT_ONLY"  ;
const char* OpticksBufferSpec::photon_interop_ = "OPTIX_OUTPUT_ONLY,INTEROP_PTR_FROM_OPENGL"  ;
#else
const char* OpticksBufferSpec::photon_compute_ = "OPTIX_INPUT_OUTPUT,BUFFER_COPY_ON_DIRTY"  ;
const char* OpticksBufferSpec::photon_interop_ = "OPTIX_INPUT_OUTPUT,INTEROP_PTR_FROM_OPENGL,BUFFER_COPY_ON_DIRTY"  ;
#endif

#endif


#if OXRAP_OPTIX_VERSION == 3080 || OXRAP_OPTIX_VERSION == 3090 

const char* OpticksBufferSpec::genstep_compute_ = "OPTIX_NON_INTEROP,OPTIX_INPUT_ONLY"  ;
const char* OpticksBufferSpec::genstep_interop_ = "OPTIX_INPUT_ONLY"  ; 

#elif OXRAP_OPTIX_VERSION == 400000 || OXRAP_OPTIX_VERSION == 40000 ||  OXRAP_OPTIX_VERSION == 40101 || OXRAP_OPTIX_VERSION == 50001

const char* OpticksBufferSpec::genstep_compute_ = "OPTIX_INPUT_ONLY,UPLOAD_WITH_CUDA,BUFFER_COPY_ON_DIRTY,VERBOSE_MODE"  ;
const char* OpticksBufferSpec::genstep_interop_ = "OPTIX_INPUT_ONLY"  ; 

#endif



#if OXRAP_OPTIX_VERSION == 3080 || OXRAP_OPTIX_VERSION == 3090 

const char* OpticksBufferSpec::source_compute_ = "OPTIX_NON_INTEROP,OPTIX_INPUT_ONLY"  ;
const char* OpticksBufferSpec::source_interop_ = "OPTIX_INPUT_ONLY"  ; 

#elif OXRAP_OPTIX_VERSION == 400000 || OXRAP_OPTIX_VERSION == 40000 ||  OXRAP_OPTIX_VERSION == 40101 || OXRAP_OPTIX_VERSION == 50001

const char* OpticksBufferSpec::source_compute_ = "OPTIX_INPUT_ONLY,UPLOAD_WITH_CUDA,BUFFER_COPY_ON_DIRTY,VERBOSE_MODE"  ;
const char* OpticksBufferSpec::source_interop_ = "OPTIX_INPUT_ONLY"  ; 

#endif



const char* OpticksBufferSpec::record_compute_ = "OPTIX_OUTPUT_ONLY"  ;
const char* OpticksBufferSpec::record_interop_ = "OPTIX_OUTPUT_ONLY"  ;

const char* OpticksBufferSpec::sequence_compute_ = "OPTIX_NON_INTEROP,OPTIX_OUTPUT_ONLY" ;
const char* OpticksBufferSpec::sequence_interop_ = "OPTIX_NON_INTEROP,OPTIX_OUTPUT_ONLY" ;

const char* OpticksBufferSpec::seed_compute_    = "OPTIX_NON_INTEROP,OPTIX_INPUT_ONLY"  ;
const char* OpticksBufferSpec::seed_interop_    = "OPTIX_NON_INTEROP,OPTIX_INPUT_ONLY"  ;




// OptiX never sees: hit, phosel or recsel, 
//    phosel, recsel : written by Thrust OpIndexer, read by OpenGL shaders to do record (and photon) selection
//             hit   : written by OEvent::downloadSelection 

const char* OpticksBufferSpec::hit_compute_ = ""  ;
const char* OpticksBufferSpec::hit_interop_ = ""  ;

const char* OpticksBufferSpec::nopstep_compute_ = ""  ;
const char* OpticksBufferSpec::nopstep_interop_ = ""  ;

const char* OpticksBufferSpec::phosel_compute_ = ""  ;
const char* OpticksBufferSpec::phosel_interop_ = ""  ;

const char* OpticksBufferSpec::recsel_compute_ = ""  ;
const char* OpticksBufferSpec::recsel_interop_ = ""  ;






const char* OpticksBufferSpec::Get(const char* name, bool compute )
{
    const char* bspc = NULL ; 
    if(     strcmp(name, OpticksEvent::genstep_)==0)  bspc = compute ? genstep_compute_ : genstep_interop_ ; 
    else if(strcmp(name, OpticksEvent::nopstep_)==0)  bspc = compute ? nopstep_compute_ : nopstep_interop_ ; 
    else if(strcmp(name, OpticksEvent::photon_)==0)   bspc = compute ? photon_compute_  : photon_interop_ ;
    else if(strcmp(name, OpticksEvent::source_)==0)   bspc = compute ? source_compute_  : source_interop_ ;
    else if(strcmp(name, OpticksEvent::record_)==0)   bspc = compute ? record_compute_  : record_interop_ ;
    else if(strcmp(name, OpticksEvent::phosel_)==0)   bspc = compute ? phosel_compute_  : phosel_interop_ ;
    else if(strcmp(name, OpticksEvent::recsel_)==0)   bspc = compute ? recsel_compute_  : recsel_interop_ ;
    else if(strcmp(name, OpticksEvent::sequence_)==0) bspc = compute ? sequence_compute_  : sequence_interop_ ;
    else if(strcmp(name, OpticksEvent::seed_)==0)     bspc = compute ? seed_compute_  : seed_interop_ ;
    else if(strcmp(name, OpticksEvent::hit_)==0)      bspc = compute ? hit_compute_  : hit_interop_ ;
    return bspc ; 
}

