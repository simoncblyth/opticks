#pragma once

class SLog ; 
template <typename T> class NPY ; 

class OpticksHub ; 
class Opticks ; 
class OpticksIdx ; 
class Opticks ; 

class OpEngine ; 
class OpTracer ; 

#include "OKOP_API_EXPORT.hh"
#include "OKOP_HEAD.hh"
#include "plog/Severity.h"

/**
OpPropagator : compute only propagator, no viz
==================================================

OpPropagator only used from OpMgr as m_propagator, which is used in
the G4Opticks approach, ie Opticks embedded inside an unsuspecting 
G4 example.   

Residents which are instanciated in ctor:

m_engine:OpEngine :
   control of GPU optical photon propagation

m_tracer:OpTracer 
   can make sequences of raytrace snapshots of geometry
   which can be saved to PPM files for subsequent conversion
   into PNG images or MP4 movies 


DevNotes
----------

Contrast with the viz enabled ok/OKPropagator

**/


class OKOP_API OpPropagator {
   public:
       static const plog::Severity LEVEL ; 
   public:
       OpPropagator(OpticksHub* hub, OpticksIdx* idx );
   public:
       void propagate();
       void cleanup();
       void snap();

   private:
       // invoked internally by propagate
       int uploadEvent();
       int downloadEvent();
   private:
       // not yet used 
       void indexEvent();
   private:
       SLog*          m_log ; 
       OpticksHub*    m_hub ; 
       OpticksIdx*    m_idx ; 
       Opticks*       m_ok ; 
       OpEngine*      m_engine ; 
       OpTracer*      m_tracer ; 
       int            m_placeholder ; 
       
};

#include "OKOP_TAIL.hh"



