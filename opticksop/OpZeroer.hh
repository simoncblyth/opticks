#pragma once

#include <cstddef>

class NumpyEvt ; 
class OPropagator ; 
class OContext ; 


class OpZeroer {
   public:
      OpZeroer(OContext* ocontext);
   public:
      void setEvt(NumpyEvt* evt);
      void setPropagator(OPropagator* propagator);
   public:
      void zeroRecords();
   private:
      void zeroRecordsViaOpenGL();
      void zeroRecordsViaOptiX();
   private:
      OContext*                m_ocontext ;
      NumpyEvt*                m_evt ;
      OPropagator*             m_propagator ;
};

inline OpZeroer::OpZeroer(OContext* ocontext)  
   :
     m_ocontext(ocontext),
     m_evt(NULL),
     m_propagator(NULL)
{
}

inline void OpZeroer::setEvt(NumpyEvt* evt)
{
    m_evt = evt ; 
}  
inline void OpZeroer::setPropagator(OPropagator* propagator)
{
    m_propagator = propagator ; 
}  



