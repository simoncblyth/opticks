#pragma once

class NumpyEvt ; 

class OpticksEvent {
   public:
       OpticksEvent(NumpyEvt* evt);
       void indexPhotonsCPU();
       NumpyEvt* getEvt();
   private:
       void init();
   private:
       NumpyEvt*  m_evt ;
};

inline OpticksEvent::OpticksEvent(NumpyEvt* evt)
  :
   m_evt(evt)
{
   init();
}

inline NumpyEvt* OpticksEvent::getEvt()
{
    return m_evt ; 
}
