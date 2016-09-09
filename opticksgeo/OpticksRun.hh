#pragma once 

class OpticksHub ; 
class Opticks ; 
class OpticksEvent ; 
template <typename T> class NPY ; 


#include "OKGEO_API_EXPORT.hh"
class OKGEO_API OpticksRun 
{ 
    public:
        OpticksRun(OpticksHub* hub);
    private:
        void init();
        void passBaton();
    public:
        OpticksEvent* getEvent();
        OpticksEvent* getG4Event();

        void setGensteps(NPY<float>* gs);

        void createEvent(unsigned tagoffset=0);  
        void loadEvent();
        void saveEvent();
    private:
        OpticksHub*    m_hub ; 
        Opticks*       m_ok ; 
        OpticksEvent*  m_g4evt ; 
        OpticksEvent*  m_evt ; 

};
