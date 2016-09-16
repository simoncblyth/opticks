#pragma once 

class Opticks ; 
class OpticksEvent ; 
template <typename T> class NPY ; 


#include "OKGEO_API_EXPORT.hh"
class OKGEO_API OpticksRun 
{ 
    public:
        OpticksRun(Opticks* ok);
    private:
        void passBaton();
    public:
        OpticksEvent* getEvent();
        OpticksEvent* getG4Event();

        void setGensteps(NPY<float>* gs);

        void createEvent(unsigned tagoffset=0);  
        void resetEvent();  
        void loadEvent();
        void saveEvent(); 
        void anaEvent(); // analysis based on saved evts 
    private:
        Opticks*       m_ok ; 
        OpticksEvent*  m_g4evt ; 
        OpticksEvent*  m_evt ; 

};
