#pragma once 

class Opticks ; 
class OpticksEvent ; 
template <typename T> class NPY ; 

/**
OpticksRun
===========

Dual G4/Opticks event handling with batton passing 
between g4evt and evt regarding the gensteps. 


**/



#include "OKCORE_API_EXPORT.hh"
class OKCORE_API OpticksRun 
{ 
    public:
        OpticksRun(Opticks* ok);
    private:
        void passBaton();
    public:
        OpticksEvent* getEvent();
        OpticksEvent* getG4Event();
        OpticksEvent* getCurrentEvent(); // either G4 or OK evt depending on options

        void setGensteps(NPY<float>* gs);
        bool hasGensteps();

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
