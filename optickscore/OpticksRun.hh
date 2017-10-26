#pragma once 

#include <string>

class Opticks ; 
class OpticksEvent ; 
template <typename T> class NPY ; 
class G4StepNPY ; 
class NParameters ;

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
        OpticksEvent* getCurrentEvent(); // returns OK evt unless G4 option specified : --vizg4 or --evtg4 
        G4StepNPY*    getG4Step(); 
        std::string brief() const ;

        void setGensteps(NPY<float>* gs);
        bool hasGensteps();

        void createEvent(unsigned tagoffset=0);  
        void resetEvent();  
        void loadEvent();
        void saveEvent(); 
        void anaEvent(); // analysis based on saved evts 
    private:
        void annotateEvent(); 
        void importGenstepData(NPY<float>* gs, const char* oac_label=NULL);
        void translateLegacyGensteps(NPY<float>* gs);

    private:
        Opticks*         m_ok ; 
        OpticksEvent*    m_g4evt ; 
        OpticksEvent*    m_evt ; 
        G4StepNPY*       m_g4step ; 
        NParameters*     m_parameters ;


};
