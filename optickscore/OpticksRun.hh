#pragma once 

#include <string>

class Opticks ; 
class OpticksEvent ; 
template <typename T> class NPY ; 
class NPYBase ; 
class G4StepNPY ; 
class NMeta ; 

#include "plog/Severity.h"


/**
OpticksRun
===========

Dual G4/Opticks event handling with batton passing 
between g4evt and evt regarding the gensteps. 


**/



#include "OKCORE_API_EXPORT.hh"
class OKCORE_API OpticksRun 
{ 
        static const plog::Severity LEVEL ; 
    public:
        OpticksRun(Opticks* ok);
    private:
        void passBaton();
    public:
        OpticksEvent* getEvent() const ;
        OpticksEvent* getG4Event() const ;
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
        G4StepNPY* importGenstepData(NPY<float>* gs, const char* oac_label=NULL);
        void translateLegacyGensteps(G4StepNPY* g4step);
        bool hasActionControl(NPYBase* npy, const char* label);

    private:
        Opticks*         m_ok ; 
        OpticksEvent*    m_g4evt ; 
        OpticksEvent*    m_evt ; 
        G4StepNPY*       m_g4step ; 
#ifdef OLD_PARAMETERS
        X_BParameters*     m_parameters ;
#else
        NMeta*           m_parameters ;
#endif


};
