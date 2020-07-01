/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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
        void importGensteps();
    public:
        OpticksEvent* getEvent() const ;
        OpticksEvent* getG4Event() const ;
        OpticksEvent* getCurrentEvent(); // returns OK evt unless G4 option specified : --vizg4 or --evtg4 
        G4StepNPY*    getG4Step(); 
        std::string brief() const ;

        void setGensteps(NPY<float>* gs);
        bool hasGensteps() const ;

        void createEvent(NPY<float>* gensteps);
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
        NPY<float>*      m_gensteps ; 

        OpticksEvent*    m_g4evt ; 
        OpticksEvent*    m_evt ; 
        G4StepNPY*       m_g4step ; 
        NMeta*           m_parameters ;


};
