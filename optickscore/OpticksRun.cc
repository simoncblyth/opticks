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



#include "NMeta.hpp"
#include "NLookup.hpp"

#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksRun.hh"
#include "OpticksEvent.hh"
#include "OpticksActionControl.hh"
#include "OpticksGenstep.hh"

#include "G4StepNPY.hpp"

#include "PLOG.hh"


const plog::Severity OpticksRun::LEVEL = PLOG::EnvLevel("OpticksRun", "DEBUG")  ; 


OpticksRun::OpticksRun(Opticks* ok) 
    :
    m_ok(ok),
    m_gensteps(NULL), 
    m_g4evt(NULL),
    m_evt(NULL),
    m_g4step(NULL),
    m_parameters(new NMeta)
{
    OK_PROFILE("OpticksRun::OpticksRun");
}


std::string OpticksRun::brief() const 
{
    std::stringstream ss ; 
    ss << "Run" 
       << " evt " << ( m_evt ? m_evt->brief() : "-" )
       << " g4evt " << ( m_g4evt ? m_g4evt->brief() : "-" )
       ;
    return ss.str();
}



/**
OpticksRun::createEvent
-------------------------

OpticksRun::createEvent is canonically invoked by the propagate methods 
of the high level managers, which happens long after geometry has been loaded.  
Specifically:

* OKG4Mgr::propagate
* OKMgr::propagate

This 

1. creates separate OpticksEvent instances for G4 and OK propagations, 
   associates them as siblings and adopts a common timestamp 

2. invokes annotateEvent



**/



void OpticksRun::createEvent(NPY<float>* gensteps) 
{
    unsigned tagoffset = gensteps ? gensteps->getArrayContentIndex() : 0 ;  // eg eventID
    LOG(LEVEL) << " tagoffset " << tagoffset ; 
    createEvent(tagoffset); 
    setGensteps(gensteps); 
}

void OpticksRun::createEvent(unsigned tagoffset)
{
    bool nog4propagate = m_ok->isNoG4Propagate() ;   // --nog4propagate

    m_ok->setTagOffset(tagoffset);
    // tagoffset is recorded with Opticks::setTagOffset within the makeEvent, but need it here before that 

    OK_PROFILE("_OpticksRun::createEvent");


    m_evt = m_ok->makeEvent(true, tagoffset) ;
    std::string tstamp = m_evt->getTimeStamp();

    if(nog4propagate) 
    { 
        m_g4evt = NULL ;   
    }
    else
    {  
        m_g4evt = m_ok->makeEvent(false, tagoffset) ;
        m_g4evt->setSibling(m_evt);
        m_g4evt->setTimeStamp( tstamp.c_str() );   // align timestamps
        m_evt->setSibling(m_g4evt);
    }

    LOG(LEVEL)
        << "(" 
        << tagoffset 
        << ") " 
        << tstamp 
        << "[ "
        << " ok:" << m_evt->brief() 
        << " g4:" << ( m_g4evt ? m_g4evt->brief() : "-" )
        << "] DONE "
        ; 

    annotateEvent();

    OK_PROFILE("OpticksRun::createEvent");
}





/**
OpticksRun::annotateEvent
----------------------------

* huh : looks like Opticks parameters are not passed into OpticksEvent metadata ?

**/


void OpticksRun::annotateEvent()
{
    // these are set into resource by GGeoTest::initCreateCSG during geometry loading
    OpticksResource* resource = m_ok->getResource();
    const char* testcsgpath = resource->getTestCSGPath();
    const char* geotestconfig = resource->getTestConfig();

    LOG(LEVEL) 
        << " testcsgpath " << ( testcsgpath ? testcsgpath : "-" )
        << " geotestconfig " << ( geotestconfig ? geotestconfig : "-" )
        ;

    if(testcsgpath)
    {  
         assert( geotestconfig ); 

         m_evt->setTestCSGPath(testcsgpath);
         m_evt->setTestConfigString(geotestconfig);

         if(m_g4evt)          
         { 
             m_g4evt->setTestCSGPath(testcsgpath);
             m_g4evt->setTestConfigString(geotestconfig);
         }
    }
}
void OpticksRun::resetEvent()
{
    LOG(LEVEL) << "[" ; 
    OK_PROFILE("_OpticksRun::resetEvent");
    m_evt->reset();
    if(m_g4evt) m_g4evt->reset(); 
    OK_PROFILE("OpticksRun::resetEvent");
    LOG(LEVEL) << "]" ; 
}


OpticksEvent* OpticksRun::getG4Event() const 
{
    return m_g4evt ; 
}
OpticksEvent* OpticksRun::getEvent() const 
{
    return m_evt ; 
}
OpticksEvent* OpticksRun::getCurrentEvent()
{
    bool g4 = m_ok->hasOpt("vizg4|evtg4") ;
    return g4 ? m_g4evt : m_evt ;
}


/**
OpticksRun::setGensteps
------------------------

THIS IS CALLED FROM VERY HIGH LEVEL IN OKMgr OR OKG4Mgr 

gensteps and maybe source photon data (via aux association) are lodged into m_g4evt
before passing baton (sharing pointers) with m_evt

**/
void OpticksRun::setGensteps(NPY<float>* gensteps)   // TODO: make this const : as gensteps are not owned by OpticksRun or OpticksEvent
{
    OK_PROFILE("_OpticksRun::setGensteps");
    assert(m_evt && "must OpticksRun::createEvent prior to OpticksRun::setGensteps");

    if(!gensteps) LOG(fatal) << "NULL gensteps" ; 
    //assert(gensteps); 

    LOG(LEVEL) << "gensteps " << ( gensteps ? gensteps->getShapeString() : "NULL" )  ;  

    m_gensteps = gensteps ;   

    if(m_gensteps) importGensteps();

    OK_PROFILE("OpticksRun::setGensteps");
}



/**
OpticksRun::importGensteps
----------------------------

Handoff from G4Event to Opticks event of the
Nopstep, Genstep and Source buffer pointers.
NB there is no cloning as these buffers are 
not distinct between Geant4 and Opticks

Nopstep and Genstep should be treated as owned 
by the m_g4evt not the Opticks m_evt 
where the m_evt pointers are just weak reference guests 
 
**/


void OpticksRun::importGensteps()
{
    OK_PROFILE("_OpticksRun::importGensteps");

    const char* oac_label = m_ok->isEmbedded() ? "GS_EMBEDDED" : NULL ; 
 
    m_g4step = importGenstepData(m_gensteps, oac_label) ;


    if(m_g4evt)
    { 
        bool progenitor=true ; 
        m_g4evt->setGenstepData(m_gensteps, progenitor);
    }

    m_evt->setGenstepData(m_gensteps);


    if(hasActionControl(m_gensteps, "GS_EMITSOURCE"))
    {
        void* aux = m_gensteps->getAux();
        assert( aux );

        NPY<float>* emitsource = (NPY<float>*)aux ; 

        if(m_g4evt) m_g4evt->setSourceData( emitsource ); 

        m_evt->setSourceData( emitsource);

        LOG(LEVEL) 
            << "GS_EMITSOURCE"
            << " emitsource " << emitsource->getShapeString()
            ;
    }
    else
    {
        m_evt->setSourceData( m_g4evt ? m_g4evt->getSourceData() : NULL ) ;   
    }

    m_evt->setNopstepData( m_g4evt ? m_g4evt->getNopstepData() : NULL );  
    OK_PROFILE("OpticksRun::importGensteps");
}



bool OpticksRun::hasGensteps() const
{
   //return m_evt->hasGenstepData() && m_g4evt->hasGenstepData() ; 
   return m_gensteps != NULL ; 
}


void OpticksRun::saveEvent()
{
    OK_PROFILE("_OpticksRun::saveEvent");
    // they skip if no photon data
    if(m_g4evt)
    {
        m_g4evt->save();
    } 
    if(m_evt)
    {
        m_evt->save();
    } 
    OK_PROFILE("OpticksRun::saveEvent");
}

void OpticksRun::anaEvent()
{
    OK_PROFILE("_OpticksRun::anaEvent");
    if(m_g4evt && m_evt )
    {
        m_ok->ana();
    }
    OK_PROFILE("OpticksRun::anaEvent");
}


void OpticksRun::loadEvent()
{
    createEvent();

    bool verbose ; 
    m_evt->loadBuffers(verbose=false);

    if(m_evt->isNoLoad())
    {
        LOG(fatal) << "OpticksRun::loadEvent LOAD FAILED " ;
        m_ok->setExit(true);
    }

    if(m_ok->isExit()) exit(EXIT_FAILURE) ;


    // hmm when g4 evt is selected, perhaps skip loading OK evt ?
    if(m_ok->hasOpt("vizg4"))
    {
        m_g4evt->loadBuffers(verbose=false);
        if(m_g4evt->isNoLoad())
        {
            LOG(fatal) << "OpticksRun::loadEvent LOAD g4evt FAILED " ;
            exit(EXIT_FAILURE);
        }
    }
}



/**
OpticksRun::importGenstepData
--------------------------------

The NPY<float> genstep buffer is wrapped in a G4StepNPY 
and metadata and labelling checks are done  and any material
translations performed.

**/


G4StepNPY* OpticksRun::importGenstepData(NPY<float>* gs, const char* oac_label)
{
    bool dbggsimport = m_ok->isDbgGSImport() ; // --dbggsimport
    if(dbggsimport)
    {
        const char* dbggsimport_path = "$TMP/OpticksRun_importGenstepData/dbggsimport.npy" ; 
        LOG(fatal) << "(--dbggsimport) saving gs to " << dbggsimport_path ; 
        gs->save(dbggsimport_path); 
    }    
 

    OK_PROFILE("_OpticksRun::importGenstepData");
    NMeta* gsp = gs->getParameters() ;
    m_parameters->append(gsp);

    gs->setBufferSpec(OpticksEvent::GenstepSpec(m_ok->isCompute()));

    // assert(m_g4step == NULL && "OpticksRun::importGenstepData can only do this once ");
    G4StepNPY* g4step = new G4StepNPY(gs);    

    OpticksActionControl oac(gs->getActionControlPtr());

    if(oac_label)
    {
        LOG(LEVEL) 
            << "adding oac_label " << oac_label ; 
        oac.add(oac_label);
    }

    LOG(LEVEL) 
        << brief()
        << " shape " << gs->getShapeString()
        << " " << oac.description("oac")
        ;

    if(oac("GS_LEGACY"))
    {
        translateLegacyGensteps(g4step);
    }
    else if(oac("GS_EMBEDDED"))
    {
        g4step->addAllowedGencodes( OpticksGenstep_G4Cerenkov_1042, OpticksGenstep_G4Scintillation_1042) ; 
        g4step->addAllowedGencodes( OpticksGenstep_DsG4Cerenkov_r3971, OpticksGenstep_DsG4Scintillation_r3971 ) ; 
        g4step->addAllowedGencodes( OpticksGenstep_TORCH);

        LOG(LEVEL) << " GS_EMBEDDED collected direct gensteps assumed translated at collection  " << oac.description("oac") ; 
    }
    else if(oac("GS_TORCH"))
    {
        g4step->addAllowedGencodes(OpticksGenstep_TORCH); 
        LOG(LEVEL) << " checklabel of torch steps  " << oac.description("oac") ; 
    }
    else if(oac("GS_FABRICATED"))
    {
        g4step->addAllowedGencodes(OpticksGenstep_FABRICATED); 
    }
    else if(oac("GS_EMITSOURCE"))
    {
        g4step->addAllowedGencodes(OpticksGenstep_EMITSOURCE); 
    }
    else
    {
        LOG(LEVEL) << " checklabel of non-legacy (collected direct) gensteps  " << oac.description("oac") ; 
        g4step->addAllowedGencodes( OpticksGenstep_G4Cerenkov_1042, OpticksGenstep_G4Scintillation_1042) ; 
        g4step->addAllowedGencodes( OpticksGenstep_DsG4Cerenkov_r3971, OpticksGenstep_DsG4Scintillation_r3971 ) ; 
        g4step->addAllowedGencodes( OpticksGenstep_EMITSOURCE ); 
    }
    g4step->checkGencodes();

    g4step->countPhotons();

    LOG(LEVEL) 
         << " Keys "
         << " OpticksGenstep_TORCH: " << OpticksGenstep_TORCH
         << " OpticksGenstep_G4Cerenkov_1042: " << OpticksGenstep_G4Cerenkov_1042
         << " OpticksGenstep_G4Scintillation_1042: " << OpticksGenstep_G4Scintillation_1042
         << " OpticksGenstep_DsG4Cerenkov_r3971: " << OpticksGenstep_DsG4Cerenkov_r3971
         << " OpticksGenstep_DsG4Scintillation_r3971: " << OpticksGenstep_DsG4Scintillation_r3971
         << " OpticksGenstep_G4GUN: " << OpticksGenstep_G4GUN  
         ;

     LOG(LEVEL) 
         << " counts " 
         << g4step->description()
         ;


    OK_PROFILE("OpticksRun::importGenstepData");
    return g4step ; 

}


G4StepNPY* OpticksRun::getG4Step()
{
    return m_g4step ; 
}



bool OpticksRun::hasActionControl(NPYBase* npy, const char* label)
{
    OpticksActionControl oac(npy->getActionControlPtr());
    return oac.isSet(label) ;
} 
 
void OpticksRun::translateLegacyGensteps(G4StepNPY* g4step)
{
    NPY<float>* gs = g4step->getNPY();

    OpticksActionControl oac(gs->getActionControlPtr());
    bool gs_torch = oac.isSet("GS_TORCH") ; 
    bool gs_legacy = oac.isSet("GS_LEGACY") ; 
    //bool gs_embedded = oac.isSet("GS_EMBEDDED") ; 

    if(!(gs_legacy)) return ; 

    assert(!gs_torch); // there are no legacy torch files ?

    if(gs->isGenstepTranslated() && gs_legacy)
    {
        LOG(warning) << "OpticksRun::translateLegacyGensteps already translated and gs_legacy  " ;
        return ; 
    }

    std::cerr << "OpticksRun::translateLegacyGensteps"
              << " gs_legacy " << ( gs_legacy ? "Y" : "N" )
              << std::endl 
              ;

    gs->setGenstepTranslated();

    NLookup* lookup = gs->getLookup();
    if(!lookup)
            LOG(fatal) << "OpticksRun::translateLegacyGensteps"
                       << " IMPORT OF LEGACY GENSTEPS REQUIRES gs->setLookup(NLookup*) "
                       << " PRIOR TO OpticksRun::setGenstepData(gs) "
                       ;

    assert(lookup); 

     // these should be appropriate for all legacy gensteps 
    int CK = OpticksGenstep_G4Cerenkov_1042 ; 
    int SI = OpticksGenstep_DsG4Scintillation_r3971 ;  
    g4step->relabel(CK, SI); 

    // Changes the old 1-based signed index into enum values  
    // that get read into ghead.i.x used in cu/generate.cu
    // dictating what to generate

    lookup->close("OpticksRun::translateLegacyGensteps GS_LEGACY");

    g4step->setLookup(lookup);   
    g4step->applyLookup(0, 2);  // jj, kk [1st quad, third value] is materialIndex

    // replaces original material indices with material lines
    // for easy access to properties using boundary_lookup GPU side

}


