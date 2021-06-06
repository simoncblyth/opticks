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


#include "BMeta.hh"

#include "NPY.hpp"
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
    m_parameters(new BMeta),
    m_resize(true),
    m_clone(true)
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
OpticksRun::createEvent with gensteps (static running)
------------------------------------------------------------

OpticksRun::createEvent is canonically invoked by the propagate methods 
of the high level managers, which happens long after geometry has been loaded.  
Specifically:

* OKG4Mgr::propagate
* OKMgr::propagate
* OpMgr::propagate

As gensteps are available the sizing of the event is 
entirely known ahread of time and buffers are sized accordingly.

* event is created and gensteps are set 

**/


void OpticksRun::createEvent(NPY<float>* gensteps, char ctrl) 
{
    unsigned tagoffset = gensteps ? gensteps->getArrayContentIndex() : 0 ;  // eg eventID
    LOG(LEVEL) << " tagoffset " << tagoffset << " ctrl [" << ctrl << "]" ; 
    createEvent(tagoffset, ctrl ); 
    setGensteps(gensteps, ctrl ); 
}




/**
OpticksRun::createEvent without gensteps (dynamic running)
--------------------------------------------------------------

1. creates separate OpticksEvent instances for recording of OK OR G4 
   propagations according to the g4_evt bool argument

2. when a prior event exists already with a negated tag it is 
   regarded as paired and the pair of events are associated as siblings 
   and the first event timestamp is adopted

2. invokes annotateEvent

This allows creation of OK and G4 events in either order. 

**/

void OpticksRun::createEvent(unsigned tagoffset, char ctrl )
{
    assert( ctrl == '+' || ctrl == '-' || ctrl == '='); 
    m_ok->setTagOffset(tagoffset);  // tagoffset is recorded with Opticks::setTagOffset within the makeEvent, but need it here before that 

    if(ctrl == '+' || ctrl == '=' )
    {
        m_evt = createOKEvent(tagoffset) ; 
        EstablishPairing( m_g4evt, m_evt, tagoffset ); 
        annotateEvent(m_evt);
    }

    if(ctrl == '-' || ctrl == '=' )
    {
        m_g4evt = createG4Event(tagoffset) ; 
        EstablishPairing( m_evt, m_g4evt, tagoffset ); 
        annotateEvent(m_g4evt);
    }

    LOG(LEVEL)
        << "(" 
        << tagoffset 
        << ") " 
        << "[ "
        << " ctrl: [" << ctrl << "]" 
        << " ok:" << ( m_evt ?   m_evt->brief()   : "-" )
        << " g4:" << ( m_g4evt ? m_g4evt->brief() : "-" )
        << "] DONE "
        ; 

}



OpticksEvent* OpticksRun::createOKEvent(unsigned tagoffset)
{
    bool is_ok_event = true ;  
    OpticksEvent* evt = m_ok->makeEvent(is_ok_event, tagoffset) ;

    unsigned skipaheadstep = m_ok->getSkipAheadStep() ; 
    unsigned skipahead =  tagoffset*skipaheadstep ; 
    LOG(info) 
        << " tagoffset " << tagoffset 
        << " skipaheadstep " << skipaheadstep
        << " skipahead " << skipahead
        ; 

    evt->setSkipAhead( skipahead ); // TODO: make configurable + move to ULL
    return evt ;  
}

OpticksEvent* OpticksRun::createG4Event(unsigned tagoffset)
{
    bool is_ok_event = false ;  
    OpticksEvent* g4evt = m_ok->makeEvent(is_ok_event, tagoffset) ;
    return g4evt ; 
}

/**
OpticksRun::EstablishPairing
------------------------------

When first and second events are paired, as idenified by their tags, then
certain properties from the first event are adopted by the second and 
sibling links are setup. 

**/

void OpticksRun::EstablishPairing(OpticksEvent* first, OpticksEvent* second, unsigned tagoffset)
{
    int first_tag  = first  ? first->getOffsetTagInteger(tagoffset) : 0 ; 
    int second_tag = second ? second->getOffsetTagInteger(tagoffset) : 0 ; 

    bool paired = first_tag != 0 && second_tag != 0 && first_tag + second_tag == 0 ; 

    if(paired)
    {
        std::string tstamp = first->getTimeStamp();
        second->setTimeStamp( tstamp.c_str() );   // align timestamps

        first->setSibling(second); 
        second->setSibling(first); 
    } 
}



/**
OpticksRun::annotateEvent
----------------------------

* huh : looks like Opticks parameters are not passed into OpticksEvent metadata ?

**/


void OpticksRun::annotateEvent(OpticksEvent* evt)
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

         evt->setTestCSGPath(testcsgpath);
         evt->setTestConfigString(geotestconfig);
    }
}



void OpticksRun::resetEvent(char ctrl)
{
    assert( ctrl == '+' || ctrl == '-' || ctrl == '='); 
    LOG(LEVEL) << "[ ctrl " << ctrl  ; 
    OK_PROFILE("_OpticksRun::resetEvent");

    if( m_evt && (ctrl == '+' || ctrl == '='))
    {
        m_evt->reset();
        delete m_evt ; 
        m_evt = NULL  ; 
    }

    if(m_g4evt && (ctrl == '-' || ctrl == '=')) 
    {
        m_g4evt->reset(); 
        delete m_g4evt ; 
        m_g4evt = NULL ; 
    }

    OK_PROFILE("OpticksRun::resetEvent");
    LOG(LEVEL) << "]" ; 
}



OpticksEvent* OpticksRun::getEvent(char ctrl) const 
{
    assert( ctrl == '+' || ctrl == '-' ); 
    OpticksEvent* evt = nullptr ; 
    switch(ctrl) 
    {
        case '+': evt = m_evt   ; break ; 
        case '-': evt = m_g4evt ; break ; 
    }
    return evt ; 
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
    char ctrl = m_ok->hasOpt("vizg4|evtg4") ? '-' : '+' ;
    return getEvent(ctrl) ; 
}



/**
OpticksRun::setGensteps
------------------------

THIS IS CALLED FROM VERY HIGH LEVEL IN OKMgr OR OKG4Mgr 

gensteps and maybe source photon data (via aux association) are lodged into m_g4evt
before passing baton (sharing pointers) with m_evt

**/
void OpticksRun::setGensteps(NPY<float>* gensteps, char ctrl)   // TODO: make this const : as gensteps are not owned by OpticksRun or OpticksEvent
{
    if(!gensteps) LOG(fatal) << "NULL gensteps" ; 
    LOG(LEVEL) 
         << "gensteps " << ( gensteps ? gensteps->getShapeString() : "NULL" )  ;  

    m_gensteps = gensteps ;   

    OK_PROFILE("_OpticksRun::setGensteps");

    if( ctrl == '+' || ctrl == '=' )
    {
        assert(m_evt && "must OpticksRun::createEvent prior to OpticksRun::setGensteps");
    }

    if( ctrl == '-' || ctrl == '=' )
    {
        assert(m_g4evt && "must OpticksRun::createEvent prior to OpticksRun::setGensteps");
    }

    importGensteps(ctrl);

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


The slightly reduced memory from having genstep, source and nopstep 
shared between m_g4evt and m_evt instances is not worth the 
complexity regarding ownership : especially when you consider the
various possibilites of not having a paired m_g4evt and loading and saving 
of events. Instead simplify by cloning these to make the paired OpticksEvent 
independent of each other.

Hmm: sharing of genstep, source and nopstep between g4evt and okevt
makes ownership too complicated when considering the varo
 
**/


void OpticksRun::importGensteps(char ctrl)
{
    OK_PROFILE("_OpticksRun::importGensteps");

    OpticksActionControl oac(m_gensteps->getActionControlPtr());

    LOG(LEVEL) 
        << " m_gensteps " << m_gensteps
        << " ctrl " << ctrl 
        << " oac.desc " << oac.desc("gs0") 
        << " oac.numSet " << oac.numSet() 
        ;

    const char* oac_label = m_ok->isEmbedded() ? "GS_EMBEDDED" : NULL ;   // TODO: bad assumption with input_photons

    LOG(LEVEL) << " oac_label " << oac_label ; 

    m_g4step = importGenstepData(m_gensteps, oac_label) ;

    if(m_g4evt && ( ctrl == '-' || ctrl == '=' ))
    { 
        m_g4evt->setGenstepData(m_gensteps, m_resize, m_clone );
    }

    if(m_evt && ( ctrl == '+' || ctrl == '=' ))
    {
        m_evt->setGenstepData(m_gensteps, m_resize, m_clone );
    }

    setupSourceData(ctrl); 

    if( m_g4evt && m_evt && ( ctrl == '='))
    {
        m_evt->setNopstepData( m_g4evt ? m_g4evt->getNopstepData() : NULL, m_clone );  
    }

    LOG(LEVEL) 
        << " oac.desc " << oac.desc("gs1") 
        << " oac.numSet " << oac.numSet() 
        ;


    OK_PROFILE("OpticksRun::importGensteps");
}


/**
OpticksRun::setupSourceData
-----------------------------

This is invoked from OpticksRun::importGensteps, it is an 
important part of how input source photons are provisioned
to both simulations.

**/
void OpticksRun::setupSourceData(char ctrl)
{
    if(hasActionControl(m_gensteps, "GS_EMITSOURCE"))
    {
        void* aux = m_gensteps->getAux();
        assert( aux );
        NPY<float>* emitsource = (NPY<float>*)aux ; 

        if(m_g4evt && (ctrl == '-' || ctrl == '=')) m_g4evt->setSourceData( emitsource, m_clone ); 
        if(m_evt   && (ctrl == '+' || ctrl == '=')) m_evt->setSourceData( emitsource, m_clone );

        LOG(LEVEL) 
            << "GS_EMITSOURCE"
            << " ctrl " << ctrl 
            << " emitsource " << emitsource->getShapeString()
            ;
    }
    else
    {
        NPY<float>* g4source = m_g4evt ? m_g4evt->getSourceData() : NULL ; 

        if( ctrl == '+' || ctrl == '=' )
        {
            m_evt->setSourceData( g4source, m_clone ) ;   
        }

    }
}





bool OpticksRun::hasGensteps() const
{
   //return m_evt->hasGenstepData() && m_g4evt->hasGenstepData() ; 
   return m_gensteps != NULL ; 
}


void OpticksRun::saveEvent(char ctrl)
{
    assert( ctrl == '+' || ctrl == '-' || ctrl == '='); 
    OK_PROFILE("_OpticksRun::saveEvent");
    // they skip if no photon data
    if(m_g4evt && ( ctrl == '-' || ctrl == '=') )
    {
        m_g4evt->save();
    } 
    if(m_evt && ( ctrl == '+' || ctrl == '=') )
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


/**

Formerly loaded m_g4evt when m_ok->hasOpt("vizg4"))
**/

void OpticksRun::loadEvent(char ctrl)
{
    assert( ctrl == '+' || ctrl == '-' || ctrl == '='); 
    if( ctrl == '=')
    {
        loadEvent('+'); 
        loadEvent('-'); 
    }
    else
    {
        unsigned tagoffset = 0 ; 
        createEvent(tagoffset, ctrl);

        OpticksEvent* evt = getEvent(ctrl); 
     
        bool verbose = false ; 
        evt->loadBuffers(verbose);

        if(evt->isNoLoad())
        {
            LOG(fatal) << "LOAD FAILED " ;
            m_ok->setExit(true);
        }

        if(m_ok->isExit()) exit(EXIT_FAILURE) ;
    }
}



/**
OpticksRun::importGenstepData
--------------------------------

Invoked by OpticksRun::importGensteps
The oac_label is only added if there is no prior label. 


The NPY<float> genstep buffer is wrapped in a G4StepNPY 
and metadata and labelling checks are done  and any material
translations performed.

OpticksActionControl is used to modify the NPYBase::m_action_control 
of the gensteps.

Why should allowed gencodes depend on the OpticksActionControl ? Surely 
what is allowed should depend solely on what is implemented in oxrap/cu/generate.cu ?
See notes/issues/G4StepNPY_checkGencodes_mismatch_assert.rst 

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
    BMeta* gsp = gs->getParameters() ;
    m_parameters->append(gsp);

    gs->setBufferSpec(OpticksEvent::GenstepSpec(m_ok->isCompute()));

    // assert(m_g4step == NULL && "OpticksRun::importGenstepData can only do this once ");
    G4StepNPY* g4step = new G4StepNPY(gs);    

    OpticksActionControl oac(gs->getActionControlPtr());

    if(oac_label)
    {
        if(oac.numSet() > 0 )
        {
            LOG(LEVEL) 
                << "NOT adding oac_label " << oac_label 
                << " as preexisting labels present: " << oac.desc()  
                ;
        }
        else
        {
            LOG(LEVEL) << "adding oac_label " << oac_label ; 
            oac.add(oac_label);
        }

    }

    LOG(LEVEL) 
        << brief()
        << " shape " << gs->getShapeString()
        << " " << oac.desc("oac")
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

        LOG(LEVEL) << " GS_EMBEDDED collected direct gensteps assumed translated at collection  " << oac.desc("oac") ; 
    }
    else if(oac("GS_TORCH"))
    {
        g4step->addAllowedGencodes(OpticksGenstep_TORCH); 
        LOG(LEVEL) << " checklabel of torch steps  " << oac.desc("oac") ; 
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
        LOG(LEVEL) << " checklabel of non-legacy (collected direct) gensteps  " << oac.desc("oac") ; 
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

/**
OpticksRun::translateLegacyGensteps
--------------------------------------

**/
 
void OpticksRun::translateLegacyGensteps(G4StepNPY* g4step)
{
    LOG(fatal) << "Attempting to elimate this" ; 
    assert( 0 ); 

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


