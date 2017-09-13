#include "NParameters.hpp"
#include "NLookup.hpp"

#include "Opticks.hh"
#include "OpticksRun.hh"
#include "OpticksEvent.hh"
#include "OpticksActionControl.hh"

#include "G4StepNPY.hpp"

#include "PLOG.hh"

OpticksRun::OpticksRun(Opticks* ok) 
   :
   m_ok(ok),
   m_g4evt(NULL),
   m_evt(NULL),
   m_g4step(NULL),
   m_parameters(new NParameters) 
{
    OK_PROFILE("OpticksRun::OpticksRun");
}


std::string OpticksRun::brief() const 
{
    std::stringstream ss ; 
    ss << "Run " 
       ;
    return ss.str();
}


void OpticksRun::createEvent(unsigned tagoffset)
{
    m_ok->setTagOffset(tagoffset);
    // tagoffset is recorded with Opticks::setTagOffset within the makeEvent, but need it here before that 

    OK_PROFILE("OpticksRun::createEvent.BEG");

    m_g4evt = m_ok->makeEvent(false, tagoffset) ;
    m_evt = m_ok->makeEvent(true, tagoffset) ;

    LOG(trace) << m_g4evt->brief() << " " << m_g4evt->getShapeString() ;  
    LOG(trace) << m_evt->brief() << " " << m_evt->getShapeString() ;  
  

    m_evt->setSibling(m_g4evt);
    m_g4evt->setSibling(m_evt);

    std::string tstamp = m_g4evt->getTimeStamp();
    m_evt->setTimeStamp( tstamp.c_str() );        // align timestamps

    LOG(debug) << "OpticksRun::createEvent(" 
              << tagoffset 
              << ") " 
              << tstamp 
              << "[ "
              << " ok:" << m_evt->getId() << " " << m_evt->getDir() 
              << " g4:" << m_g4evt->getId() << " " << m_g4evt->getDir()
              << "] DONE "
              ; 

    OK_PROFILE("OpticksRun::createEvent.END");
}

void OpticksRun::resetEvent()
{
    OK_PROFILE("OpticksRun::resetEvent.BEG");
    m_g4evt->reset();
    m_evt->reset();
    OK_PROFILE("OpticksRun::resetEvent.END");
}


OpticksEvent* OpticksRun::getG4Event()
{
    return m_g4evt ; 
}
OpticksEvent* OpticksRun::getEvent()
{
    return m_evt ; 
}
OpticksEvent* OpticksRun::getCurrentEvent()
{
    bool g4 = m_ok->hasOpt("vizg4|evtg4") ;
    return g4 ? m_g4evt : m_evt ;
}


void OpticksRun::setGensteps(NPY<float>* gensteps)
{
    bool no_gensteps = gensteps == NULL ; 
    if(no_gensteps) LOG(fatal) << "OpticksRun::setGensteps given NULL gensteps" ; 
    assert(!no_gensteps); 

    LOG(info) << "OpticksRun::setGensteps " << gensteps->getShapeString() ;  

    assert(m_evt && m_g4evt && "must OpticksRun::createEvent prior to OpticksRun::setGensteps");

    bool progenitor=true ; 

    const char* oac_label = m_ok->isEmbedded() ? "GS_EMBEDDED" : NULL ; 
 
    importGenstepData(gensteps, oac_label) ;

    m_g4evt->setGenstepData(gensteps, progenitor);

    passBaton();  
}

void OpticksRun::passBaton()
{
    NPY<float>* nopstep = m_g4evt->getNopstepData() ;
    NPY<float>* genstep = m_g4evt->getGenstepData() ;

    LOG(info) << "OpticksRun::passBaton"
              << " nopstep " << nopstep
              << " genstep " << genstep
              ;


   //
   // Not-cloning as these buffers are not actually distinct 
   // between G4 and OK.
   //
   // Nopstep and Genstep should be treated as owned 
   // by the m_g4evt not the Opticks m_evt 
   // where the m_evt pointers are just weak reference guests 
   //

    m_evt->setNopstepData(nopstep);  
    m_evt->setGenstepData(genstep);
}

bool OpticksRun::hasGensteps()
{
   return m_evt->hasGenstepData() && m_g4evt->hasGenstepData() ; 
}


void OpticksRun::saveEvent()
{
    OK_PROFILE("OpticksRun::saveEvent.BEG");
    // they skip if no photon data
    if(m_g4evt)
    {
        m_g4evt->save();
    } 
    if(m_evt)
    {
        m_evt->save();
    } 
    OK_PROFILE("OpticksRun::saveEvent.END");
}

void OpticksRun::anaEvent()
{
    OK_PROFILE("OpticksRun::anaEvent.BEG");
    if(m_g4evt && m_evt )
    {
        m_ok->ana();
    }
    OK_PROFILE("OpticksRun::anaEvent.END");
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






void OpticksRun::importGenstepData(NPY<float>* gs, const char* oac_label)
{
    NParameters* gsp = gs->getParameters();
    m_parameters->append(gsp);

    gs->setBufferSpec(OpticksEvent::GenstepSpec(m_ok->isCompute()));

    assert(m_g4step == NULL && "OpticksRun::importGenstepData can only do this once ");
    m_g4step = new G4StepNPY(gs);    

    OpticksActionControl oac(gs->getActionControlPtr());
    if(oac_label)
    {
        LOG(debug) << "OpticksRun::importGenstepData adding oac_label " << oac_label ; 
        oac.add(oac_label);
    }


    LOG(debug) << "OpticksRun::importGenstepData"
               << brief()
               << " shape " << gs->getShapeString()
               << " " << oac.description("oac")
               ;

    if(oac("GS_LEGACY"))
    {
        translateLegacyGensteps(gs);
    }
    else if(oac("GS_EMBEDDED"))
    {
        std::cerr << "OpticksEvent::importGenstepData GS_EMBEDDED " << std::endl ; 
        translateLegacyGensteps(gs);
    }
    else if(oac("GS_TORCH"))
    {
        LOG(debug) << " checklabel of torch steps  " << oac.description("oac") ; 
        m_g4step->checklabel(TORCH); 
    }
    else if(oac("GS_FABRICATED"))
    {
        m_g4step->checklabel(FABRICATED); 
    }
    else
    {
        LOG(debug) << " checklabel of non-legacy (collected direct) gensteps  " << oac.description("oac") ; 
        m_g4step->checklabel(CERENKOV, SCINTILLATION);
    }

    m_g4step->countPhotons();

    LOG(debug) 
         << " Keys "
         << " TORCH: " << TORCH 
         << " CERENKOV: " << CERENKOV 
         << " SCINTILLATION: " << SCINTILLATION  
         << " G4GUN: " << G4GUN  
         ;

     LOG(debug) 
         << " counts " 
         << m_g4step->description()
         ;

}


G4StepNPY* OpticksRun::getG4Step()
{
    return m_g4step ; 
}


void OpticksRun::translateLegacyGensteps(NPY<float>* gs)
{
    OpticksActionControl oac(gs->getActionControlPtr());
    bool gs_torch = oac.isSet("GS_TORCH") ; 
    bool gs_legacy = oac.isSet("GS_LEGACY") ; 
    bool gs_embedded = oac.isSet("GS_EMBEDDED") ; 

    if(!(gs_legacy || gs_embedded)) return ; 

    assert(!gs_torch); // there are no legacy torch files ?


    if(gs->isGenstepTranslated() && gs_legacy) // gs_embedded needs translation relabelling every time
    {
        LOG(warning) << "OpticksRun::translateLegacyGensteps already translated and gs_legacy  " ;
        return ; 
    }


    std::cerr << "OpticksRun::translateLegacyGensteps"
              << " gs_legacy " << ( gs_legacy ? "Y" : "N" )
              << " gs_embedded " << ( gs_embedded ? "Y" : "N" )
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

    m_g4step->relabel(CERENKOV, SCINTILLATION); 

    // CERENKOV or SCINTILLATION codes are used depending on 
    // the sign of the pre-label 
    // this becomes the ghead.i.x used in cu/generate.cu
    // which dictates what to generate

    lookup->close("OpticksRun::translateLegacyGensteps GS_LEGACY");

    m_g4step->setLookup(lookup);   
    m_g4step->applyLookup(0, 2);  // jj, kk [1st quad, third value] is materialIndex

    // replaces original material indices with material lines
    // for easy access to properties using boundary_lookup GPU side

}





