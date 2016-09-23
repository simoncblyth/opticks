#include "Opticks.hh"
#include "OpticksRun.hh"
#include "OpticksEvent.hh"

#include "PLOG.hh"

OpticksRun::OpticksRun(Opticks* ok) 
   :
   m_ok(ok),
   m_g4evt(NULL),
   m_evt(NULL)
{
    OK_PROFILE("OpticksRun::OpticksRun");
}
void OpticksRun::createEvent(unsigned tagoffset)
{
    m_ok->setTagOffset(tagoffset);
    // tagoffset is recorded with Opticks::setTagOffset within the makeEvent, but need it here before that 

    OK_PROFILE("OpticksRun::createEvent.BEG");

    m_g4evt = m_ok->makeEvent(false, tagoffset) ;
    m_evt = m_ok->makeEvent(true, tagoffset) ;

    m_evt->setSibling(m_g4evt);
    m_g4evt->setSibling(m_evt);

    std::string tstamp = m_g4evt->getTimeStamp();
    m_evt->setTimeStamp( tstamp.c_str() );        // align timestamps

    LOG(info) << "OpticksRun::createEvent(" 
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

void OpticksRun::setGensteps(NPY<float>* gensteps)
{
    LOG(info) << "OpticksRun::setGensteps " << gensteps->getShapeString() ;  

    assert(m_evt && m_g4evt && "must OpticksRun::createEvent prior to OpticksRun::setGensteps");

    m_g4evt->setGenstepData(gensteps);

    passBaton();  
}

void OpticksRun::passBaton()
{
    NPY<float>* nopstep = m_g4evt->getNopstepData() ;
    NPY<float>* genstep = m_g4evt->getGenstepData() ;

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
}

