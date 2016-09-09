#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksRun.hh"
#include "OpticksEvent.hh"

#include "PLOG.hh"

OpticksRun::OpticksRun(OpticksHub* hub) 
   :
   m_hub(hub),
   m_ok(hub->getOpticks()),
   m_g4evt(NULL),
   m_evt(NULL)
{
    init();
}

void OpticksRun::init()
{
}


void OpticksRun::createEvent()
{
    m_g4evt = m_ok->makeEvent(false) ;
    m_evt = m_ok->makeEvent(true) ;

    m_evt->setSibling(m_g4evt);
    m_g4evt->setSibling(m_evt);

    std::string tstamp = m_g4evt->getTimeStamp();
    m_evt->setTimeStamp( tstamp.c_str() );        // align timestamps
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
    assert(m_evt && m_g4evt && "must OpticksRun::createEvent prior to OpticksRun::setGensteps");

    m_g4evt->setGenstepData(gensteps);

    passBaton();  
}

void OpticksRun::passBaton()
{
    NPY<float>* nopstep = m_g4evt->getNopstepData() ;
    NPY<float>* genstep = m_g4evt->getGenstepData() ;

   // not-cloning as not actually distinct (?)
   // nopstep and genstep should be treated as owned 
   // by the m_g4evt not the Opticks m_evt 
   // where the pointers are just weak reference guests 

    m_evt->setNopstepData(nopstep);  
    m_evt->setGenstepData(genstep);
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
}

void OpticksRun::saveEvent()
{
    // TODO: detect if the event has been populated
    if(m_g4evt)
    {
        m_g4evt->dumpDomains("OpticksRun::saveEvent g4evt domains");
        m_g4evt->save();
    } 
    if(m_evt)
    {
        m_evt->dumpDomains("OpticksRun::saveEvent evt domains");
        m_evt->save();
    } 
}

