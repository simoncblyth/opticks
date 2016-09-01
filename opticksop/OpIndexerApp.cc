#include <cstddef>

// opop-
#include "OpIndexerApp.hh"
#include "OpIndexer.hh"


// opticks-
#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "OpticksEvent.hh"

// opticksgeo-
#include "OpticksHub.hh"


// npy-
#include "PLOG.hh"



OpIndexerApp::OpIndexerApp(int argc, char** argv) 
   :   
     m_opticks(new Opticks(argc, argv)),
     m_hub(new OpticksHub(m_opticks)),
     m_cfg(m_opticks->getCfg()),
     m_indexer(new OpIndexer(m_hub, NULL))
{
    init();
}


void OpIndexerApp::init()
{
} 


void OpIndexerApp::configure()
{
    m_hub->configure();

    LOG(debug) << "OpIndexerApp::configure" ; 
}


void OpIndexerApp::loadEvtFromFile()
{
    m_opticks->setSpaceDomain(0.f,0.f,0.f,1000.f);  // this is required before can create an evt 

    m_hub->loadPersistedEvent();

    OpticksEvent* evt = m_hub->getEvent();
    evt->Summary("OpIndexerApp::configure");
 
    if(evt->isNoLoad())
    {    
        LOG(info) << "App::loadEvtFromFile LOAD FAILED " ;
        return ; 
    }    

}

void OpIndexerApp::makeIndex()
{
    OpticksEvent* evt = m_hub->getEvent();
    if(evt->isIndexed())
    {
        bool forceindex = m_opticks->hasOpt("forceindex");
        if(forceindex)
        {
            LOG(info) << "OpIndexerApp::makeIndex evt is indexed already, but --forceindex option in use, so proceeding..." ;
        }
        else
        {
            LOG(info) << "OpIndexerApp::makeIndex evt is indexed already, SKIPPING " ;
            return  ;
        }
    }
    if(evt->isNoLoad())
    {
        LOG(info) << "OpIndexerApp::makeIndex evt failed to load, SKIPPING " ;
        return  ;
    }


    evt->Summary("OpIndexerApp::makeIndex");

    //evt->prepareForIndexing();

    m_indexer->indexSequence();

    evt->saveIndex(true);
}




