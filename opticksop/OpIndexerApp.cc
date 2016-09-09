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
#include "OpticksRun.hh"


// npy-
#include "PLOG.hh"



OpIndexerApp::OpIndexerApp(int argc, char** argv) 
   :   
     m_opticks(new Opticks(argc, argv)),
     m_hub(new OpticksHub(m_opticks)),
     m_cfg(m_opticks->getCfg()),
     m_indexer(new OpIndexer(m_hub, NULL)),
     m_run(m_hub->getRun())
{
    init();
}


void OpIndexerApp::init()
{
} 


void OpIndexerApp::configure()
{
    LOG(debug) << "OpIndexerApp::configure" ; 
}


void OpIndexerApp::loadEvtFromFile()
{
    m_opticks->setSpaceDomain(0.f,0.f,0.f,1000.f);  // this is required before can create an evt 

    m_run->loadEvent();

    OpticksEvent* evt = m_run->getEvent();
    evt->Summary("OpIndexerApp::configure");
 
    if(evt->isNoLoad())
    {    
        LOG(info) << "App::loadEvtFromFile LOAD FAILED " ;
        return ; 
    }    

}

void OpIndexerApp::makeIndex()
{
    OpticksEvent* evt = m_run->getEvent();
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




