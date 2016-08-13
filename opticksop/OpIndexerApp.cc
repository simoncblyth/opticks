#include <cstddef>
// opop-
#include "OpIndexerApp.hh"
#include "OpIndexer.hh"

// opticks-
#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "OpticksEvent.hh"

// npy-
#include "PLOG.hh"



OpIndexerApp::OpIndexerApp(int argc, char** argv) 
   :   
     m_argc(argc),
     m_argv(argv),
     m_opticks(NULL),
     m_cfg(NULL),
     m_evt(NULL),
     m_indexer(NULL)
{
    init();
}


void OpIndexerApp::init()
{
    m_opticks = new Opticks(m_argc, m_argv);
    m_cfg = m_opticks->getCfg();

    m_indexer = new OpIndexer(m_opticks, NULL);
} 


void OpIndexerApp::configure()
{
    m_cfg->commandline(m_argc, m_argv); 

    LOG(debug) << "OpIndexerApp::configure" ; 

    m_evt = m_opticks->makeEvent();
    m_evt->Summary("OpIndexerApp::configure");

    m_indexer->setEvent(m_evt);
}


void OpIndexerApp::loadEvtFromFile(bool verbose)
{
    m_evt->loadBuffers(verbose);

    if(m_evt->isNoLoad())
    {    
        LOG(info) << "App::loadEvtFromFile LOAD FAILED " ;
        return ; 
    }    

}

void OpIndexerApp::makeIndex()
{
    if(m_evt->isIndexed())
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
    if(m_evt->isNoLoad())
    {
        LOG(info) << "OpIndexerApp::makeIndex evt failed to load, SKIPPING " ;
        return  ;
    }


    m_evt->Summary("OpIndexerApp::makeIndex");

    //m_evt->prepareForIndexing();

    m_indexer->indexSequence();

    m_evt->saveIndex(true);
}




