// opop-
#include "OpIndexerApp.hh"
#include "OpIndexer.hh"

// opticks-
#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "OpticksEvent.hh"

// npy-
#include "BLog.hh"


void OpIndexerApp::init()
{
    m_opticks = new Opticks(m_argc, m_argv);
    m_cfg = m_opticks->getCfg();

    m_indexer = new OpIndexer(NULL);
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
        LOG(info) << "OpIndexerApp::makeIndex evt is indexed already, SKIPPING " ;
        return  ;
    }

    m_evt->Summary("OpIndexerApp::makeIndex");

    //m_evt->prepareForIndexing();

    m_indexer->indexSequence();

    m_evt->saveIndex(true);
}




