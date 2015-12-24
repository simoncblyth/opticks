// opop-
#include "OpIndexerApp.hh"
#include "OpIndexer.hh"

// opticks-
#include "Opticks.hh"
#include "OpticksCfg.hh"

// npy-
#include "NumpyEvt.hpp"
#include "NLog.hpp"


void OpIndexerApp::init()
{
    m_log = new NLog("OpIndexerApp.log", "info");
    m_opticks = new Opticks();
    m_cfg = m_opticks->getCfg();

    m_indexer = new OpIndexer();
} 


void OpIndexerApp::configure(int argc, char** argv)
{
    m_log->configure(argc, argv);
    m_log->init("/tmp");

    m_cfg->commandline(argc, argv); 

    LOG(debug) << "OpIndexerApp::configure" ; 

    m_evt = m_opticks->makeEvt();
    m_evt->Summary("OpIndexerApp::configure");

    m_indexer->setEvt(m_evt);
}


void OpIndexerApp::loadEvtFromFile(bool verbose)
{
    m_evt->load(verbose);

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

    m_evt->prepareForIndexing();

    m_indexer->indexSequence();

    m_evt->saveIndex(true);
}







