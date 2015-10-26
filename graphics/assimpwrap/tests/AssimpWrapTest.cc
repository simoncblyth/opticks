/*
Setup envvars and run with:

   assimpwrap-test 

Comparing with pycollada

   g4daenode.sh -i --daepath dyb_noextra

*/

#include "AssimpGeometry.hh"
#include "AssimpTree.hh"
#include "AssimpNode.hh"
#include "AssimpGGeo.hh"

#include "GCache.hh"
#include "GGeo.hh"
#include "GBoundaryLib.hh"
#include "GMergedMesh.hh"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>



#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
//#include <boost/log/utility/setup/file.hpp>
#include "boost/log/utility/setup.hpp"
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void inilog()
{
    boost::log::core::get()->set_filter
    (    
        boost::log::trivial::severity >= boost::log::trivial::info
    );   
}


// cf with App::loadGeometry and GLoader::load where the below is canonically done  

int main(int argc, char* argv[])
{
    inilog();

    GCache cache("GGEOVIEW_");
    cache.Summary();
    
    GGeo* m_ggeo = new GGeo(&cache);
    m_ggeo->setLoaderImp(&AssimpGGeo::load); 
    m_ggeo->loadFromG4DAE();
    m_ggeo->Summary("main");    

    m_ggeo->traverse();

    GBoundaryLib* blib = m_ggeo->getBoundaryLib();
    blib->Summary("GBoundaryLib");
    //blib->dumpSurfaces();


    GMergedMesh* mm = m_ggeo->makeMergedMesh();
    mm->Summary("GMergedMesh");

    GMaterialLib* mlib = m_ggeo->getMaterialLib();
    mlib->Summary();
    mlib->createBuffer();
    mlib->saveToCache();


    return 0 ; 
}
    

