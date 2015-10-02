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

    GCache cache(argc > 1 ? argv[1] : "ASSIMPWRAP_" );

    cache.Summary();
    
    GGeo* ggeo = new GGeo(&cache);

    int rc = AssimpGGeo::load(ggeo);

    assert(rc == 0);

    ggeo->Summary("main");    

    GBoundaryLib* lib = ggeo->getBoundaryLib();

    lib->Summary("GBoundaryLib");

    // loads .idmap sibling of G4DAE file and traverses nodes doing GSolid::setSensor for sensitve nodes
    ggeo->sensitize(cache.getIdPath(), "idmap");  

    GMergedMesh* mm = ggeo->makeMergedMesh();
    
    mm->Summary("GMergedMesh");


    return 0 ; 
}
    

