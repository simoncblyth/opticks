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


int main(int argc, char* argv[])
{
    inilog();

    const char* prefix = argc > 1 ? argv[1] : "ASSIMPWRAP_" ;
    GCache cache(prefix);
    cache.Summary();
    
    const char* path  = cache.getPath();
    const char* query = cache.getQuery();
    const char* ctrl  = cache.getCtrl();

    LOG(info) << argv[0]
              << " prefix " << prefix 
              << " query " << query 
              << " ctrl " << ctrl 
              ; 

    LOG(info) << "AssimpWrapTest"
              << " path " << path 
              ;

    assert(path);
    assert(query);
    assert(ctrl);

    AssimpGeometry ageo(path);
    ageo.import();
    AssimpSelection* selection = ageo.select(query);

    AssimpGGeo agg(ageo.getTree(), selection); 
    GGeo* ggeo = agg.convert(ctrl);

    ggeo->Summary("main");    
    GBoundaryLib* lib = ggeo->getBoundaryLib();
    lib->Summary("GBoundaryLib");


    // needs to be sensitized first, otherwise get sensor assert
    //GMergedMesh* mm = ggeo->makeMergedMesh();
    //mm->Summary("GMergedMesh");

    return 0 ; 
}
    

