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

#include "GMaterial.hh"
#include "GBoundaryLib.hh"
#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GScintillatorLib.hh"

#include "GMergedMesh.hh"

#include "NPY.hpp"


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
    mlib->save();

    GSurfaceLib* slib = m_ggeo->getSurfaceLib();
    slib->Summary();
    slib->dump();
    slib->save();

    // canonically done by GLoader::load
    m_ggeo->findScintillatorMaterials("SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB");
    GPropertyMap<float>* scint = dynamic_cast<GPropertyMap<float>*>(m_ggeo->getScintillatorMaterial(0));

    GScintillatorLib* sclib = m_ggeo->getScintillatorLib(); 
    sclib->add(scint);
    sclib->save();

    GBndLib* bnd = m_ggeo->getBndLib();
    bnd->dump();
    bnd->save();


    return 0 ; 
}
    

