// op --assimp

/*
Setup envvars and run with:

   assimpwrap-test 

Comparing with pycollada

   g4daenode.sh -i --daepath dyb_noextra

*/



#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "NGLM.hpp"
#include "NPY.hpp"
#include "Opticks.hh"


#include "GDomain.hh"
#include "GAry.hh"
#include "GProperty.hh"
#include "GPropertyMap.hh"
#include "GMaterial.hh"
#include "GMaterialLib.hh"
#include "GBndLib.hh"
#include "GSurfaceLib.hh"
#include "GScintillatorLib.hh"
#include "GMergedMesh.hh"
#include "GGeo.hh"


#include "AssimpGeometry.hh"
#include "AssimpTree.hh"
#include "AssimpNode.hh"
#include "AssimpGGeo.hh"


#include "PLOG.hh"

#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "ASIRAP_LOG.hh"


// cf with App::loadGeometry and GLoader::load where the below is canonically done  

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    BRAP_LOG_ ;
    NPY_LOG_ ;
    OKCORE_LOG_ ;
    GGEO_LOG_ ;
    ASIRAP_LOG_ ;

    Opticks ok(argc, argv);

    LOG(info) << "ok" ;

    ok.configure();

    const char* daepath = ok.getDAEPath();

    if(!daepath)
    {
        LOG(error) << "NULL daepath" ;
        return 0 ; 
    } 


    GGeo* m_ggeo = new GGeo(&ok);
    printf("after gg\n");
    m_ggeo->setLoaderImp(&AssimpGGeo::load); 
    m_ggeo->loadFromG4DAE();
    m_ggeo->Summary("main");    

    m_ggeo->traverse();


    GMergedMesh* mm = m_ggeo->makeMergedMesh();
    mm->Summary("GMergedMesh");

    GMaterialLib* mlib = m_ggeo->getMaterialLib();
    mlib->Summary();
    mlib->save();

    GSurfaceLib* slib = m_ggeo->getSurfaceLib();
    //slib->dump();
    slib->save();


    GScintillatorLib* sclib = m_ggeo->getScintillatorLib();  // gets populated by GGeo::prepareScintillatorLib/GGeo::loadFromG4DAE
    sclib->save();


    GBndLib* bnd = m_ggeo->getBndLib();
    //bnd->dump();
    bnd->save();


    return 0 ; 
}
    

