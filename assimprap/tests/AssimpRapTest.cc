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

#include "BFile.hh"

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


#include "AssimpImporter.hh"
#include "AssimpTree.hh"
#include "AssimpNode.hh"
#include "AssimpGGeo.hh"


#include "PLOG.hh"

#include "SYSRAP_LOG.hh"
#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "ASIRAP_LOG.hh"


// cf with canonical : opticksgeo-/OpticksGeometry::loadGeometry()

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    SYSRAP_LOG__ ;
    BRAP_LOG__ ;
    NPY_LOG__ ;
    OKCORE_LOG__ ;
    GGEO_LOG__ ;
    ASIRAP_LOG__ ;

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

    unsigned verbosity = 0 ; 

    unsigned idx = 0 ; 
    GNode* base = NULL ;  // global transforms
    GNode* root = m_ggeo->getNode(0); 

    GMergedMesh* mm = m_ggeo->makeMergedMesh(idx, base, root, verbosity); // this only makes when not existing TODO:split 
    mm->Summary("GMergedMesh");

    GMaterialLib* mlib = m_ggeo->getMaterialLib();
    mlib->Summary();

    GSurfaceLib* slib = m_ggeo->getSurfaceLib();
    slib->Summary();

    GScintillatorLib* sclib = m_ggeo->getScintillatorLib();  // gets populated by GGeo::prepareScintillatorLib/GGeo::loadFromG4DAE
    sclib->Summary();

    GBndLib* bnd = m_ggeo->getBndLib();
    bnd->Summary();


    return 0 ; 
}
    

