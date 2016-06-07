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

#include "GGeo.hh"

#include "GMaterial.hh"
#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GScintillatorLib.hh"

#include "GMergedMesh.hh"

#include "Opticks.hh"

#include "NPY.hpp"


#include <cstdio>
#include <cstdlib>
#include <cassert>


#include "NLog.hpp"


// cf with App::loadGeometry and GLoader::load where the below is canonically done  

int main(int argc, char** argv)
{
    Opticks ok(argc, argv, "assimpwrap.log");
    printf("after ok\n");


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
    

