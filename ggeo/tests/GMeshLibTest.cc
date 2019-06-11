// TEST=GMeshLibTest om-t

/**
GMeshLibTest.cc
==================

Loads the GMeshLib include NCSG solids from geocache and
allows dumping.

In direct workflow only need one envvar, the OPTICKS_KEY to identify the 
geocache to load from and the "--envkey" option to switch on sensitivity to it::
 
   GMeshLibTest --envkey --dbgmesh sFasteners
 

In legacy workflow needed the op.sh script to set multiple envvars in order to 
find the geocache::
 
   op.sh --dsst --gmeshlib --dbgmesh near_top_cover_box0xc23f970  
   op.sh --dsst --gmeshlib --dbgmesh near_top_cover_box0x


dsst
    sets geometry selection envvars, defining the path to the geocache
gmeshlib
    used by op.sh script to pick this executable GMeshLibTest 
dbgmesh
    name of mesh to dump 

**/

#include <cassert>
#include <iostream>
#include <iomanip>

#include "Opticks.hh"
#include "NCSG.hpp"
#include "GMesh.hh"
#include "GMeshLib.hh"
#include "GMesh.hh"
#include "NBBox.hpp"
#include "NQuad.hpp"
#include "NNode.hpp"
#include "NGLMExt.hpp"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    ok.configure();

    GMeshLib* meshlib = GMeshLib::Load(&ok);

    unsigned num_mesh = meshlib->getNumMeshes(); 
    LOG(info) << " num_mesh " << num_mesh ; 

    for(unsigned i=0 ; i < num_mesh ; i++)
    {
        const GMesh* mesh = meshlib->getMesh(i); 
        const char* name = mesh->getName() ; 
        const NCSG* solid = mesh->getCSG(); 
        nbbox bba = solid->bbox_analytic(); // global frame bbox
        nvec4 ce = bba.center_extent() ; 

        nnode* root = solid->getRoot(); 

           


        if( root->transform && !root->transform->is_identity() ) LOG(info) << " tr " << *root->transform ; 


        std::cout  
            << std::setw(2) << i 
            << std::setw(45) << ( name ? name : "NULL" )
            << " bba " << bba.description()
            << " ce " << std::setw(25) << ce.desc()
            << " " << std::setw(2) << i 
            << std::endl    
            ; 
    }


/*
    const char* dbgmesh = ok.getDbgMesh();
    if(dbgmesh)
    {
        bool startswith = true ; 
        const GMesh* mesh = meshlib->getMesh(dbgmesh, startswith);
        mesh->dump("GMesh::dump", 50);

        const NCSG* solid = mesh->getCSG(); 
        assert( solid );     
        solid->dump();  

    }
    else
    {
        LOG(info) << "no dbgmesh" ; 
        meshlib->dump();
    }
*/

    return 0 ; 
}

