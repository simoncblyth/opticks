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

#include "Opticks.hh"
#include "NCSG.hpp"
#include "GMeshLib.hh"
#include "GMesh.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    ok.configure();

    GMeshLib* ml = GMeshLib::Load(&ok);

    const char* dbgmesh = ok.getDbgMesh();
    if(dbgmesh)
    {
        bool startswith = true ; 
        const GMesh* mesh = ml->getMesh(dbgmesh, startswith);
        mesh->dump("GMesh::dump", 50);

        const NCSG* solid = mesh->getCSG(); 
        assert( solid );     
        solid->dump();  

    }
    else
    {
        LOG(info) << "no dbgmesh" ; 
        ml->dump();
    }

    return 0 ; 
}

