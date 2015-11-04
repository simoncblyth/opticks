//  ggv --testbox

#include "GCache.hh"
#include "GBndLib.hh"
#include "GMergedMesh.hh"
#include "GTestBox.hh"

int main(int argc, char** argv)
{
    GCache* cache = new GCache("GGEOVIEW_", "testbox.log", "info");
    cache->configure(argc, argv);

    GMergedMesh* mm = GMergedMesh::load(cache, 1);  // instance-1  triangulated PMT 5-solids
    mm->dumpSolids();




    GBndLib* blib = GBndLib::load(cache, true) ;

    GTestBox* box = new GTestBox(cache) ;

    box->setBndLib(blib);

    box->configure(); 


    gbbox bb = mm->getBBox(0);   
    
    unsigned int mm_index = 1000 ; 
    unsigned int mesh_index = 1000 ; 

    unsigned int node_index = mm->getNumSolids() ;   // 5 for DYB PMT
    // node indices need to be contiguous ?
    // TODO: how to find appropriate indices post cache ? 
    
    box->make(bb, mesh_index, node_index );

    GSolid* solid = box->getSolid();

    GMergedMesh* com = GMergedMesh::combine( mm_index, mm , solid );   

    com->Dump();





    return 1 ;
}
