
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>

#include <glm/glm.hpp>

// opticks-
#include "Opticks.hh"
#include "OpticksResource.hh"

// npy-

// ggeo-
#include "GMesh.hh"

//
#include "MWrap.hh"
#include "MMesh.hh"
#include "PLOG.hh"
#include "GGEO_LOG.hh"
#include "MESHRAP_LOG.hh"





int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    GGEO_LOG__ ;
    MESHRAP_LOG__ ;

    LOG(info) << argv[0] ;

    Opticks ok(argc, argv);

    const char* idpath = ok.getIdPath();
    LOG(info) << "idpath " << ( idpath ? idpath : "NULL" ) ; 
    assert(idpath && "OpenMeshRapTest::main idpath is required");

    GMesh* gm = GMesh::load_deduped( idpath, "GMergedMesh/0" );
    LOG(info) << " after load_deduped " ;  

    if(!gm)
    {
        LOG(error) << "gm NULL " ;
        return 0 ;   
    } 

    gm->Summary();

    // huh : mm0 is nowadays the full non-instanced geometry, but mesh fixing 
    // must be applied to individual meshes that "should" be topologically (Eulers characteristic) 
    // correct meshes.



    MWrap<MMesh> ws(new MMesh);
    ws.load(gm);  // asserting in here

    int ncomp = ws.labelConnectedComponentVertices("component"); 
    printf("ncomp: %d \n", ncomp);


    if(ncomp != 2) return 1 ; 

    typedef MMesh::VertexHandle VH ; 
    typedef std::map<VH,VH> VHM ;

    MWrap<MMesh> wa(new MMesh);
    MWrap<MMesh> wb(new MMesh);

    VHM s2c_0 ;  
    ws.partialCopyTo(wa.getMesh(), "component", 0, s2c_0);

    VHM s2c_1 ;  
    ws.partialCopyTo(wb.getMesh(), "component", 1, s2c_1);

    ws.dump("ws",0);
    wa.dump("wa",0);
    wb.dump("wb",0);

    ws.dumpBounds("ws.dumpBounds");
    wb.dumpBounds("wb.dumpBounds");
    wa.dumpBounds("wa.dumpBounds");


    wa.write("/tmp/comp%d.off", 0 );
    wb.write("/tmp/comp%d.off", 1 );

    wa.calcFaceCentroids("centroid"); 
    wb.calcFaceCentroids("centroid"); 

    // xyz delta maximum and w: minimal dot product of normals, -0.999 means very nearly back-to-back
    //glm::vec4 delta(10.f, 10.f, 10.f, -0.999 ); 

    OpticksResource* resource = ok.getResource(); 
    glm::vec4 delta = resource->getMeshfixFacePairingCriteria();

    MWrap<MMesh>::labelSpatialPairs( wa.getMesh(), wb.getMesh(), delta, "centroid", "paired");

    wa.deleteFaces("paired");
    wb.deleteFaces("paired");

    wa.collectBoundaryLoop();
    wb.collectBoundaryLoop();

    VHM a2b = MWrap<MMesh>::findBoundaryVertexMap(&wa, &wb );  

    MWrap<MMesh> wdst(new MMesh);

    wdst.createWithWeldedBoundary( &wa, &wb, a2b );

    GMesh* result = wdst.createGMesh(); 
    result->setVersion("_v0");
    result->save( "/tmp", "GMergedMesh/0" );

    return 0;
}


