
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
#include "NLog.hpp"

// ggeo-
#include "GMesh.hh"

//
#include "MWrap.hh"

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
typedef OpenMesh::TriMesh_ArrayKernelT<>  MyMesh;

int main(int argc, char** argv)
{
    Opticks ok(argc, argv, "openmeshrap.log");

    const char* idpath = ok.getIdPath();
    LOG(info) << "idpath " << ( idpath ? idpath : "NULL" ) ; 
    assert(idpath && "OpenMeshRapTest::main idpath is required");

    GMesh* gm = GMesh::load_deduped( idpath, "GMergedMesh/0" );
    gm->Summary();

    MWrap<MyMesh> ws(new MyMesh);
    ws.load(gm);

    int ncomp = ws.labelConnectedComponentVertices("component"); 
    printf("ncomp: %d \n", ncomp);


    if(ncomp != 2) return 1 ; 

    typedef MyMesh::VertexHandle VH ; 
    typedef std::map<VH,VH> VHM ;

    MWrap<MyMesh> wa(new MyMesh);
    MWrap<MyMesh> wb(new MyMesh);

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

    MWrap<MyMesh>::labelSpatialPairs( wa.getMesh(), wb.getMesh(), delta, "centroid", "paired");

    wa.deleteFaces("paired");
    wb.deleteFaces("paired");

    wa.collectBoundaryLoop();
    wb.collectBoundaryLoop();

    VHM a2b = MWrap<MyMesh>::findBoundaryVertexMap(&wa, &wb );  

    MWrap<MyMesh> wdst(new MyMesh);

    wdst.createWithWeldedBoundary( &wa, &wb, a2b );

    GMesh* result = wdst.createGMesh(); 
    result->setVersion("_v0");
    result->save( idpath, "GMergedMesh/0" );

    return 0;
}


