#include "GTestBox.hh"
#include "GCache.hh"
#include "GBBoxMesh.hh"
#include "GMesh.hh"
#include "GSolid.hh"
#include "GVector.hh"
#include "GMatrix.hh"
#include "NLog.hpp"

GSolid* GTestBox::makeSolid(gbbox& bbox, unsigned int meshindex, unsigned int nodeindex)
{
    LOG(info) << "GTestBox::make" ;

    GMesh* mesh = makeMesh(bbox, meshindex);

    GMatrixF* transform = new GMatrix<float>();

    GSolid* solid = new GSolid(nodeindex, transform, mesh, UINT_MAX, NULL );     

    solid->setBoundary(0);     // unlike ctor these create arrays
    solid->setSensor( NULL );      

    return solid ; 
}

GMesh* GTestBox::makeMesh(gbbox& bbox, unsigned int meshindex)
{
    gfloat3* vertices = new gfloat3[NUM_VERTICES] ;
    guint3* faces = new guint3[NUM_FACES] ;
    gfloat3* normals = new gfloat3[NUM_VERTICES] ;

    GBBoxMesh::twentyfour(bbox, vertices, faces, normals );

    GMesh* mesh = new GMesh(meshindex, vertices, NUM_VERTICES,  
                                       faces, NUM_FACES,    
                                       normals,  
                                       NULL ); // texcoords

    mesh->setColors(  new gfloat3[NUM_VERTICES]);
    mesh->setColor(0.5,0.5,0.5);  

    return mesh ; 
}





