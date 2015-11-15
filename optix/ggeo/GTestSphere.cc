#include "GTestSphere.hh"

#include "GMesh.hh"
#include "GSolid.hh"
#include "GVector.hh"
#include "GMatrix.hh"
#include "NLog.hpp"

GSolid* GTestSphere::makeSolid(glm::vec4& spec, unsigned int meshindex, unsigned int nodeindex)
{
    LOG(debug) << "GTestSphere::makeSolid" ;

    GMesh* mesh = makeMesh(spec, meshindex);

    GMatrixF* transform = new GMatrix<float>();

    GSolid* solid = new GSolid(nodeindex, transform, mesh, UINT_MAX, NULL );     

    solid->setBoundary(0);     // unlike ctor these create arrays
    solid->setSensor( NULL );      

    return solid ; 
}

GMesh* GTestSphere::makeMesh(glm::vec4& spec, unsigned int meshindex)
{
    gfloat3* vertices = new gfloat3[NUM_VERTICES] ;
    guint3* faces = new guint3[NUM_FACES] ;
    gfloat3* normals = new gfloat3[NUM_VERTICES] ;

    tesselate(spec.w, vertices, faces, normals );

    GMesh* mesh = new GMesh(meshindex, vertices, NUM_VERTICES,  
                                       faces, NUM_FACES,    
                                       normals,  
                                       NULL ); // texcoords

    mesh->setColors(  new gfloat3[NUM_VERTICES]);
    mesh->setColor(0.5,0.5,0.5);  

    return mesh ; 
}


void GTestSphere::tesselate(float radius, gfloat3* vertices, guint3* faces, gfloat3* normals)
{


}

