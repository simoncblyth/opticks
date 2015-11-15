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
    float radius = spec.w ; 

    NPY<float>* buf = NPY<float>::load("/tmp/icosahedron.npy");

    unsigned int nj = buf->getShape(1) ;
    unsigned int nk = buf->getShape(2) ;
    assert( nj == 3 && nk == 3); 

    unsigned int ntri = buf->getNumItems();
    unsigned int nvert = ntri*3 ; 

    gfloat3* vertices = new gfloat3[nvert] ;
    guint3* faces = new guint3[ntri] ;
    gfloat3* normals = new gfloat3[nvert] ;

    unsigned int v = 0 ; 
    unsigned int f = 0 ; 

    for(unsigned int i=0 ; i < ntri ; i++)
    {
        guint3& tri = faces[f] ;
        for(unsigned int j=0 ; j < 3 ; j++)
        {
             float* vals = buf->getValues() + i*nj*nk + j*nk ;

             gfloat3& vtx = vertices[v] ;
             gfloat3& nrm = normals[v] ;

             nrm.x = *(vals + 0); 
             nrm.y = *(vals + 1); 
             nrm.z = *(vals + 2); 

             vtx.x = nrm.x * radius ; 
             vtx.y = nrm.y * radius ;
             vtx.z = nrm.z * radius ;  

             v += 1 ; 
        }

        tri.x = i*3 + 0 ; 
        tri.y = i*3 + 1 ; 
        tri.z = i*3 + 2 ; 

        f += 1 ; 
    }
   

    GMesh* mesh = new GMesh(meshindex, vertices, nvert,  
                                       faces, ntri,    
                                       normals,  
                                       NULL ); // texcoords

    mesh->setColors(  new gfloat3[nvert]);
    mesh->setColor(0.5,0.5,0.5);  

    return mesh ; 
}


//  https://www.cosc.brocku.ca/Offerings/3P98/course/OpenGL/glut-3.7/progs/advanced/sphere.c



