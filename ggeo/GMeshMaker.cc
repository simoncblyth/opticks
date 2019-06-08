
#include "NPY.hpp"
#include "NGLM.hpp"
#include "GMesh.hh"
#include "GMeshMaker.hh"


GMesh* GMeshMaker::Make(NPY<float>* vtx3, NPY<unsigned>* tri3, unsigned meshindex)
{
    assert( vtx3->hasShape(-1,3) ); 
    assert( tri3->hasShape(-1,3) ); 

    unsigned nvert = vtx3->getShape(0); 
    unsigned nface = tri3->getShape(0); 

    assert( vtx3->getNumBytes(0) == sizeof(gfloat3)*nvert );
    assert( tri3->getNumBytes(0) == sizeof(guint3)*nface );

    gfloat3* vertices = new gfloat3[nvert] ;
    guint3*  faces    = new guint3[nface] ;

    vtx3->write( (void*)vertices );
    tri3->write( (void*)faces );

    gfloat3* normals = new gfloat3[nvert] ;

    for(unsigned f=0 ; f < nface ; f++)
    {
        guint3& tri = faces[f] ;

        // grab 3 floats by pointer offsetting 
        glm::vec3 v0 = glm::make_vec3( reinterpret_cast<float*>(vertices + tri.x) ) ;  
        glm::vec3 v1 = glm::make_vec3( reinterpret_cast<float*>(vertices + tri.y) ) ;  
        glm::vec3 v2 = glm::make_vec3( reinterpret_cast<float*>(vertices + tri.z) ) ;  

        glm::vec3 nrm = glm::normalize( glm::cross(v1-v0, v2-v0)) ;

        // Hmm the normals will be overwritten according to whichever face comes last.
        //
        // Hmm is that the reason for the inefficent vertex storage ?
        // to provide a distinct slot for the normal for each vertex from every face 
        // it appears in ?
        //
        // If that causes a problem will have to increase the vertex slots.

        gfloat3& nrm0 = normals[tri.x] ;
        gfloat3& nrm1 = normals[tri.y] ;
        gfloat3& nrm2 = normals[tri.z] ;

        nrm0.x = nrm.x ; 
        nrm0.y = nrm.y ; 
        nrm0.z = nrm.z ; 

        nrm1.x = nrm.x ; 
        nrm1.y = nrm.y ; 
        nrm1.z = nrm.z ; 

        nrm2.x = nrm.x ; 
        nrm2.y = nrm.y ; 
        nrm2.z = nrm.z ; 
    }

   
    GMesh* mesh = new GMesh(meshindex, vertices, nvert,  
                                       faces, nface,    
                                       normals,  
                                       NULL ); // texcoords

    mesh->setColors(  new gfloat3[nvert]);
    mesh->setColor(0.5,0.5,0.5);  


    // expedient workaround, as YOG needs NPY buffers
    // but the GMesh manages to loose them thanks to using GBuffer

    tri3->reshape(-1,1);

    mesh->m_x4src_vtx = vtx3 ;  
    mesh->m_x4src_idx = tri3 ;  

    return mesh ; 
}






GMesh* GMeshMaker::Make(NPY<float>* triangles, unsigned meshindex) // static
{
    unsigned int ni = triangles->getShape(0) ;
    unsigned int nj = triangles->getShape(1) ;
    unsigned int nk = triangles->getShape(2) ;
    assert( nj == 3 && nk == 3); 

    // June 2018 : 
    //   Putting all the vertices together like this does not profit from vertex reuse 
    //   that can get by splitting the vertices and indices. 
    //   BUT : it does provide separate normals slots for every face that a vertex 
    //   is on the edge.
    //

    float* bvals = triangles->getValues() ;

    unsigned int nface = ni ; 
    unsigned int nvert = ni*3 ;  // <-- means lots of vertices are duplicated 

    gfloat3* vertices = new gfloat3[nvert] ;
    guint3* faces = new guint3[nface] ;
    gfloat3* normals = new gfloat3[nvert] ;

    unsigned int v = 0 ; 
    unsigned int f = 0 ; 

    for(unsigned int i=0 ; i < nface ; i++)
    {
        guint3& tri = faces[f] ;

        // grab 3 floats from requisite points in triangles buffer 
        glm::vec3 v0 = glm::make_vec3(bvals + i*nj*nk + 0*nk) ;
        glm::vec3 v1 = glm::make_vec3(bvals + i*nj*nk + 1*nk) ;
        glm::vec3 v2 = glm::make_vec3(bvals + i*nj*nk + 2*nk) ;

        /*
                    v0
                   /  \
                  /    \
                 /      \
                /        \
               v1--------v2

              counter clockwise winding, normal should be outwards
         */

        glm::vec3 nrm = glm::normalize( glm::cross(v1-v0, v2-v0)) ;

        gfloat3& vtx0 = vertices[v+0] ;
        vtx0.x = v0.x ; 
        vtx0.y = v0.y ; 
        vtx0.z = v0.z ; 

        gfloat3& vtx1 = vertices[v+1] ;
        vtx1.x = v1.x ; 
        vtx1.y = v1.y ; 
        vtx1.z = v1.z ; 

        gfloat3& vtx2 = vertices[v+2] ;
        vtx2.x = v2.x ; 
        vtx2.y = v2.y ; 
        vtx2.z = v2.z ; 

        gfloat3& nrm0 = normals[v+0] ;
        gfloat3& nrm1 = normals[v+1] ;
        gfloat3& nrm2 = normals[v+2] ;

        // same normal for all three

        nrm0.x = nrm.x ; 
        nrm0.y = nrm.y ; 
        nrm0.z = nrm.z ; 

        nrm1.x = nrm.x ; 
        nrm1.y = nrm.y ; 
        nrm1.z = nrm.z ; 

        nrm2.x = nrm.x ; 
        nrm2.y = nrm.y ; 
        nrm2.z = nrm.z ; 

        v += 3 ; 

        tri.x = i*3 + 0 ; 
        tri.y = i*3 + 1 ; 
        tri.z = i*3 + 2 ; 

        f += 1 ; 
    }
   
    GMesh* mesh = new GMesh(meshindex, vertices, nvert,  
                                       faces, nface,    
                                       normals,  
                                       NULL ); // texcoords

    mesh->setColors(  new gfloat3[nvert]);
    mesh->setColor(0.5,0.5,0.5);  

    return mesh ; 
}


/**
GMeshMaker::MakeSphereLocal
----------------------------

Using the normalized position as the normal 
restricts triangles to being in spherelocal coordinates
ie the sphere must be centered at origin


**/

GMesh* GMeshMaker::MakeSphereLocal(NPY<float>* triangles, unsigned meshindex)
{
    unsigned int ni = triangles->getShape(0) ;
    unsigned int nj = triangles->getShape(1) ;
    unsigned int nk = triangles->getShape(2) ;
    assert( nj == 3 && nk == 3); 

    float* bvals = triangles->getValues() ;

    unsigned int nface = ni ; 
    unsigned int nvert = ni*3 ; 

    gfloat3* vertices = new gfloat3[nvert] ;
    guint3* faces = new guint3[nface] ;
    gfloat3* normals = new gfloat3[nvert] ;

    unsigned int v = 0 ; 
    unsigned int f = 0 ; 

    for(unsigned int i=0 ; i < nface ; i++)
    {
        guint3& tri = faces[f] ;
        for(unsigned int j=0 ; j < 3 ; j++)
        {
             float* ijv = bvals + i*nj*nk + j*nk ;

             gfloat3& vtx = vertices[v] ;
             gfloat3& nrm = normals[v] ;

             glm::vec3 vpos = glm::make_vec3( ijv )  ;
             glm::vec3 npos = glm::normalize(vpos) ; 

             nrm.x = npos.x ; 
             nrm.y = npos.y ; 
             nrm.z = npos.z ; 

             vtx.x = vpos.x ; 
             vtx.y = vpos.y ;
             vtx.z = vpos.z ;  

             v += 1 ; 
        }

        tri.x = i*3 + 0 ; 
        tri.y = i*3 + 1 ; 
        tri.z = i*3 + 2 ; 

        f += 1 ; 
    }
   
    GMesh* mesh = new GMesh(meshindex, vertices, nvert,  
                                       faces, nface,    
                                       normals,  
                                       NULL ); // texcoords

    mesh->setColors(  new gfloat3[nvert]);
    mesh->setColor(0.5,0.5,0.5);  

    return mesh ; 
}



