// TEST=GMeshTest om-t


#include <cassert>

#include "GLMFormat.hpp"
#include "GMesh.hh"

#include "OPTICKS_LOG.hh"


GMesh* make_trapezoid(float zoffset=0.f, float z=20.f, float x1=20.f, float y1=20.f, float x2=20.f, float y2=20.f ) 
{
   /*  
    z-order verts


                  6----------7
                 /|         /|
                / |        / |
               4----------5  |
               |  |       |  |                       
               |  |       |  |         Z    
               |  2-------|--3         |  Y
               | /        | /          | /
               |/         |/           |/
               0----------1            +------ X
                         

    x1: x length at -z
    y1: y length at -z

    x2: x length at +z
    y2: y length at +z

    z:  z length

    */


    unsigned num_vertices = 8 ; 
    gfloat3* v = new gfloat3[num_vertices] ; 

    float dz = zoffset  ; 
                                                 // ZYX
    v[0] = { -x1/2.f, -y1/2.f , -z/2.f + dz } ;  // 000
    v[1] = {  x1/2.f, -y1/2.f , -z/2.f + dz } ;  // 001 
    v[2] = { -x1/2.f,  y1/2.f , -z/2.f + dz } ;  // 010
    v[3] = {  x1/2.f,  y1/2.f , -z/2.f + dz } ;  // 011

    v[4] = { -x2/2.f, -y2/2.f ,  z/2.f + dz } ;  // 100
    v[5] = {  x2/2.f, -y2/2.f ,  z/2.f + dz } ;  // 101
    v[6] = { -x2/2.f,  y2/2.f ,  z/2.f + dz } ;  // 110
    v[7] = {  x2/2.f,  y2/2.f ,  z/2.f + dz } ;  // 111

    // hmm to do normals properly need to dismember into 6 quads, 1 for each face ?
    gfloat3* n = new gfloat3[num_vertices] ; 

    n[0] = { 0.f, -1.f,  0.f } ; 
    n[1] = { 0.f, -1.f,  0.f } ; 
    n[2] = { 0.f,  1.f,  0.f } ; 
    n[3] = { 0.f,  1.f,  0.f } ; 
    n[4] = { 0.f, -1.f,  0.f } ; 
    n[5] = { 0.f, -1.f,  0.f } ; 
    n[6] = { 0.f,  1.f,  0.f } ; 
    n[7] = { 0.f,  1.f,  0.f } ; 

    unsigned num_faces = 12 ; 
    guint3* f = new guint3[num_faces] ; 

    f[ 0] = { 0, 1, 5 } ;   // -Y
    f[ 1] = { 0, 5, 5 } ;   // -Y
    f[ 2] = { 4, 7, 6 } ;   // +Z
    f[ 3] = { 4, 5, 7 } ;   // +Z
    f[ 4] = { 1, 7, 5 } ;   // +X 
    f[ 5] = { 1, 3, 7 } ;   // +X
    f[ 6] = { 2, 6, 7 } ;   // +Y
    f[ 7] = { 2, 7, 3 } ;   // +Y
    f[ 8] = { 0, 2, 3 } ;   // -Z
    f[ 9] = { 0, 3, 1 } ;   // -Z
    f[10] = { 0, 4, 6 } ;   // -X    
    f[11] = { 0, 6, 2 } ;   // -X 


    unsigned index = 0 ; 
    gfloat3* vertices = v ; 
    gfloat3* normals  = n ; 
    gfloat2* texcoords  = NULL ; 

    guint3* faces = f ;  

    //for( unsigned i=0 ; i < num_vertices ; i++) LOG(info) << i << " " << (vertices + i )->desc() ;   
    GMesh* mesh = new GMesh( index, 
                             vertices, num_vertices,
                             faces   , num_faces,  
                             normals, 
                             texcoords );   

    return mesh ; 
}




void test_make_trapezoid()
{
    GMesh* a = make_trapezoid(0.f); 
    GMesh* b = make_trapezoid(100.f); 

    a->Summary("a 0.f");
    a->dump("a", 10);

    b->Summary("b 100.f");
    b->dump("b", 10);
}

void test_applyTranslation_centering()
{
    GMesh* c = make_trapezoid(100.f); 
    glm::vec4 ce = c->getCE(0);  
    LOG(info) << " ce " << gformat(ce) ; 
    c->applyTranslation(-ce.x, -ce.y, -ce.z ); 
    c->dump("c", 10); 

    glm::vec4 ce2 = c->getCE(0); 
    LOG(info) << " ce2 " << gformat(ce2) ; 
    
    assert( ce2.x == 0.f ); 
    assert( ce2.y == 0.f ); 
    assert( ce2.z == 0.f ); 
}

void test_applyCentering()
{
    float zoffset = 100.f ; 
    GMesh* c = make_trapezoid(zoffset); 
    glm::vec4 ce = c->getCE(0);  
    LOG(info) << " ce " << gformat(ce) ; 
    assert( ce.z == zoffset  ); 

    c->applyCentering(); 
    glm::vec4 ce2 = c->getCE(0);  
    LOG(info) << " ce2 " << gformat(ce2) ; 
    assert( ce2.z == 0.f ); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_make_trapezoid(); 
    //test_applyTranslation_centering();
    test_applyCentering();

    return 0 ;
}
