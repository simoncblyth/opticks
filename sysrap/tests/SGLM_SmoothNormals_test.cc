/**
SGLM_SmoothNormals_test.cc
=============================

::

    ~/o/sysrap/tests/SGLM_SmoothNormals_test.sh

* https://computergraphics.stackexchange.com/questions/4031/programmatically-generating-vertex-normals

* https://iquilezles.org/articles/normals/


HUH: U4Mesh/Geant4 with G4Orb is coming up with a dud last vertex 
at origin ? Causing nan for the normal as the vertex not 
used by the tri. 






**/


#include "NPFold.h"
#include "SGLM.h"

int main()
{
    NPFold* fold = NPFold::Load("$MESH_FOLD"); 
    const NP* a_vtx = fold->get("vtx"); 
    const NP* a_tri = fold->get("tri"); 
    int num_vtx = a_vtx ? a_vtx->shape[0] : 0 ; 
    int num_tri = a_tri ? a_tri->shape[0] : 0 ; 

    std::cout 
        << " fold " << ( fold ? fold->desc() : "-" )
        << " a_vtx "  << ( a_vtx ? a_vtx->sstr() : "-" )
        << " a_tri "  << ( a_tri ? a_tri->sstr() : "-" )
        << " num_vtx " << num_vtx
        << " num_tri " << num_tri
        << std::endl
        ;   

    typedef glm::tvec3<double> D3 ; 
    typedef glm::tvec3<float>  F3 ; 
    typedef glm::tvec3<int>    I3 ; 

    assert( sizeof(D3) == sizeof(double)*3 ); 
    assert( sizeof(F3) == sizeof(float)*3 ); 
    assert( sizeof(I3) == sizeof(int)*3 ); 

    std::vector<D3> vtx(num_vtx) ; 
    std::vector<I3> tri(num_tri) ; 
    std::vector<D3> nrm(num_vtx, {0,0,0}) ;


    std::cout 
         << " a_vtx.arr_bytes " << a_vtx->arr_bytes() 
         << " vtx.size " << vtx.size()
         << " sizeof(D3)*vtx.size() " << sizeof(D3)*vtx.size()
         << std::endl 
         ; 

    assert( sizeof(D3)*vtx.size() == a_vtx->arr_bytes() ); 
    assert( sizeof(I3)*tri.size() == a_tri->arr_bytes() ); 

    memcpy( vtx.data(), a_vtx->bytes(), a_vtx->arr_bytes() ); 
    memcpy( tri.data(), a_tri->bytes(), a_tri->arr_bytes() ); 

    for(int i=0 ; i < num_tri ; i++)
    {
        const I3& t = tri[i] ; 
        assert( t.x > -1 && t.x < num_vtx ); 
        assert( t.y > -1 && t.y < num_vtx ); 
        assert( t.z > -1 && t.z < num_vtx ); 

        
        
        D3& v0 = vtx[t.x] ; 
        D3& v1 = vtx[t.y] ; 
        D3& v2 = vtx[t.z] ; 

        D3 n = glm::cross(v1-v0, v2-v0) ;

        nrm[t.x] += n ; 
        nrm[t.y] += n ; 
        nrm[t.z] += n ; 

        // see https://iquilezles.org/articles/normals/
        // this is a cunning technique for smooth normals
        // weighted by tri area (because did not normalize *n*)

        std::cout 
            << "["
            << std::setw(3) << t.x
            << "," 
            << std::setw(3) << t.y
            << "," 
            << std::setw(3) << t.z 
            << "]"
            << std::endl
            ;

    }
    for(int i=0 ; i < num_vtx ; i++) nrm[i] = glm::normalize( nrm[i] ); 

    for(int i=0 ; i < num_vtx ; i++)
    {
        D3& n = nrm[i] ; 
        std::cout 
            << "["
            << std::setw(3) << i
            << "]"
            << "["
            << std::setw(10) << std::fixed << std::setprecision(4) << n.x
            << ","
            << std::setw(10) << std::fixed << std::setprecision(4) << n.y
            << ","
            << std::setw(10) << std::fixed << std::setprecision(4) << n.z
            << "]"
            << std::endl
            ;
    }


    return 0 ;  
}
