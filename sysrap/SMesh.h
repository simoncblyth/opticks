#pragma once

#include <glm/glm.hpp>
#include "NPFold.h"

struct SMesh 
{
    const NP* tri ; 
    const NP* _vtx ; 
    const NP* _nrm ; 
    const NP* vtx ; 
    const NP* nrm ;

    int indices_num ; 
    int indices_offset ; 
    const char* name ; 

    // sframe feels too heavy 
    glm::vec3 mn = {} ; 
    glm::vec3 mx = {} ; 
    glm::vec4 ce = {} ; 

    static SMesh* Load(const char* dir); 
    SMesh(); 
    void import(const NPFold* fold); 

    static float Extent( const glm::vec3& low, const glm::vec3& high ); 
    static glm::vec4 CenterExtent( const glm::vec3& low, const glm::vec3& high ); 
    void find_center_extent(); 

    std::string desc() const ; 

    template<typename T> static void SmoothNormals( 
               std::vector<glm::tvec3<T>>& nrm, 
         const std::vector<glm::tvec3<T>>& vtx, 
         const std::vector<glm::tvec3<int>>& tri );

    static NP* SmoothNormals( const NP* a_vtx, const NP* a_tri ); 

};

inline SMesh* SMesh::Load(const char* dir)
{
    NPFold* fold = NPFold::Load(dir) ; 
    SMesh* mesh = new SMesh ; 
    mesh->import(fold); 
    return mesh ; 
}

inline SMesh::SMesh()
    :   
    tri(nullptr),
    _vtx(nullptr),
    _nrm(nullptr),
    vtx(nullptr),
    nrm(nullptr),
    indices_num(0),
    indices_offset(0),
    name(nullptr),
    ce(0.f,0.f,0.f,0.f)
{
}

inline void SMesh::import(const NPFold* fold)
{
    tri = fold->get("tri");
    _vtx = fold->get("vtx");
    _nrm = SmoothNormals( _vtx, tri ); // smooth in double precision 
    vtx = NP::MakeNarrowIfWide(_vtx);
    nrm = NP::MakeNarrowIfWide(_nrm);

    assert( tri->shape.size() == 2 );
    indices_num = tri->shape[0]*tri->shape[1] ;
    indices_offset = 0 ;

    find_center_extent(); 

    std::string n = desc();
    name = strdup(n.c_str());
}

inline std::string SMesh::desc() const
{
    std::stringstream ss ;
    ss
       << " tri "  << ( tri ? tri->sstr() : "-" )
       << " vtx "  << ( vtx ? vtx->sstr() : "-" )
       << " indices_num " << indices_num
       << " indices_offset " << indices_offset
       ;  
    
    ss << " mn [" ; 
    for(int i=0 ; i < 3 ; i++ ) ss << std::fixed << std::setw(7) << std::setprecision(3) << mn[i] << " " ; 
    ss << "]" ;

    ss << " mx [" ;  
    for(int i=0 ; i < 3 ; i++ ) ss << std::fixed << std::setw(7) << std::setprecision(3) << mx[i] << " " ; 
    ss << "]" ;

    ss << " ce [" ;  
    for(int i=0 ; i < 4 ; i++ ) ss << std::fixed << std::setw(7) << std::setprecision(3) << ce[i] << " " ; 
    ss << "]" ;

    std::string str = ss.str();
    return str ;
}



inline float SMesh::Extent( const glm::vec3& low, const glm::vec3& high ) // static
{ 
    glm::vec3 dim(high.x - low.x, high.y - low.y, high.z - low.z );
    float _extent(0.f) ;
    _extent = std::max( dim.x , _extent );
    _extent = std::max( dim.y , _extent );
    _extent = std::max( dim.z , _extent );
    _extent = _extent / 2.0f ;    
    return _extent ; 
}

inline glm::vec4 SMesh::CenterExtent( const glm::vec3& low, const glm::vec3& high ) // static
{
    glm::vec3 center = (low + high)/2.f ;  
    glm::vec4 _ce ; 
    _ce.x = center.x ; 
    _ce.y = center.y ; 
    _ce.z = center.z ;
    _ce.w = Extent( low, high ); 
    return _ce ; 
}

inline void SMesh::find_center_extent()
{
    int item_stride = 1 ; 
    int item_offset = 0 ; 
    const_cast<NP*>(vtx)->minmax2D_reshaped<3,float>(&mn.x, &mx.x, item_stride, item_offset );  
    ce = CenterExtent( mn, mx ); 
}


/**
SMesh::SmoothNormals
---------------------

* https://computergraphics.stackexchange.com/questions/4031/programmatically-generating-vertex-normals

The smoothing of normals is actually a 
cunning technique described by Inigo Quilezles (of SDF fame)

* https://iquilezles.org/articles/normals/

Essentially are combining non-normalized cross products 
from each face into the vertex normals... so the effect 
is to do a weighted average of the normals from all faces 
adjacent to the vertex with a weighting according to tri area.

**/


template<typename T>
inline void SMesh::SmoothNormals( std::vector<glm::tvec3<T>>& nrm, const std::vector<glm::tvec3<T>>& vtx, const std::vector<glm::tvec3<int>>& tri ) // static
{
    int num_vtx = vtx.size(); 
    int num_tri = tri.size(); 

    typedef glm::tvec3<T>      R3 ; 
    typedef glm::tvec3<int>    I3 ; 

    nrm.resize(num_vtx); 
    for(int i=0 ; i < num_vtx ; i++) nrm[i] = R3{}  ; 

    for(int i=0 ; i < num_tri ; i++)
    {
        const I3& t = tri[i] ; 
        assert( t.x > -1 && t.x < num_vtx ); 
        assert( t.y > -1 && t.y < num_vtx ); 
        assert( t.z > -1 && t.z < num_vtx ); 
        
        const R3& v0 = vtx[t.x] ; 
        const R3& v1 = vtx[t.y] ; 
        const R3& v2 = vtx[t.z] ; 

        R3 n = glm::cross(v1-v0, v2-v0) ;

        nrm[t.x] += n ; 
        nrm[t.y] += n ; 
        nrm[t.z] += n ; 
    }
    for(int i=0 ; i < num_vtx ; i++) nrm[i] = glm::normalize( nrm[i] ); 
}

/**
SMesh::SmoothNormals
---------------------

See decription in the lower level method. 

**/

inline NP* SMesh::SmoothNormals( const NP* a_vtx, const NP* a_tri ) // static
{
    int num_vtx = a_vtx ? a_vtx->shape[0] : 0 ; 
    int num_tri = a_tri ? a_tri->shape[0] : 0 ; 

    typedef glm::tvec3<double> D3 ; 
    typedef glm::tvec3<float>  F3 ; 
    typedef glm::tvec3<int>    I3 ; 

    assert( sizeof(D3) == sizeof(double)*3 ); 
    assert( sizeof(F3) == sizeof(float)*3 ); 
    assert( sizeof(I3) == sizeof(int)*3 ); 

    std::vector<I3> tri(num_tri) ; 
    assert( sizeof(I3)*tri.size() == a_tri->arr_bytes() ); 
    memcpy( tri.data(), a_tri->bytes(), a_tri->arr_bytes() ); 

    NP* a_nrm = nullptr ; 
    if( a_vtx->ebyte == 8 )
    {
        std::vector<D3> vtx(num_vtx) ; 
        std::vector<D3> nrm(num_vtx, {0,0,0}) ;
        assert( sizeof(D3)*vtx.size() == a_vtx->arr_bytes() ); 
        memcpy( vtx.data(), a_vtx->bytes(), a_vtx->arr_bytes() ); 

        SmoothNormals<double>( nrm, vtx, tri );  

        a_nrm = NP::Make<double>( num_vtx, 3 ); 
        memcpy( a_nrm->bytes(), nrm.data(), a_nrm->arr_bytes() );  
    } 
    else if( a_vtx->ebyte == 4 )
    {
        std::vector<F3> vtx(num_vtx) ; 
        std::vector<F3> nrm(num_vtx, {0.f,0.f,0.f}) ;
        assert( sizeof(F3)*vtx.size() == a_vtx->arr_bytes() ); 
        memcpy( vtx.data(), a_vtx->bytes(), a_vtx->arr_bytes() ); 

        SmoothNormals<float>( nrm, vtx, tri );  

        a_nrm = NP::Make<float>( num_vtx, 3 ); 
        memcpy( a_nrm->bytes(), nrm.data(), a_nrm->arr_bytes() );  
    }

    std::cout 
        << " SMesh::SmoothNormals "
        << " a_vtx "  << ( a_vtx ? a_vtx->sstr() : "-" )
        << " a_tri "  << ( a_tri ? a_tri->sstr() : "-" )
        << " a_nrm "  << ( a_nrm ? a_nrm->sstr() : "-" )
        << " num_vtx " << num_vtx
        << " num_tri " << num_tri
        << std::endl
        ;   

    return a_nrm ; 
}




