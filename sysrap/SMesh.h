#pragma once
/**
SMesh.h
========

::

    ~/o/sysrap/tests/SMesh_test.sh


**/

#include <glm/glm.hpp>
#include "NPFold.h"

struct SMesh 
{
    static constexpr const char* NAME = "SMesh" ;  

    const char* loaddir ; 
    const NP* tri ; 
    const NP* _vtx ; 
    const NP* _nrm ; 
    const NP* vtx ; 
    const NP* nrm ;
    const NP* face ; 

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

    template<typename T>
    static std::string Desc2D(const NP* a, const char* label=nullptr) ;

    template<typename T, typename S>
    static std::string Desc2D(const NP* a, const NP* b, const char* label=nullptr);  


    std::string descFace() const ; 
    std::string descTri() const ; 
    std::string descVtx() const ; 
    std::string descTriVtx() const ; 
    std::string descFaceVtx() const ; 

    std::string descVtxNrm() const ; 


    static std::string Desc2D_Ref_2D_int_float(const NP* a, const NP* b,  const char* label); 
  

    std::string desc() const ; 

    template<typename T> static void SmoothNormals( 
               std::vector<glm::tvec3<T>>& nrm, 
         const std::vector<glm::tvec3<T>>& vtx, 
         const std::vector<glm::tvec3<int>>& tri );

    template<typename T> static void FlatNormals( 
               std::vector<glm::tvec3<T>>& nrm, 
         const std::vector<glm::tvec3<T>>& vtx, 
         const std::vector<glm::tvec3<int>>& tri );


    static NP* MakeNormals( const NP* a_vtx, const NP* a_tri, bool smooth ); 

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
    loaddir(nullptr),
    tri(nullptr),
    _vtx(nullptr),
    _nrm(nullptr),
    vtx(nullptr),
    nrm(nullptr),
    face(nullptr),
    indices_num(0),
    indices_offset(0),
    name(nullptr),
    ce(0.f,0.f,0.f,0.f)
{
}

inline void SMesh::import(const NPFold* fold)
{
    loaddir = fold->loaddir ? strdup(fold->loaddir) : nullptr ; 


    tri = fold->get("tri");
    _vtx = fold->get("vtx");

    bool smooth = true ; 
    //bool smooth = false ; 
    _nrm = MakeNormals( _vtx, tri, smooth );  // uses doubles

    vtx = NP::MakeNarrowIfWide(_vtx);
    nrm = NP::MakeNarrowIfWide(_nrm);
    face = fold->get("face"); 

    assert( tri->shape.size() == 2 );
    indices_num = tri->shape[0]*tri->shape[1] ;
    indices_offset = 0 ;

    find_center_extent(); 

    name = loaddir ? loaddir : NAME ; 
}

template<typename T>
inline std::string SMesh::Desc2D(const NP* a, const char* label) 
{
    const T* vv = a->cvalues<T>(); 
    int ni = a->shape[0] ; 
    int nj = a->shape[1] ; 
    std::stringstream ss ; 
    if(label) ss << label << std::endl ; 

    for(int i=0 ; i < ni ; i++)
    {
        ss << std::setw(3) << i << " : " ; 
        for(int j=0 ; j < nj ; j++)
        {
            int idx = i*nj + j ; 
            if( a->uifc == 'i' || a->uifc == 'u' )
            {
                ss << std::setw(3) << vv[idx] << " " ; 
            }
            else
            {
                ss << std::setw(7) << std::fixed << std::setprecision(2) << vv[idx] << " " ; 
            }
        }
        ss << std::endl ; 
    }
    std::string str = ss.str() ; 
    return str ; 
}



template<typename T, typename S>
inline std::string SMesh::Desc2D(const NP* a, const NP* b, const char* label) 
{
    const T* a_vv = a->cvalues<T>(); 
    int a_ni = a->shape[0] ; 
    int a_nj = a->shape[1] ; 

    const S* b_vv = b->cvalues<T>(); 
    int b_ni = b->shape[0] ; 
    int b_nj = b->shape[1] ; 

    assert( a_ni == b_ni ); 
    int ni = a_ni ; 

    std::stringstream ss ; 
    if(label) ss << label << std::endl ; 

    for(int i=0 ; i < ni ; i++)
    {
        ss << std::setw(3) << i << " : " ; 

        for(int j=0 ; j < a_nj ; j++)
        {
            int idx = i*a_nj + j ; 
            if( a->uifc == 'i' || a->uifc == 'u' )
            {
                ss << std::setw(3) << a_vv[idx] << " " ; 
            }
            else
            {
                ss << std::setw(7) << std::fixed << std::setprecision(2) << a_vv[idx] << " " ; 
            }
        }

        for(int j=0 ; j < b_nj ; j++)
        {
            int idx = i*b_nj + j ; 
            if( b->uifc == 'i' || b->uifc == 'u' )
            {
                ss << std::setw(3) << b_vv[idx] << " " ; 
            }
            else
            {
                ss << std::setw(7) << std::fixed << std::setprecision(2) << b_vv[idx] << " " ; 
            }
        }
        ss << std::endl ; 

    }
    std::string str = ss.str() ; 
    return str ; 
}










inline std::string SMesh::descFace() const
{
    return Desc2D<int>(face,"SMesh::descFace") ; 
}
inline std::string SMesh::descTri() const
{
    return Desc2D<int>(tri,"SMesh::descTri") ; 
}
inline std::string SMesh::descVtx() const
{
    return Desc2D<float>(vtx,"SMesh::descVtx") ; 
}

inline std::string SMesh::descTriVtx() const
{
    return Desc2D_Ref_2D_int_float(tri,vtx,"SMesh::descTriVtx") ; 
}
inline std::string SMesh::descFaceVtx() const
{
    return Desc2D_Ref_2D_int_float(face,vtx,"SMesh::descFaceVtx") ; 
}

inline std::string SMesh::descVtxNrm() const
{
    return Desc2D<float,float>(vtx,nrm,"SMesh::descVtxNrm") ; 
}



inline std::string SMesh::Desc2D_Ref_2D_int_float(const NP* a, const NP* b,  const char* label)  // static
{
    const int* a_vv = a->cvalues<int>(); 
    int a_ni = a->shape[0] ; 
    int a_nj = a->shape[1] ;

    const float* b_vv = b->cvalues<float>(); 
    int b_ni = b->shape[0] ; 
    int b_nj = b->shape[1] ;
 
    std::stringstream ss ; 
    if(label) ss << label << std::endl ; 
    for(int i=0 ; i < a_ni ; i++)
    {
        for(int j=0 ; j < a_nj ; j++)
        {
            int a_idx = i*a_nj + j ; 
            int v = a_vv[a_idx] ; 
            assert( v < b_ni );   

            ss << std::setw(3) << v << " : " ; 
            for(int bj=0 ; bj < b_nj ; bj++)
            {
                int b_idx =  v*b_nj + bj ; 
                ss << std::setw(7) << std::fixed << std::setprecision(2) << b_vv[b_idx] << " " ; 
            }
            ss << std::endl ; 
            
        }
        ss << std::endl ; 
    }
    std::string str = ss.str() ; 
    return str ; 
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


Triangle CW/CCW winding order ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

           C  
           v2
           /. 
          /  .
         /    . 
       v0 ---- v1
       A        B

glm::cross(v1-v0, v2-v0) is out the page (
glm::cross(AB, AC) 

http://www.fromatogra.com/math/6-triangles

    To know whether a given triangle A, B, C has a clockwise (cw) or counter
    clockwise (ccw) winding order, you can look at the sign of the cross product of
    AB and AC. 

    HUH: cross product is a vector, need to pick some direction... to dot product
    the normal with and compare signs of that 

    How cw and ccw are mapped to a positive or negative cross product
    depends on how your cross product is defined and whether your y axis points up
    or down. Remember, the cross product of two vectors is the scaled sine of the
    angle between the vectors. When you use a mathematical coordinate system where
    y is up, ccw is positive and cw is negative. However in most graphical systems
    where y is down, cw is positive and ccw is negative.

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
SMesh::FlatNormals
-------------------

Note the overwriting of the normal, last face wins

**/

template<typename T>
inline void SMesh::FlatNormals( std::vector<glm::tvec3<T>>& nrm, const std::vector<glm::tvec3<T>>& vtx, const std::vector<glm::tvec3<int>>& tri ) // static
{
    int num_vtx = vtx.size(); 
    int num_tri = tri.size(); 

    typedef glm::tvec3<T>      R3 ; 
    typedef glm::tvec3<int>    I3 ; 

    nrm.resize(num_vtx); 

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

        nrm[t.x] = n ; 
        nrm[t.y] = n ; 
        nrm[t.z] = n ; 
    }
}



/**
SMesh::MakeNormals
---------------------

**/

inline NP* SMesh::MakeNormals( const NP* a_vtx, const NP* a_tri, bool smooth ) // static
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

        if(smooth)
        {
            SmoothNormals<double>( nrm, vtx, tri );  
        }
        else
        {
            FlatNormals<double>( nrm, vtx, tri );  
        }

        a_nrm = NP::Make<double>( num_vtx, 3 ); 
        memcpy( a_nrm->bytes(), nrm.data(), a_nrm->arr_bytes() );  
    } 
    else if( a_vtx->ebyte == 4 )
    {
        std::vector<F3> vtx(num_vtx) ; 
        std::vector<F3> nrm(num_vtx, {0.f,0.f,0.f}) ;
        assert( sizeof(F3)*vtx.size() == a_vtx->arr_bytes() ); 
        memcpy( vtx.data(), a_vtx->bytes(), a_vtx->arr_bytes() ); 

        if(smooth)
        {
            SmoothNormals<float>( nrm, vtx, tri );  
        }
        else
        {
            FlatNormals<float>( nrm, vtx, tri );  
        }

        a_nrm = NP::Make<float>( num_vtx, 3 ); 
        memcpy( a_nrm->bytes(), nrm.data(), a_nrm->arr_bytes() );  
    }

    std::cout 
        << " SMesh::MakeNormals "
        << " a_vtx "  << ( a_vtx ? a_vtx->sstr() : "-" )
        << " a_tri "  << ( a_tri ? a_tri->sstr() : "-" )
        << " a_nrm "  << ( a_nrm ? a_nrm->sstr() : "-" )
        << " num_vtx " << num_vtx
        << " num_tri " << num_tri
        << std::endl
        ;   

    return a_nrm ; 
}




