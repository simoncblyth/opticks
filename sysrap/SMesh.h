#pragma once
/**
SMesh.h
========

::

    ~/o/sysrap/tests/SMesh_test.sh


**/

#include <ostream>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

#include "stra.h"
#include "NPFold.h"

struct SMesh 
{
    static SMesh* Concatenate(std::vector<const SMesh*>& submesh, int ridx ); 

    static constexpr const char* NAME = "SMesh" ;  
    static constexpr const char* VTX_SPEC = "3,GL_FLOAT,GL_FALSE,12,0,false" ;  // 12=3*sizeof(float)
    static constexpr const char* NRM_SPEC = "3,GL_FLOAT,GL_FALSE,12,0,false" ;  
    static constexpr const bool  NRM_SMOOTH = true ; 
    // 3:vec3, 12:byte_stride 0:byte_offet

    const char* name    ;            // metadata : loaddir or manually set name
    glm::tmat4x4<double> tr0 = {} ;  // informational only, as gets applied by init  

    const NP* tri ; 
    const NP* vtx ; 
    const NP* nrm ;

    glm::tvec3<float> mn = {} ; 
    glm::tvec3<float> mx = {} ; 
    glm::tvec4<float> ce = {} ; 

    static SMesh* Load(const char* dir,      const glm::tmat4x4<double>* tr=nullptr ); 
    static SMesh* Import(const NPFold* fold, const glm::tmat4x4<double>* tr=nullptr );
    void import(         const NPFold* fold, const glm::tmat4x4<double>* tr=nullptr ); 

    NPFold* serialize() const ; 
    void save(const char* dir) const ; 

    SMesh(); 

    void set_tri( const NP* _tri ); 
    int indices_num() const ; 
    int indices_offset() const ; 
    static const char* FormName(int ridx); 
    void set_name(int ridx); 

    void set_vtx( const NP* wvtx, const glm::tmat4x4<double>* tr, std::ostream* out  ); 
    void set_vtx_range(); 

    template<typename T>
    static T Extent( const glm::tvec3<T>& low, const glm::tvec3<T>& high ); 

    template<typename T>
    static glm::tvec4<T> CenterExtent( const glm::tvec3<T>& low, const glm::tvec3<T>& high ); 

    template<typename T>
    static void FindCenterExtent(const NP* vtx, glm::tvec3<T>& mn, glm::tvec3<T>& mx, glm::tvec4<T>& ce );

    template<typename T>
    static std::string Desc2D(const NP* a, const char* label=nullptr) ;

    template<typename T, typename S>
    static std::string Desc2D(const NP* a, const NP* b, const char* label=nullptr);  

    std::string descTransform() const ; 

    std::string brief() const ; 
    std::string desc() const ; 
    std::string descTri() const ; 
    std::string descVtx() const ; 
    std::string descTriVtx() const ; 
    std::string descVtxNrm() const ; 
    std::string descShape() const ; 
    std::string descRange() const ; 


    static std::string Desc2D_Ref_2D_int_float(const NP* a, const NP* b,  const char* label); 
  


    template<typename T> static void SmoothNormals( 
               std::vector<glm::tvec3<T>>& nrm, 
         const std::vector<glm::tvec3<T>>& vtx, 
         const std::vector<glm::tvec3<int>>& tri,
         std::ostream* out );

    template<typename T> static void FlatNormals( 
               std::vector<glm::tvec3<T>>& nrm, 
         const std::vector<glm::tvec3<T>>& vtx, 
         const std::vector<glm::tvec3<int>>& tri, 
         std::ostream* out );


    static NP* MakeNormals( const NP* a_vtx, const NP* a_tri, bool smooth, std::ostream* out ); 
};

inline SMesh* SMesh::Concatenate(std::vector<const SMesh*>& submesh, int ridx )
{
    SMesh* com = new SMesh ; 
    com->set_name(ridx); 

    std::vector<const NP*> subtri ; 
    std::vector<const NP*> subvtx ; 
    std::vector<const NP*> subnrm ; 

    int tot_vtx = 0 ; 
    for(int i=0 ; i < int(submesh.size()) ; i++)
    {
        const NP* _tri = submesh[i]->tri ;  
        const NP* _vtx = submesh[i]->vtx ;  
        const NP* _nrm = submesh[i]->nrm ;  
         
        int num_vtx = _vtx->num_items() ; 
        int num_nrm = _nrm->num_items() ; 
        assert( num_vtx == num_nrm ); 

        subtri.push_back(NP::Incremented(_tri, tot_vtx)) ; 
        subvtx.push_back(_vtx) ; 
        subnrm.push_back(_nrm) ; 

        tot_vtx += num_vtx ; 
    } 
    com->tri = NP::Concatenate(subtri) ; 
    com->vtx = NP::Concatenate(subvtx) ; 
    com->nrm = NP::Concatenate(subnrm) ; 
    com->set_vtx_range(); 

    return com ; 
}

inline SMesh* SMesh::Load(const char* dir, const glm::tmat4x4<double>* tr)
{
    NPFold* fold = NPFold::Load(dir) ; 
    return Import(fold, tr); 
}
inline SMesh* SMesh::Import(const NPFold* fold, const glm::tmat4x4<double>* tr)
{
    SMesh* mesh = new SMesh ; 
    mesh->import(fold, tr); 
    return mesh ; 
}

inline void SMesh::import(const NPFold* fold, const glm::tmat4x4<double>* tr )
{
    name = fold->loaddir ? strdup(fold->loaddir) : nullptr ; 

    const NP* triangles = fold->get("tri");
    const NP* vertices = fold->get("vtx")->copy() ;

    bool dump = false ; 
    std::stringstream ss ; 
    std::ostream* out = dump ? &ss : nullptr ; 

    if(out) *out << "[SMesh::import" << std::endl ; 

    set_tri( triangles ); 
    set_vtx( vertices, tr, out ); 

    if(out) *out << "]SMesh::import" << std::endl ; 
    if(dump) std::cout << ss.str() ; 
}

NPFold* SMesh::serialize() const 
{
    NPFold* fold = new NPFold ; 
    fold->add("tri", tri); 
    fold->add("vtx", vtx); 
    fold->add("nrm", nrm); 
    return fold ; 
}
void SMesh::save(const char* dir) const 
{
    NPFold* fold = serialize(); 
    fold->save(dir); 
}   


inline SMesh::SMesh()
    :   
    name(nullptr),
    tri(nullptr),
    vtx(nullptr),
    nrm(nullptr),
    tr0(1.),
    ce(0.)
{
}

/**
SMesh::set_tri
---------------------

Removed face which was passenger only from U4Mesh for debugging 

**/

inline void SMesh::set_tri( const NP* _tri )
{
    tri = _tri ; 
}

inline int SMesh::indices_num() const 
{
    if(tri == nullptr) return 0 ; 
    assert( tri->shape.size() == 2 );
    return tri->num_values() ; 
}
inline int SMesh::indices_offset() const 
{
    return 0 ; 
}

inline const char* SMesh::FormName(int ridx) // static
{
    std::stringstream ss ; 
    ss << ridx ; 
    std::string str = ss.str(); 
    return strdup( str.c_str() ); 
}
inline void SMesh::set_name( int ridx )
{
    name = FormName(ridx);    
}

inline void SMesh::set_vtx( const NP* _wvtx, const glm::tmat4x4<double>* tr,  std::ostream* out  )
{
    assert( tri ); 
    assert( _wvtx->shape.size() == 2 );
    assert( _wvtx->shape[0] > 2  );   // need at least 3 vtx to make a face
    assert( _wvtx->shape[1] == 3 );

    if(tr) memcpy( glm::value_ptr(tr0), glm::value_ptr(*tr), sizeof(tr0) );  // informational only 
    NP* wvtx = stra<double>::MakeTransformedArray( _wvtx, tr );     
    NP* wnrm = MakeNormals( wvtx, tri, NRM_SMOOTH, out );  // uses doubles

    vtx = NP::MakeNarrowIfWide(wvtx);
    nrm = NP::MakeNarrowIfWide(wnrm);
    set_vtx_range(); 
}

inline void SMesh::set_vtx_range()
{
    FindCenterExtent(vtx, mn, mx, ce); 
}

template<typename T>
inline std::string SMesh::Desc2D(const NP* a, const char* label) 
{
    std::stringstream ss ; 
    if(label) ss << label << std::endl ; 
    if(a == nullptr)
    {
        ss << " (null) " << std::endl ; 
    }
    else
    {
        const T* vv = a->cvalues<T>(); 
        int ni = a->shape[0] ; 
        int nj = a->shape[1] ; 

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
    }
    std::string str = ss.str() ; 
    return str ; 
}



template<typename T, typename S>
inline std::string SMesh::Desc2D(const NP* a, const NP* b, const char* label) 
{

    std::stringstream ss ; 
    if(label) ss << label << std::endl ; 

    if( a == nullptr || b == nullptr )
    {
        ss << " missing a or b " << std::endl ; 
    }
    else
    {

        const T* a_vv = a->cvalues<T>(); 
        int a_ni = a->shape[0] ; 
        int a_nj = a->shape[1] ; 

        const S* b_vv = b->cvalues<T>(); 
        int b_ni = b->shape[0] ; 
        int b_nj = b->shape[1] ; 

        assert( a_ni == b_ni ); 
        int ni = a_ni ; 

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
    }
    std::string str = ss.str() ; 
    return str ; 
}

inline std::string SMesh::descTransform() const 
{
    return stra<double>::Desc(tr0); 
}

inline std::string SMesh::brief() const
{
    std::stringstream ss ; 
    ss << "SMesh::brief " ; 
    //ss << ( loaddir ? loaddir : "-" ) ; 
    ss << " tri " << tri->sstr() ;  
    ss << " vtx " << vtx->sstr() ;  
    ss << " nrm " << nrm->sstr() ;  

    std::string str = ss.str() ; 
    return str ; 
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
inline std::string SMesh::descVtxNrm() const
{
    return Desc2D<float,float>(vtx,nrm,"SMesh::descVtxNrm") ; 
}



inline std::string SMesh::Desc2D_Ref_2D_int_float(const NP* a, const NP* b,  const char* label)  // static
{
    std::stringstream ss ; 
    if(label) ss << label << std::endl ; 
    if( a == nullptr || b == nullptr )
    {
        ss << " a or b missing " << std::endl ;  
    }
    else
    {
        const int* a_vv = a->cvalues<int>(); 
        int a_ni = a->shape[0] ; 
        int a_nj = a->shape[1] ;

        const float* b_vv = b->cvalues<float>(); 
        int b_ni = b->shape[0] ; 
        int b_nj = b->shape[1] ;
     
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
    }
    std::string str = ss.str() ; 
    return str ; 
}

inline std::string SMesh::desc() const
{
    std::stringstream ss ;
    ss << "SMesh::desc" 
       << std::endl 
       << descShape()
       << std::endl 
       << descRange() 
       << std::endl 
       << descTri() 
       << std::endl 
       << descVtx() 
       << std::endl 
       << descTriVtx() 
       << std::endl 
       << descVtxNrm() 
       << std::endl 
       ;

    std::string str = ss.str();
    return str ;
}


inline std::string SMesh::descShape() const
{
    std::stringstream ss ;
    ss
       << "SMesh::descShape" << std::endl 
       << " tri "  << ( tri ? tri->sstr() : "-" )
       << " vtx "  << ( vtx ? vtx->sstr() : "-" )
       << " indices_num " << indices_num()
       << " indices_offset " << indices_offset()
       ; 
    std::string str = ss.str();
    return str ;
}

inline std::string SMesh::descRange() const
{
    std::stringstream ss ;
    ss << "SMesh::descRange" << std::endl ;  
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



template<typename T>
inline T SMesh::Extent( const glm::tvec3<T>& low, const glm::tvec3<T>& high ) // static
{ 
    glm::tvec3<T> dim(high.x - low.x, high.y - low.y, high.z - low.z );
    T _extent(0) ;
    _extent = std::max( dim.x , _extent );
    _extent = std::max( dim.y , _extent );
    _extent = std::max( dim.z , _extent );
    _extent = _extent / T(2) ;    
    return _extent ; 
}


template<typename T>
inline glm::tvec4<T> SMesh::CenterExtent( const glm::tvec3<T>& low, const glm::tvec3<T>& high ) // static
{
    glm::tvec3<T> center = (low + high)/T(2) ;  
    glm::tvec4<T> _ce ; 
    _ce.x = center.x ; 
    _ce.y = center.y ; 
    _ce.z = center.z ;
    _ce.w = Extent<T>( low, high ); 
    return _ce ; 
}


template<typename T>
inline void SMesh::FindCenterExtent(const NP* vtx, glm::tvec3<T>& mn, glm::tvec3<T>& mx, glm::tvec4<T>& ce )
{
    assert( vtx->ebyte == sizeof(T) ); 
    int item_stride = 1 ; 
    int item_offset = 0 ; 
    const_cast<NP*>(vtx)->minmax2D_reshaped<3,T>(&mn.x, &mx.x, item_stride, item_offset );  
    ce = CenterExtent<T>( mn, mx ); 
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
inline void SMesh::SmoothNormals( 
    std::vector<glm::tvec3<T>>& nrm, 
    const std::vector<glm::tvec3<T>>& vtx, 
    const std::vector<glm::tvec3<int>>& tri, 
    std::ostream* out  ) // static
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
        if(out) *out << " tri[" << i << "] " << glm::to_string(t) << std::endl ;  

        bool x_expected =  t.x > -1 && t.x < num_vtx ;
        bool y_expected =  t.y > -1 && t.y < num_vtx ;
        bool z_expected =  t.z > -1 && t.z < num_vtx ;

        bool expected = x_expected && y_expected && z_expected ; 

        if(!expected ) std::cout 
            << "SMesh::SmoothNormals"
            << " FATAL NOT expected " 
            << " i [" << i << "] "
            << " t [" << glm::to_string(t) << "]"
            << " num_vtx " << num_vtx 
            << " num_tri " << num_tri 
            << std::endl 
            ; 

        assert( x_expected ); 
        assert( y_expected ); 
        assert( z_expected ); 
        
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
inline void SMesh::FlatNormals( 
    std::vector<glm::tvec3<T>>& nrm, 
    const std::vector<glm::tvec3<T>>& vtx, 
    const std::vector<glm::tvec3<int>>& tri,
    std::ostream* out ) // static
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

inline NP* SMesh::MakeNormals( const NP* a_vtx, const NP* a_tri, bool smooth, std::ostream* out ) // static
{
    int num_vtx = a_vtx ? a_vtx->shape[0] : 0 ; 
    int num_tri = a_tri ? a_tri->shape[0] : 0 ; 

    if(out) *out 
        << "[ SMesh::MakeNormals "
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
            SmoothNormals<double>( nrm, vtx, tri, out );  
        }
        else
        {
            FlatNormals<double>( nrm, vtx, tri, out );  
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
            SmoothNormals<float>( nrm, vtx, tri, out );  
        }
        else
        {
            FlatNormals<float>( nrm, vtx, tri, out );  
        }

        a_nrm = NP::Make<float>( num_vtx, 3 ); 
        memcpy( a_nrm->bytes(), nrm.data(), a_nrm->arr_bytes() );  
    }

    if(out) *out 
        << "] SMesh::MakeNormals "
        << " a_vtx "  << ( a_vtx ? a_vtx->sstr() : "-" )
        << " a_tri "  << ( a_tri ? a_tri->sstr() : "-" )
        << " a_nrm "  << ( a_nrm ? a_nrm->sstr() : "-" )
        << " num_vtx " << num_vtx
        << " num_tri " << num_tri
        << std::endl
        ;   

    return a_nrm ; 
}


