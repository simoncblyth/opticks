#pragma once
/**
stra.h
========

Following some investigations of Geant4 transform handling
and noting that inverses are being done their at source
have concluded that dealing with transforms together with
their inverses is not worth the overhead and complication.
Of course inverting should be minimized.

Hence are bringing over functionality from stran.h as its needed
in new code.

**/

#include <array>

#include "glm/glm.hpp"
#include <glm/gtc/constants.hpp>
#include "glm/gtx/string_cast.hpp"
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/component_wise.hpp>

#include "NP.hh"


template<typename T>
struct stra
{
    static constexpr const bool VERBOSE = false ;

    static std::string Desc(const glm::tmat4x4<T>& t, const glm::tmat4x4<T>& v, const char* tl, const char* vl );
    static std::string Desc(
        const glm::tmat4x4<double>& a,
        const glm::tmat4x4<double>& b,
        const glm::tmat4x4<double>& c,
        const char* al,
        const char* bl,
        const char* cl);

    static std::string Desc(const T* aa, const T* bb, const T* cc, const char* al, const char* bl, const char* cl) ;

    static std::string Desc(const glm::tmat4x4<T>& tr);
    static std::string Desc(const glm::tvec4<T>& t);
    static std::string Desc(const glm::tvec3<T>& t);
    static std::string Desc(const T* tt, int num);
    static std::string Desc(const T& t);

    static std::string DescItems(const T* tt, int num_elem, int num_item, int edge_items=10 );

    static std::string Desc(const T* tt, int ni, int nj, int item_stride=0) ;


    static std::string Array(const glm::tmat4x4<T>& tr);
    static std::string Array(const T* tt, int num);


    static glm::tmat4x4<T> FromData(const T* data );

    static glm::tmat4x4<T> Translate( const T tx, const T ty, const T tz, const T sc, bool flip=false );
    static glm::tmat4x4<T> Translate( const glm::tvec3<T>& tlate, bool flip=false );

    static glm::tmat4x4<T> Rotate( const T ax, const T ay, const T az, const T angle_deg, bool flip=false) ;

    static glm::tvec3<T>   LeastParallelAxis(const glm::tvec3<T>& a );
    static glm::tmat4x4<T> RotateA2B_nonparallel( const glm::tvec3<T>& a, const glm::tvec3<T>& b, bool flip );
    static glm::tmat4x4<T> RotateA2B_parallel(    const glm::tvec3<T>& a, const glm::tvec3<T>& b, bool flip );
    static glm::tmat4x4<T> RotateA2B(             const glm::tvec3<T>& a, const glm::tvec3<T>& b, bool flip );

    static glm::tmat4x4<T> Place(                 const glm::tvec3<T>& a,  const glm::tvec3<T>& b, const glm::tvec3<T>& c, bool flip);
    static glm::tmat4x4<T> Model2World(           const glm::tvec3<T>& ax, const glm::tvec3<T>& up, const glm::tvec3<T>& translation );

    static glm::tmat4x4<T> Dupe(                  const glm::tvec3<T>& a, const glm::tvec3<T>& b, const glm::tvec3<T>& c, bool flip);

    static T Maxdiff_from_Identity(const glm::tmat4x4<T>& m);
    static bool IsIdentity(const glm::tmat4x4<T>& m, T epsilon=1e-6);
    static bool IsIdentity(const glm::tmat4x4<T>& t, const glm::tmat4x4<T>& v, T epsilon=1e-6);

    static void Rows(glm::tvec4<T>& q0,
                     glm::tvec4<T>& q1,
                     glm::tvec4<T>& q2,
                     glm::tvec4<T>& q3,
                     const glm::tmat4x4<T>& t ) ;

    static void Min(glm::tvec4<T>& q, const glm::tvec4<T>& a, const glm::tvec4<T>& b);
    static void Max(glm::tvec4<T>& q, const glm::tvec4<T>& a, const glm::tvec4<T>& b);

    static void Transform_AABB( T* aabb1, const T* aabb0, const glm::tmat4x4<T>& t );
    static void Transform_AABB_Inplace(         T* aabb,  const glm::tmat4x4<T>& t );

    static void Transform_Vec( glm::tvec4<T>& pos, const glm::tvec4<T>&  pos0, const glm::tmat4x4<T>& t );
    static void Transform_Strided( T* _pos, const T* _pos0,  int ni, int nj, int item_stride, const glm::tmat4x4<T>& t, T w = 1.  );
    static void Transform_Strided_Inplace(        T* _pos,   int ni, int nj, int item_stride, const glm::tmat4x4<T>& t, T w = 1.  );
    static void Transform_Data(    T* _pos, const T* _pos0,  const glm::tmat4x4<T>* t, T w = 1.  );
    static void Transform_Array( NP* d , const NP* s, const glm::tmat4x4<T>* t, T w=1.  , int stride=0, int offset=0 );

    static NP* MakeTransformedArray(const NP* a, const glm::tmat4x4<T>* t, T w=1.  , int stride=0, int offset=0 );


    static void Copy_Columns_3x4( T* dst, const T* src );
    static void Copy_Columns_3x4( T* dst, const glm::tmat4x4<T>& tr );
    static void Copy_Columns_3x4( glm::tmat4x4<T>& dst, const glm::tmat4x4<T>& src );


};


template<typename T>
std::string stra<T>::Desc(const glm::tmat4x4<T>& t, const glm::tmat4x4<T>& v, const char* tl, const char* vl )  // static
{
    glm::tmat4x4<double> tv = t*v ;

    std::stringstream ss ;
    ss << tl << "*" << vl ;
    std::string tv_l = ss.str();

    return Desc( t, v, tv,  tl, vl, tv_l.c_str() );
}

template<typename T>
std::string stra<T>::Desc(
    const glm::tmat4x4<double>& a,
    const glm::tmat4x4<double>& b,
    const glm::tmat4x4<double>& c,
    const char* al,
    const char* bl,
    const char* cl)
{
    return Desc(glm::value_ptr(a),glm::value_ptr(b),glm::value_ptr(c),al,bl,cl) ;
}

template<typename T>
std::string stra<T>::Desc(const T* aa, const T* bb, const T* cc, const char* al, const char* bl, const char* cl)
{
    std::stringstream ss ;
    ss << "\n" ;

    if(al) ss << " " << std::setw(54) << std::left << al ;
    if(bl) ss << " " << std::setw(54) << std::left << bl ;
    if(cl) ss << " " << std::setw(54) << std::left << cl ;

    for(int i=0 ; i < 4 ; i++)
    {
        ss << "\n" ;
        if(aa) for(int j=0 ; j < 4 ; j++) ss << " " << std::fixed << std::setw(10) << std::setprecision(4) << aa[i*4+j] ;
        ss << " " << std::setw(10) << " " ;
        if(bb) for(int j=0 ; j < 4 ; j++) ss << " " << std::fixed << std::setw(10) << std::setprecision(4) << bb[i*4+j] ;
        ss << " " << std::setw(10) << " " ;
        if(cc) for(int j=0 ; j < 4 ; j++) ss << " " << std::fixed << std::setw(10) << std::setprecision(4) << cc[i*4+j] ;
    }
    ss << "\n" ;
    std::string s = ss.str();
    return s ;
}


template<typename T>
std::string stra<T>::Desc(const glm::tmat4x4<T>& tr)
{
    const T* tt = glm::value_ptr(tr);
    return Desc(tt, 16 );
}
template<typename T>
std::string stra<T>::Desc(const glm::tvec4<T>& t)
{
    const T* tt = glm::value_ptr(t);
    return Desc(tt, 4 );
}
template<typename T>
std::string stra<T>::Desc(const glm::tvec3<T>& t)
{
    const T* tt = glm::value_ptr(t);
    return Desc(tt, 3 );
}


template<typename T>
std::string stra<T>::Desc(const T* tt, int num)
{
    std::stringstream ss ;
    for(int i=0 ; i < num ; i++)
        ss
            << ( i % 4 == 0 && num > 4 ? ".\n" : "" )
            << " " << Desc(tt[i])
            << ( i == num-1 && num > 4 ? ".\n" : "" )
            ;

    std::string str = ss.str();
    return str ;
}

template<typename T>
std::string stra<T>::Desc(const T& t)
{
    std::stringstream ss ;
    ss << std::fixed << std::setw(10) << std::setprecision(4) << t ;
    std::string str = ss.str();
    return str ;
}


template<typename T>
std::string stra<T>::DescItems(const T* tt, int num_elem, int num_item, int edge_items )
{
    std::stringstream ss ;
    for(int i=0 ; i < num_item ; i++ )
    {
        if( i < edge_items || i > (num_item - edge_items) ) ss
            << " i " << i
            << std::endl
            << Desc( tt + i*num_elem, num_elem )
            << std::endl
            ;
    }
    std::string str = ss.str();
    return str ;
}








template<typename T>
std::string stra<T>::Desc(const T* tt, int ni, int nj, int item_stride)
{
    int stride = item_stride == 0 ? nj : item_stride ;

    std::stringstream ss ;
    for(int i=0 ; i < ni ; i++)
    {
        for(int j=0 ; j < nj ; j++)
        {
            int idx = i*stride + j ;
            ss
                << " " << std::fixed << std::setw(10) << std::setprecision(4) << tt[idx]
                ;
        }
        if( i < ni - 1) ss << std::endl ;
    }
    std::string s = ss.str();
    return s ;
}







template<typename T>
std::string stra<T>::Array(const glm::tmat4x4<T>& tr)
{
    const T* tt = glm::value_ptr(tr);
    return Array(tt, 16 );
}

template<typename T>
std::string stra<T>::Array(const T* tt, int num)
{
    int w = 7 ;
    int p = 3 ;

    std::stringstream ss ;
    ss << "np.array(" ;
    if(num == 16 ) ss << "[[" ;

    for(int i=0 ; i < num ; i++) ss
        << ( i % 4 == 0 && num > 4 && i > 0 ? "],[" : ( i > 0 ? "," : "" ) )
        << std::fixed << std::setw(w) << std::setprecision(p) << tt[i]
        ;

    if(num == 16 ) ss << "]]" ;
    ss << ",dtype=np.float64)" ;
    std::string s = ss.str();

    bool squeeze = true ;
    if( squeeze )
    {
        const char* p = s.c_str();
        std::stringstream zz ;
        for(int i=0 ; i < int(strlen(p)) ; i++) if(p[i] != ' ') zz << p[i] ;
        s = zz.str();
    }
    return s ;
}







template<typename T>
glm::tmat4x4<T> stra<T>::FromData(const T* data)  // static
{
    glm::tmat4x4<T> tran(1.);
    memcpy( glm::value_ptr(tran), data, 16*sizeof(T) );
    return tran ;
}





template<typename T>
inline glm::tmat4x4<T> stra<T>::Translate(  const T tx, const T ty, const T tz, const T sc, bool flip )
{
    glm::tvec3<T> tlate(tx*sc,ty*sc,tz*sc);
    return Translate(tlate, flip) ;
}

template<typename T>
inline glm::tmat4x4<T> stra<T>::Translate( const glm::tvec3<T>& tlate, bool flip )
{
    glm::tmat4x4<T> tr = glm::translate(glm::tmat4x4<T>(1.), tlate ) ;
    return flip ? glm::transpose(tr) : tr ;
}

template<typename T>
inline glm::tmat4x4<T> stra<T>::Rotate( const T ax, const T ay, const T az, const T angle_deg, bool flip)
{
    T angle_rad = glm::pi<T>()*angle_deg/T(180.) ;
    glm::tvec3<T> axis(ax,ay,az);
    glm::tmat4x4<T> tr = glm::rotate(glm::tmat4x4<T>(1.), ( flip ? -1. : 1. )* angle_rad, axis ) ;
    return tr ;
}





template<typename T>
inline glm::tvec3<T> stra<T>::LeastParallelAxis( const glm::tvec3<T>& a )
{
    glm::tvec3<T> aa( glm::abs(a) );
    glm::tvec3<T> lpa(0.);

    if( aa.x <= aa.y && aa.x <= aa.z )
    {
        lpa.x = 1.f ;
    }
    else if( aa.y <= aa.x && aa.y <= aa.z )
    {
        lpa.y = 1.f ;
    }
    else
    {
        lpa.z = 1.f ;
    }

    if(VERBOSE) std::cout
        << "stra::LeastParallelAxis"
        << " a " <<  glm::to_string(a)
        << " aa " <<  glm::to_string(aa)
        << " lpa " << glm::to_string(lpa)
        << std::endl
        ;

    return lpa ;
}



/**
stra::RotateA2B
----------------------

See ana/make_rotation_matrix.py

* http://cs.brown.edu/research/pubs/pdfs/1999/Moller-1999-EBA.pdf
  "Efficiently Building a Matrix to Rotate One Vector To Another"
  Tomas Moller and John F Hughes

* ~/opticks_refs/Build_Rotation_Matrix_vec2vec_Moller-1999-EBA.pdf

Found this paper via thread:

* https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

Another very long discussion of rotation matrices by Rebecca Brannon:

* https://my.mech.utah.edu/~brannon/public/rotation.pdf
* ~/opticks_refs/utah_brannon_opus_rotation.pdf

TODO: compare with Quaternions

**/

template<typename T>
inline glm::tmat4x4<T> stra<T>::RotateA2B_nonparallel(const glm::tvec3<T>& a, const glm::tvec3<T>& b, bool flip )
{
    T one(1.);
    T zero(0.);

    T c = glm::dot(a,b);
    T h = (one - c)/(one - c*c);

    glm::tvec3<T> v = glm::cross(a, b) ;
    T vx = v.x ;
    T vy = v.y ;
    T vz = v.z ;

    if(VERBOSE) std::cout
        << "stra::RotateA2B_nonparallel"
        << " a " << glm::to_string(a)
        << " b " << glm::to_string(b)
        << " c " << std::fixed << std::setw(10) << std::setprecision(5) << c
        << " h " << std::fixed << std::setw(10) << std::setprecision(5) << h
        << " vx " << std::fixed << std::setw(10) << std::setprecision(5) << vx
        << " vy " << std::fixed << std::setw(10) << std::setprecision(5) << vy
        << " vz " << std::fixed << std::setw(10) << std::setprecision(5) << vz
        << std::endl
        ;


    T hxx = h*vx*vx ;
    T hyy = h*vy*vy ;
    T hzz = h*vz*vz ;

    T hxy = h*vx*vy ;
    T hxz = h*vx*vz ;
    T hyz = h*vy*vz ;

    T f = flip ? -1. : 1. ;   // flip:true is equivalent to transposing the matrix

    std::array<T, 16> vals = {{
          c + hxx    , hxy - f*vz   ,  hxz + f*vy  , zero,
          hxy + f*vz , c + hyy      ,  hyz - f*vx  , zero,
          hxz - f*vy , hyz + f*vx   ,  c + hzz     , zero,
          zero       , zero         ,  zero        , one
    }};

    return glm::make_mat4x4<T>( vals.data() );
}


template<typename T>
inline glm::tmat4x4<T> stra<T>::RotateA2B_parallel(const glm::tvec3<T>& a, const glm::tvec3<T>& b, bool flip )
{
    glm::tvec3<T> x = LeastParallelAxis(a);
    glm::tvec3<T> u = x - a ;
    glm::tvec3<T> v = x - b  ;

    if(VERBOSE) std::cout
        << "Trab::RotateA2B_parallel"
        << std::endl
        << " x         " << glm::to_string(x) << std::endl
        << " u = x - a " << glm::to_string(u) << std::endl
        << " v = x - b " << glm::to_string(v) << std::endl
        ;


    T uu = glm::dot(u, u);
    T vv = glm::dot(v, v);
    T uv = glm::dot(u, v);

    std::array<T, 16> vals ;
    vals.fill(0.) ;

    for(int i=0 ; i < 3 ; i++) for(int j=0 ; j < 3 ; j++)
    {
        int idx = flip == false ? i*4+j : j*4 + i ;
        vals[idx] = ( i == j ? 1. : 0.)  - 2.*u[i]*u[j]/uu -2.*v[i]*v[j]/vv + 4.*uv*v[i]*u[j]/(uu*vv) ;

    }
    vals[15] = 1. ;

    return glm::make_mat4x4<T>( vals.data() );
}


/**
stra::RotateA2B
------------------

Form a rotation transform that rotates vector a to vector b with proper handling
of parallel or anti-parallel edge cases when the absolute dot product of a and b
is close to 1.

**/

template<typename T>
inline glm::tmat4x4<T> stra<T>::RotateA2B(const glm::tvec3<T>& a, const glm::tvec3<T>& b, bool flip)
{
    T c = glm::dot(a,b);
    return std::abs(c) < 0.99 ? RotateA2B_nonparallel(a,b,flip) : RotateA2B_parallel(a,b,flip) ;
}



/**
stra::Place
-------------

Create a transform combining rotation from a to b and translation of c
The flip argument controls transposes being applied to the rotation
matrix and the order of multiplication of the rotation and the translation.

HMM: not sure regards the "tla*rot" vs "rot*tla"  order flip :
it depends on whether will using the transform to right or left multiply.

BUT what is certain is that the placement transform needs to rotate first and then translate
The complexity comes from different packages needing different layouts,
so generally have to experiment to get things to work.
**/

template<typename T>
inline glm::tmat4x4<T> stra<T>::Place(const glm::tvec3<T>& a, const glm::tvec3<T>& b, const glm::tvec3<T>& c, bool flip )
{
    glm::tmat4x4<T> rot = RotateA2B(a,b,flip) ;
    glm::tmat4x4<T> tla = Translate(c) ;              // this has the translation in last row
    glm::tmat4x4<T> tra = flip == true ? tla * rot : rot * tla ;

    return tra ;
}


template<typename T>
inline glm::tmat4x4<T> stra<T>::Model2World(const glm::tvec3<T>& ax, const glm::tvec3<T>& up, const glm::tvec3<T>& translation )
{
    // 1. Calculate the 'Right' vector (X-axis)
    // Right = Up x Forward (ax)
    glm::tvec3<T> right = glm::normalize(glm::cross(up, ax));

    // 2. Re-calculate Up to ensure perfect perpendicularity
    // Up = Forward (ax) x Right
    glm::tvec3<T> up_rectified = glm::cross(ax, right);

    // 3. Create the 4x4 Identity Matrix
    glm::tmat4x4<T> m2w = glm::tmat4x4<T>(1.0f);

    // 4. Set the columns (Basis vectors)
    // GLM is Column-Major: mat[column_index] = vec4(vector, 0.0 or 1.0)
    m2w[0] = glm::tvec4<T>(right,        0.0f); // X-axis (Column 0)
    m2w[1] = glm::tvec4<T>(up_rectified, 0.0f); // Y-axis (Column 1)
    m2w[2] = glm::tvec4<T>(ax,           0.0f); // Z-axis (Column 2)

    // 5. Set the translation column
    m2w[3] = glm::tvec4<T>(translation,  1.0f); // Position (Column 3)

    return m2w;
}



template<typename T>
inline glm::tmat4x4<T> stra<T>::Dupe(const glm::tvec3<T>& a, const glm::tvec3<T>& b, const glm::tvec3<T>& c, bool flip )
{
    glm::tmat4x4<T> tr(1.);
    T* src = glm::value_ptr(tr) ;

    if( flip == false )
    {
        for(int l=0 ; l < 3 ; l++) src[4*0+l] = a[l] ;
        for(int l=0 ; l < 3 ; l++) src[4*1+l] = b[l] ;
        for(int l=0 ; l < 3 ; l++) src[4*2+l] = c[l] ;
    }
    else
    {
        for(int l=0 ; l < 3 ; l++) src[4*l+0] = a[l] ;
        for(int l=0 ; l < 3 ; l++) src[4*l+1] = b[l] ;
        for(int l=0 ; l < 3 ; l++) src[4*l+2] = c[l] ;
    }
    return tr ;
}








template<typename T>
inline T stra<T>::Maxdiff_from_Identity(const glm::tmat4x4<T>& m)
{
    T mxdif = 0. ;
    for(int j=0 ; j < 4 ; j++ )
    for(int k=0 ; k < 4 ; k++ )
    {
        T val = m[j][k] ;
        T xval = j == k ? T(1) : T(0) ;
        T dif = std::abs( val - xval ) ;
        if(dif > mxdif) mxdif = dif ;
    }
    return mxdif ;
}

template<typename T>
inline bool stra<T>::IsIdentity(const glm::tmat4x4<T>& m, T epsilon )
{
    T mxdif = Maxdiff_from_Identity(m);
    return mxdif < epsilon ;
}

template<typename T>
inline bool stra<T>::IsIdentity(const glm::tmat4x4<T>& t, const glm::tmat4x4<T>& v, T epsilon )
{
    return IsIdentity(t, epsilon) && IsIdentity(v, epsilon) ;
}

template<typename T>
inline void stra<T>::Rows(
    glm::tvec4<T>& q0,
    glm::tvec4<T>& q1,
    glm::tvec4<T>& q2,
    glm::tvec4<T>& q3,
    const glm::tmat4x4<T>& t )
{
    memcpy( glm::value_ptr(q0), glm::value_ptr(t) +  0,  4*sizeof(T) );
    memcpy( glm::value_ptr(q1), glm::value_ptr(t) +  4,  4*sizeof(T) );
    memcpy( glm::value_ptr(q2), glm::value_ptr(t) +  8,  4*sizeof(T) );
    memcpy( glm::value_ptr(q3), glm::value_ptr(t) + 12,  4*sizeof(T) );
}

template<typename T>
void stra<T>::Min(glm::tvec4<T>& q, const glm::tvec4<T>& a, const glm::tvec4<T>& b)
{
    q.x = std::min( a.x, b.x );
    q.y = std::min( a.y, b.y );
    q.z = std::min( a.z, b.z );
    q.w = std::min( a.w, b.w );
}

template<typename T>
void stra<T>::Max(glm::tvec4<T>& q, const glm::tvec4<T>& a, const glm::tvec4<T>& b)
{
    q.x = std::max( a.x, b.x );
    q.y = std::max( a.y, b.y );
    q.z = std::max( a.z, b.z );
    q.w = std::max( a.w, b.w );
}




/**
stra::Transform_AABB
---------------------

Impl from sqat4.h transform_aabb_inplace

**/

template<typename T>
inline void stra<T>::Transform_AABB( T* aabb_1, const T* aabb, const glm::tmat4x4<T>& t )
{
    glm::tvec4<T> q0(0.);
    glm::tvec4<T> q1(0.);
    glm::tvec4<T> q2(0.);
    glm::tvec4<T> q3(0.);

    Rows(q0,q1,q2,q3,t);

    T x0 = *(aabb+0) ;
    T y0 = *(aabb+1) ;
    T z0 = *(aabb+2) ;
    T x1 = *(aabb+3) ;
    T y1 = *(aabb+4) ;
    T z1 = *(aabb+5) ;

    glm::tvec4<T> xa = q0 * x0  ;
    glm::tvec4<T> xb = q0 * x1 ;
    glm::tvec4<T> xmi ;
    glm::tvec4<T> xma ;
    stra<T>::Min( xmi, xa, xb);
    stra<T>::Max( xma, xa, xb);

    glm::tvec4<T> ya = q1 * y0 ;
    glm::tvec4<T> yb = q1 * y1 ;
    glm::tvec4<T> ymi ;
    glm::tvec4<T> yma ;
    stra<T>::Min( ymi, ya, yb);
    stra<T>::Max( yma, ya, yb);

    glm::tvec4<T> za = q2 * z0 ;
    glm::tvec4<T> zb = q2 * z1 ;
    glm::tvec4<T> zmi ;
    glm::tvec4<T> zma ;
    stra<T>::Min( zmi, za, zb);
    stra<T>::Max( zma, za, zb);

    *(aabb_1+0) = xmi.x + ymi.x + zmi.x + q3.x ;
    *(aabb_1+1) = xmi.y + ymi.y + zmi.y + q3.y ;
    *(aabb_1+2) = xmi.z + ymi.z + zmi.z + q3.z ;
    *(aabb_1+3) = xma.x + yma.x + zma.x + q3.x ;
    *(aabb_1+4) = xma.y + yma.y + zma.y + q3.y ;
    *(aabb_1+5) = xma.z + yma.z + zma.z + q3.z ;
}

template<typename T>
inline void stra<T>::Transform_AABB_Inplace( T* aabb, const glm::tmat4x4<T>& t )
{
    Transform_AABB( aabb, aabb, t );
}



template<typename T>
inline void stra<T>::Transform_Vec( glm::tvec4<T>& pos, const glm::tvec4<T>&  pos0, const glm::tmat4x4<T>& t )
{
    pos = t * pos0 ;
}




template<typename T>
inline void stra<T>::Transform_Strided( T* _pos, const T* _pos0, int ni, int nj, int item_stride, const glm::tmat4x4<T>& t, T w  )
{
    assert( nj == 3 || nj == 4 );
    int stride = item_stride == 0 ? nj : item_stride ;

    for(int i=0 ; i < ni ; i++)
    {
        glm::tvec4<T> pos0 ;

        for(int j=0 ; j < nj ; j++)
        {
            int idx = i*stride + j ;
            switch(j)
            {
                case 0: pos0.x = _pos0[idx] ; break ;
                case 1: pos0.y = _pos0[idx] ; break ;
                case 2: pos0.z = _pos0[idx] ; break ;
                case 3: pos0.w = _pos0[idx] ; break ;
            }
        }
        if( nj == 3 ) pos0.w = w ;

        glm::tvec4<T> pos = t * pos0 ;


        for(int j=0 ; j < nj ; j++)
        {
            int idx = i*stride + j ;
            switch(j)
            {
                case 0: _pos[idx] = pos.x ; break ;
                case 1: _pos[idx] = pos.y ; break ;
                case 2: _pos[idx] = pos.z ; break ;
                case 3: _pos[idx] = pos.w ; break ;
            }
        }
    }
}


template<typename T>
inline void stra<T>::Transform_Strided_Inplace( T* _pos, int ni, int nj, int item_stride, const glm::tmat4x4<T>& t, T w  )
{
    const T* _pos0 = const_cast<const T*>(_pos) ;
    Transform_Strided( _pos, _pos0, ni, nj, item_stride, t, w );
}




/**
stra::Transform_Data
----------------------

1. form pos0:tvec4 from _pos0:T pointer using w param (so pos0 can be three elements)
2. pre-multiply transform t and pos0 to give pos:tvec4
3. copy pos elements into _pos

**/

template<typename T>
inline void stra<T>::Transform_Data( T* _pos, const T* _pos0,  const glm::tmat4x4<T>* t_ptr, T w  )
{
    glm::tvec4<T> pos0 ;
    pos0.x = *(_pos0 + 0) ;
    pos0.y = *(_pos0 + 1) ;
    pos0.z = *(_pos0 + 2) ;
    pos0.w = w ;

    //std::cout << "stra::Transform_Data  pos0 " << glm::to_string(pos0) << std::endl ;

    glm::tvec4<T> pos = (t_ptr == nullptr ) ? pos0 : (*t_ptr) * pos0 ;

    //std::cout << "stra::Transform_Data  pos  " << glm::to_string(pos) << std::endl ;

    *(_pos+0) = pos.x ;
    *(_pos+1) = pos.y ;
    *(_pos+2) = pos.z ;
}

template<typename T>
inline void stra<T>::Transform_Array( NP* d , const NP* s, const glm::tmat4x4<T>* t, T w , int stride, int offset ) // static
{
    assert( s->shape.size() == 2 && s->shape[1] >= 3 );
    assert( s->shape == d->shape );
    assert( s->uifc == d->uifc );

    int num_item = s->shape[0] ;
    if(stride == 0) stride = s->num_itemvalues() ;

    bool dump = false ;
    if(dump) std::cout
         << "stra::Transform_Array "
         << " num_item " << num_item
         << " stride " << stride
         <<  std::endl
         ;

    const T* ss = s->cvalues<T>();
    T* dd = d->values<T>();

    for(int i=0 ; i < num_item ; i++)
    {
        const T* v0 = ss + i*stride + offset ;
        T*       v  = dd + i*stride + offset ;
        Transform_Data( v, v0, t, w );
    }
}

template<typename T>
inline NP* stra<T>::MakeTransformedArray(const NP* a, const glm::tmat4x4<T>* t, T w , int stride, int offset )
{
    bool dump = false ;
    if(dump) std::cout << "[ stra::MakeTransformedArray" << std::endl ;

    NP* b = NP::MakeLike(a);
    Transform_Array( b, a, t, w, stride, offset );

    if(dump) std::cout << "] stra::MakeTransformedArray" << std::endl ;
    return b ;
}


/**
stra::Copy_Columns_3x4
-----------------------

* After sqat4.h qat4::copy_columns_3x4

* Suitable for filling optixInstance transforms

* Assumes standard OpenGL memory layout of the 16 elements of source
  with translation in slots 12,13,14::

     . 0  1  2  3
       4  5  6  7
       8  9 10 11
     [12 13 14]15     tx ty tz

dst::

     . 0  4  8  -
       1  5  9  -
       2  6 10  -
       3  7 11  -

**/

template<typename T>
inline void stra<T>::Copy_Columns_3x4( T* dst, const T* src ) // static
{
    dst[0] = src[0] ;
    dst[1] = src[4] ;
    dst[2] = src[8] ;
    dst[3] = src[12] ;

    dst[4] = src[1] ;
    dst[5] = src[5] ;
    dst[6] = src[9] ;
    dst[7] = src[13] ;

    dst[8]  = src[2] ;
    dst[9]  = src[6] ;
    dst[10] = src[10] ;
    dst[11] = src[14] ;
}

template<typename T>
inline void stra<T>::Copy_Columns_3x4( T* dst, const glm::tmat4x4<T>& src )
{
    Copy_Columns_3x4( dst, glm::value_ptr(src) );
}

template<typename T>
inline void stra<T>::Copy_Columns_3x4( glm::tmat4x4<T>& dst, const glm::tmat4x4<T>& src )
{
    Copy_Columns_3x4( glm::value_ptr(dst), glm::value_ptr(src) );
}




template<typename T>
inline std::ostream& operator<<(std::ostream& os, const glm::tmat4x4<T>& m)
{
    os << stra<T>::Desc(m) ;
    return os;
}

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const glm::tvec4<T>& v)
{
    os << stra<T>::Desc(v) ;
    return os;
}



