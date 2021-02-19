/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <array>
#include <iterator>
#include <csignal>

#include "SDigest.hh"
#include "NPY.hpp"

#include "NPlane.hpp"
#include "NGLMExt.hpp"
#include "NGLMCF.hpp"
#include "GLMFormat.hpp"

#include <glm/gtx/component_wise.hpp> 
#include <glm/gtx/matrix_operation.hpp>


#include "PLOG.hh"


/**
nglmext::GetEyeUVW
--------------------

Used for example from examples/UseOptiXGeometry 

When the gaze direction and the up direction coincide this 
yields 



Adapted from Composition::getEyeUVW and examples/UseGeometryShader:getMVP

**/

const plog::Severity nglmext::LEVEL = PLOG::EnvLevel("nglmext", "DEBUG"); 


void nglmext::GetEyeUVW(
    const glm::vec4& ce, 
    const glm::vec3& _eye_m, 
    const glm::vec3& _look_m, 
    const glm::vec3& _up_m, 
    const unsigned width, 
    const unsigned height, 
    const float tanYfov, // reciprocal of camera zoom  
    glm::vec3& eye, 
    glm::vec3& U, 
    glm::vec3& V, 
    glm::vec3& W, 
    const bool dump
    )
{
    glm::vec3 tr(ce.x, ce.y, ce.z);  // ce is center-extent of model
    glm::vec3 sc(ce.w);
    glm::vec3 isc(1.f/ce.w);
    // model frame unit coordinates from/to world 
    glm::mat4 model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);
    //glm::mat4 world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr);
    const glm::mat4& m2w = model2world ; 


   // View::getTransforms
    glm::vec4 eye_m(  _eye_m ,1.f);  //  viewpoint in unit model frame 
    glm::vec4 look_m( _look_m,1.f); 
    glm::vec4 up_m(   _up_m  ,1.f); 

    glm::vec4 gze_m( look_m - eye_m ) ; 

    glm::vec3 eye_ = glm::vec3( m2w * eye_m ) ; 
    //glm::vec3 look = glm::vec3( m2w * look_m ) ; 
    glm::vec3 up = glm::vec3( m2w * up_m ) ; 
    glm::vec3 gaze = glm::vec3( m2w * gze_m ) ;    


    float epsilon = 1e-5 ; 
    glm::vec3 up_original(up);    
    int rc = HandleDegenerateGaze(up, gaze, epsilon, dump); 
    if(rc == 1 && dump)
    {
        LOG(info) 
            << " up vector changed by HandleDegenerateGaze "
            << " up_original " << glm::to_string(up_original)
            << " up " << glm::to_string(up)
            << " gaze " << glm::to_string(gaze)
            ;
    }


    glm::vec3 forward_ax = glm::normalize(gaze);
    glm::vec3 right_ax   = glm::normalize(glm::cross(forward_ax,up));  
    // right hand: palm horizontal facing upwards curled from directly ahead to up, thumb to the right 

    glm::vec3 top_ax     = glm::normalize(glm::cross(right_ax,forward_ax));
    // right hand: vertical palm facing left curled from right around to gaze front, thumb upwards 


    float aspect = float(width)/float(height) ;
    float gazelength = glm::length( gaze ) ; 
    float v_half_height = gazelength * tanYfov ;
    float u_half_width  = v_half_height * aspect ;

    U = right_ax * u_half_width ;
    V = top_ax * v_half_height ;
    W = forward_ax * gazelength ; 
    eye = eye_ ; 


    if(dump)
    {
        std::cout << gpresent("m2w",m2w) << std::endl ;  
        std::cout << "gze" << gpresent(gaze) << std::endl ;  
        std::cout << "up"  << gpresent(up) << std::endl ;  
        std::cout << "for" << gpresent(forward_ax) << std::endl ;  
        std::cout << "rgt" << gpresent(right_ax) << std::endl ;  
        std::cout << "top" << gpresent(top_ax) << std::endl ;  
 
        std::cout << std::setw(10) << "ce"      << gpresent(ce) << std::endl ; 
        std::cout << std::setw(10) << "eye_m "  << gpresent(eye_m) << std::endl ; 
        std::cout << std::setw(10) << "look_m " << gpresent(look_m) << std::endl ; 
        std::cout << std::setw(10) << "up_m "   << gpresent(up_m) << std::endl ; 

        std::cout << std::setw(10) << "eye"  << gpresent(eye) << std::endl ; 
        std::cout << std::setw(10) << "U "   << gpresent(U) << std::endl ; 
        std::cout << std::setw(10) << "V "   << gpresent(V) << std::endl ; 
        std::cout << std::setw(10) << "W "   << gpresent(W) << std::endl ; 

    }


}




/**
nglmext::HandleDegenerateGaze
--------------------------------

Gaze directions parallel to the up vector yield
a degenerate basis. In these cases the up vector 
is changed to the first other axis that is not degenerate.

cf OpticksCore/View::handleDegenerates

**/
int nglmext::HandleDegenerateGaze( glm::vec3& up, const glm::vec3& gaze, const float epsilon, const bool dump ) 
{
    glm::vec3 forward_ax = glm::normalize(gaze);
    float eul = glm::length(glm::cross(forward_ax, up));
    if(eul > epsilon) return 0 ; 

    std::vector<glm::vec3> axes = { 
         {1.f,0.f,0.f}, 
         {0.f,1.f,0.f}, 
         {0.f,0.f,1.f}, 
         {-1.f,0.f,0.f}, 
         {0.f,-1.f,0.f}, 
         {0.f,0.f,-1.f} 
        } ; 

   for(unsigned i=0 ; i < axes.size() ; i++)
   {   
        glm::vec3 axis = axes[i] ; 
        float axd = glm::dot(forward_ax, axis);   // 1.,0. or -1.
        float aul = glm::length(glm::cross(forward_ax, axis)); // 0. or 1. 

        if(dump)
        std::cout << gpresent(axis) << " aul:" << aul << " axd:" << axd << std::endl ;  

        if(aul > 0.f)
        {   
            up = axis ; 
            LOG(LEVEL) << " changing \"up\" to " << glm::to_string(up) ; 
            return 1 ; 
        } 
    }   

    assert( 0 && "failed to change up"); 
    return -1 ; 
}




/**
nglmext::least_parallel_axis
------------------------------

Given a vector direction, this returns the unit axis 
least parallel to it, ie one of:: 

      (1,0,0)
      (0,1,0)
      (0,0,1)

For example an input directopn along Z axis (0,0,1)
yields least parallel axis (1,0,0)

**/

glm::vec3 nglmext::least_parallel_axis( const glm::vec3& dir )
{
    glm::vec3 adir(glm::abs(dir));
    glm::vec3 lpa(0) ; 

    if( adir.x <= adir.y && adir.x <= adir.z )
    {
        lpa.x = 1.f ; 
    }
    else if( adir.y <= adir.x && adir.y <= adir.z )
    {
        lpa.y = 1.f ; 
    }
    else
    {
        lpa.z = 1.f ; 
    }
    return lpa ; 
}

/**
nglmext::pick_transverse_direction
------------------------------------

Obtains a vector which is perpendicular to the input.  As there 
are an infinite number of such transverse directions 
it uses a heuristic approach to yield one of them by first 
finding the coordinate axis least parallel to the input direction
and taking the cross product with that.

For example an input direction (dir) along Z axis (0,0,1)
yields least parallel axis (lpa) along X (1,0,0) 
the cross product lpa ^ dir is then -Y (0,-1,0)
which is a transverse direction (trd)  to the input direction.

        Z
        |
        |
        +----Y 
       /
      /
     X


**/

glm::vec3 nglmext::pick_transverse_direction( const glm::vec3& dir, bool dump)
{
    glm::vec3 lpa = least_parallel_axis(dir) ;
    glm::vec3 trd = glm::normalize( glm::cross( lpa, dir )) ; 

    if(dump)
    {
        std::cout 
                  << "nglext::pick_transverse_direction"
                  << " dir " << gpresent(dir)
                  << " lpa " << gpresent(lpa)
                  << " trd " << gpresent(trd)
                  << std::endl 
                  ;
    }
    return trd ; 
}



void nglmext::copyTransform( std::array<float,16>& dst, const glm::mat4& src )
{
    const float* p = glm::value_ptr(src);
    std::copy(p, p+16, std::begin(dst));
}


std::string nglmext::xform_string( const std::array<float, 16>& xform )
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < 16 ; i++) 
    {
        bool translation =  i == 12 || i == 13 || i == 14 ; 
        int fwid = translation ? 8 : 6 ;  
        int fprec = translation ? 2 : 3 ; 
        ss << std::setw(fwid) << std::fixed << std::setprecision(fprec) << xform[i] << ' ' ; 
    }
    return ss.str();
}


// Extracts from /usr/local/opticks/externals/yoctogl/yocto-gl/yocto/yocto_gltf.cpp

std::array<float, 16> nglmext::_float4x4_mul( const std::array<float, 16>& a, const std::array<float, 16>& b) 
{
    auto c = std::array<float, 16>();
    for (auto i = 0; i < 4; i++) {
        for (auto j = 0; j < 4; j++) {
            c[j * 4 + i] = 0;
            for (auto k = 0; k < 4; k++)
                c[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
        }
    }
    return c;
}

const std::array<float, 16> nglmext::_identity_float4x4 = {{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};




glm::dmat4 nglmext::upconvert( const glm::mat4& ft )
{
    glm::dmat4 dt(1.) ; 
    for(int i=0 ; i < 4 ; i++) 
    for(int j=0 ; j < 4 ; j++)
       dt[i][j] = double(ft[i][j]) ; 
    return dt ; 
}

glm::mat4 nglmext::downconvert( const glm::dmat4& dt )
{
    glm::mat4 ft(1.f) ; 
    for(int i=0 ; i < 4 ; i++) 
    for(int j=0 ; j < 4 ; j++)
       ft[i][j] = float(dt[i][j]) ; 
    return dt ; 
}




glm::mat4 nglmext::invert_tr( const glm::mat4& tr )
{
    /**
       input transforms are rotation first then translation :  T*R*v
     
       invert by dis-membering tr into r and t by inspection and separately  
       transpose the rotation and negate the translation then 
       multiply in reverse order

               IR*IT 
    **/

    glm::mat4 ir = glm::transpose(glm::mat4(glm::mat3(tr)));
    glm::mat4 it = glm::translate(glm::mat4(1.f), -glm::vec3(tr[3])) ; 
    glm::mat4 irit = ir*it ;    // <--- inverse of tr 
    return irit ; 
}

glm::mat4* nglmext::invert_tr( const glm::mat4* tr )
{
    if(tr == NULL) return NULL ; 
    return new glm::mat4( invert_tr(*tr) );
}

glm::mat4 nglmext::average_to_inverse_transpose( const glm::mat4& m )
{
    glm::mat4 it = glm::inverse(glm::transpose(m)) ;
    return (m + it)/2.f ;
}

template <typename T>
glm::tmat4x4<T> nglmext::average_to_inverse_transpose_( const glm::tmat4x4<T>& m )
{
    glm::tmat4x4<T> it = glm::inverse(glm::transpose(m)) ;
    return (m + it)/T(2) ;
}




/**
nglmext::polar_decomposition
-----------------------------

Decompose a 4x4 TRS (translate-rotate-scale) matrix 
into T,R and S component matrices and their inverses
held within the ndeco struct.

**/


void nglmext::polar_decomposition( const glm::mat4& trs, ndeco& d,  bool verbose )
{
    d.t = glm::translate(glm::mat4(1.f), glm::vec3(trs[3])) ; 
    d.it = glm::translate(glm::mat4(1.f), -glm::vec3(trs[3])) ; 

    d.rs = glm::mat4(glm::mat3(trs)) ;

    glm::mat4 prev = d.rs ; 
    glm::mat4 next ; 

    float diff, diff2  ; 
    int count(0) ; 
    do {
        next = average_to_inverse_transpose( prev ) ;
        diff = compDiff(prev, next) ;
        diff2 = compDiff2(prev, next) ;
        prev = next ; 

        if(verbose)
        std::cout << "polar_decomposition"
                  << " diff " << diff 
                  << " diff2 " << diff2 
                  << " count " << count 
                  << std::endl ; 

    } while( ++count < 100 && diff > 0.0001f ); 

    d.r = next ;
    d.ir = glm::transpose(d.r) ;
    d.s = glm::transpose(d.r) * d.rs ;   //  input rs matrix M,  S = R^-1 M

    glm::vec4 isca(0,0,0,1) ; 
    for(unsigned i=0 ; i < 3 ; i++) isca[i] = 1.f/d.s[i][i] ; 
    
    d.is = glm::diagonal4x4(isca);

    d.isirit = d.is * d.ir * d.it ; 
    d.trs = d.t * d.r * d.s  ; 
    d.tr = d.t * d.r ;
}

template<typename T>
void nglmext::polar_decomposition_( const glm::tmat4x4<T>& trs, ndeco_<T>& d,  bool verbose )
{
    d.t = glm::translate(glm::tmat4x4<T>(1.), glm::tvec3<T>(trs[3])) ; 
    d.it = glm::translate(glm::tmat4x4<T>(1.), -glm::tvec3<T>(trs[3])) ; 

    d.rs = glm::tmat4x4<T>(glm::tmat3x3<T>(trs)) ;

    glm::tmat4x4<T> prev = d.rs ; 
    glm::tmat4x4<T> next ; 

    T diff, diff2 ;
    T epsilon(0.0001);  
 
    int count(0) ; 
    do {
        next = average_to_inverse_transpose_( prev ) ;
        diff = compDiff_(prev, next) ;
        diff2 = compDiff2_(prev, next) ;
        prev = next ; 

        if(verbose)
        std::cout << "polar_decomposition"
                  << " diff " << diff 
                  << " diff2 " << diff2 
                  << " count " << count 
                  << std::endl ; 

    } while( ++count < 100 && diff > epsilon ); 


    d.r = next ;
    d.ir = glm::transpose(d.r) ;
    d.s = glm::transpose(d.r) * d.rs ;   //  input rs matrix M,  S = R^-1 M

    glm::tvec4<T> isca(0,0,0,1) ; 
    for(unsigned i=0 ; i < 3 ; i++) isca[i] = T(1)/d.s[i][i] ; 
    
    d.is = glm::diagonal4x4(isca);

    d.isirit = d.is * d.ir * d.it ; 
    d.trs = d.t * d.r * d.s  ; 
    d.tr = d.t * d.r ;
}

 

glm::vec3 nglmext::pluck_scale( const ndeco& d )
{
    glm::vec3 scale(0,0,0); 
    scale.x = d.s[0][0] ; 
    scale.y = d.s[1][1] ; 
    scale.z = d.s[2][2] ; 
    return scale ; 
}

bool nglmext::has_scale( const glm::vec3& scale, float epsilon )
{ 
    glm::vec3 identity(1.f) ; 
    glm::vec3 delta(scale - identity); 
    return glm::length( delta ) > epsilon ;  
}



glm::vec3 nglmext::pluck_translation( const glm::mat4& t )
{
    glm::vec3 tla(0,0,0); 
    tla.x = t[3].x ; 
    tla.y = t[3].y ; 
    tla.z = t[3].z ; 
    return tla ; 
}

bool nglmext::is_identity(const glm::mat4& t, float eps) 
{
    glm::mat4 id(1.0) ; 
    float dt = nglmext::compDiff(t, id);
    return dt < eps  ; 
}


/**
nglmext::invert_trs
---------------------

The match bool should return true, when it returns false
it indicates possible precision problems with the inverse.

**/


glm::mat4 nglmext::invert_trs( const glm::mat4& trs, bool& match )
{
    /**
    Input transforms are TRS (scale first, then rotate, then translate)::

          T*R*S*v

    invert by dis-membering trs into rs and t by inspection 
    then extract the r by polar decomposition, ie by 
    iteratively averaging with the inverse transpose until 
    the iteration stops changing much ... at which point
    are left with the rotation portion

    Then separately transpose the rotation,
    negate the translation and reciprocate the scaling 
    and multiply in reverse order

          IS*IR*IT

    The result should be close to directly taking 
    the inverse and has advantage that it tests the form 
    of the transform.
 
    **/

    bool verbose = false ; 
    ndeco d ;
    polar_decomposition( trs, d, verbose) ;
    glm::mat4 isirit = d.isirit ; 
    glm::mat4 i_trs = glm::inverse( trs ) ; 

    NGLMCF cf(isirit, i_trs );

    if(!cf.match) 
    {
        LOG(error) << cf.desc("ngmext::invert_trs polar_decomposition inverse and straight inverse are mismatched " );
    }

    match = cf.match ; 

    return isirit ; 
}


/**
nglmext::compDiff
------------------

Maximum absolute componentwise difference 

**/


float nglmext::compDiff(const glm::vec2& a , const glm::vec2& b )
{
    glm::vec2 amb = a - b ; 
    glm::vec2 aamb = glm::abs(amb) ; 
    return glm::compMax(aamb) ; 
}

float nglmext::compDiff(const glm::vec3& a , const glm::vec3& b )
{
    glm::vec3 amb = a - b ; 
    glm::vec3 aamb = glm::abs(amb) ; 
    return glm::compMax(aamb) ; 
}
float nglmext::compDiff(const glm::vec4& a , const glm::vec4& b )
{
    glm::vec4 amb = a - b ; 
    glm::vec4 aamb = glm::abs(amb) ; 
    return glm::compMax(aamb) ; 
}


float nglmext::compDiff(const glm::mat3& a , const glm::mat3& b )
{
    glm::mat3 amb = a - b ; 

    glm::mat3 aamb ; 
    for(unsigned i=0 ; i < 3 ; i++) aamb[i] = glm::abs(amb[i]) ; 

    glm::vec3 colmax ; 
    for(unsigned i=0 ; i < 3 ; i++) colmax[i] = glm::compMax(aamb[i]) ; // compMax returns float, here using index addressing of vec4

    return glm::compMax(colmax) ; 
}

float nglmext::compDiff(const glm::mat4& a , const glm::mat4& b )
{
    glm::mat4 amb = a - b ; 

    glm::mat4 aamb ; 
    for(unsigned i=0 ; i < 4 ; i++) aamb[i] = glm::abs(amb[i]) ; 

    glm::vec4 colmax ; 
    for(unsigned i=0 ; i < 4 ; i++) colmax[i] = glm::compMax(aamb[i]) ; // compMax returns float, here using index addressing of vec4

    return glm::compMax(colmax) ; 
}

template<typename T>
T nglmext::compDiff_(const glm::tmat4x4<T>& a , const glm::tmat4x4<T>& b )
{
    glm::tmat4x4<T> amb = a - b ; 

    glm::tmat4x4<T> aamb ; 
    for(unsigned i=0 ; i < 4 ; i++) aamb[i] = glm::abs(amb[i]) ; 

    glm::tvec4<T> colmax ; 
    for(unsigned i=0 ; i < 4 ; i++) colmax[i] = glm::compMax(aamb[i]) ; // compMax returns T, here using index addressing of vec4

    return glm::compMax(colmax) ; 
}








/*
In [1]: a = 2.16489e-17

In [2]: b = 0 

In [3]: (a+b)/2
Out[3]: 1.082445e-17

In [4]: avg = (a+b)/2

In [5]: ab = a-b 

In [6]: ab/avg
Out[6]: 2.0

*/

float nglmext::compDiff2(const float a_ , const float b_, bool fractional, float epsilon  )
{
    float a = fabsf(a_) < epsilon  ? 0.f : a_ ; 
    float b = fabsf(b_) < epsilon  ? 0.f : b_ ; 

    float d = fabsf(a - b);

    float denom = (a+b)/2.f ; 
    if(fractional && denom != 0.f) d /= denom    ; 
    return d ; 
}


template<typename T>
T nglmext::abs_(T v)
{
   return T(0) ; 
}
template <> 
float nglmext::abs_<float>(float v) 
{ 
   return std::abs(v) ; 
} 
template <> 
double nglmext::abs_<double>(double v) 
{ 
   return std::abs(v) ; 
} 

template<typename T>
T nglmext::compDiff2_(const T a_ , const T b_, bool fractional, T epsilon  )
{
    T a = abs_(a_) < epsilon  ? T(0) : a_ ; 
    T b = abs_(b_) < epsilon  ? T(0) : b_ ; 
    T d = abs_(a - b);
    T denom = (a+b)/T(2) ; 
    if(fractional && denom != T(0)) d /= denom    ; 
    return d ; 
}




float nglmext::compDiff2(const glm::mat4& a_ , const glm::mat4& b_, bool fractional, float epsilon, float epsilon_translation )
{
    float a, b, d, maxdiff = 0.f ; 
    for(unsigned i=0 ; i < 4 ; i++){
    for(unsigned j=0 ; j < 4 ; j++)
    { 
        a = a_[i][j] ; 
        b = b_[i][j] ; 
        d = compDiff2(a, b, fractional, i == 3 ? epsilon_translation : epsilon );
        if( d > maxdiff ) maxdiff = d ; 
    }
    }
    return maxdiff ; 
}

template<typename T>
T nglmext::compDiff2_(const glm::tmat4x4<T>& a_ , const glm::tmat4x4<T>& b_, bool fractional, T  epsilon, T epsilon_translation )
{
    T a, b, d, maxdiff = T(0.) ; 
    for(unsigned i=0 ; i < 4 ; i++){
    for(unsigned j=0 ; j < 4 ; j++)
    { 
        a = a_[i][j] ; 
        b = b_[i][j] ; 
        d = compDiff2_(a, b, fractional, i == 3 ? epsilon_translation : epsilon );
        if( d > maxdiff ) maxdiff = d ; 
    }
    }
    return maxdiff ; 
}









glm::mat4 nglmext::make_transpose(const glm::mat4& t )
{
   return glm::transpose(t);
}

glm::mat4 nglmext::make_translate(const float x, const float y, const float z)
{
   glm::vec3 tlate(x,y,z);
   return make_translate(tlate);
}

glm::mat4 nglmext::make_rotate(const float x, const float y, const float z, const float w)
{
   glm::vec4 aa(x,y,z,w);
   return make_rotate(aa);
}
glm::mat4 nglmext::make_scale(const float x, const float y, const float z)
{
   glm::vec3 tsc(x,y,z);
   return make_scale(tsc);
}







glm::mat4 nglmext::make_translate(const glm::vec3& tlat)
{
    return glm::translate(glm::mat4(1.f), tlat);
}



glm::mat4 nglmext::make_rotate(const glm::vec4& axis_angle)
{
    float angle = nglmext::angle_radians(axis_angle.w) ; 
    return glm::rotate(glm::mat4(1.f), angle , glm::vec3(axis_angle)) ;
}
glm::mat4 nglmext::make_scale(const glm::vec3& scal)
{
    return glm::scale(glm::mat4(1.f), scal);
}


float nglmext::angle_radians(float angle_degrees)
{
    return glm::pi<float>()*angle_degrees/180.f ; 
}

template<typename T>
T nglmext::angle_radians_(T angle_degrees)
{
    return glm::pi<T>()*angle_degrees/T(180.) ; 
}





/**
nglmext::make_transform
--------------------------

For fourth column translation unmodified the "t" must come last, ie "trs" ?? shouldnt that be order="srt"
    
Despite this intution from reading the code  there is the row-major business too..
    
See tests/NGLMExtTests.cc:test_make_transform it shows that 

1. order argument "trs" : does scaling and rotation before translation, 
2. order argument "srt" : does translation before rotation and scaling
    
Usually "trs" is the most convenient order to use

* because want to orient about a nearby origin before translating into position
* what is confusing is that to get the translation done last, 
  needs to do glm::translate first 
   

**/


glm::mat4 nglmext::make_transform(const std::string& order, const glm::vec3& tlat, const glm::vec4& axis_angle, const glm::vec3& scal )
{
    glm::mat4 mat(1.f) ;
    
    float angle = nglmext::angle_radians(axis_angle.w) ; 

    for(unsigned i=0 ; i < order.length() ; i++)
    {
        switch(order[i])
        {
           case 's': mat = glm::scale(mat, scal)         ; break ; 
           case 'r': mat = glm::rotate(mat, angle , glm::vec3(axis_angle)) ; break ; 
           case 't': mat = glm::translate(mat, tlat )    ; break ; 
        }
    }
    return mat  ; 
}


template<typename T>
glm::tmat4x4<T> nglmext::make_transform_(const std::string& order, const glm::tvec3<T>& tlat, const glm::tvec4<T>& axis_angle, const glm::tvec3<T>& scal )
{
    glm::tmat4x4<T> mat(1.) ;
    
    T angle = nglmext::angle_radians_<T>(axis_angle.w) ; 

    for(unsigned i=0 ; i < order.length() ; i++)
    {
        switch(order[i])
        {
           case 's': mat = glm::scale(mat, scal)         ; break ; 
           case 'r': mat = glm::rotate(mat, angle , glm::tvec3<T>(axis_angle)) ; break ; 
           case 't': mat = glm::translate(mat, tlat )    ; break ; 
        }
    }
    return mat  ; 
}




glm::mat4 nglmext::make_transform(const std::string& order)
{
    glm::vec3 tla(0,0,100) ; 
    glm::vec4 rot(0,0,1,45);
    glm::vec3 sca(1,1,1) ; 

    return make_transform(order, tla, rot, sca );
}


glm::mat4 nglmext::make_rotate_a2b(const glm::vec3& a, const glm::vec3& b, bool dump)
{   
    float ab = glm::dot(a, b);  // 0. when perpendicualar
    float la = glm::length(a) ;     
    float lb = glm::length(b) ;      
    float cos_angle = ab/(lb*la) ;  // 0. when perpendicular,  1. or -1. when collinear
    float angle_radians = acos(cos_angle);  //  pi/2 when perpendicular,  0 or pi when collinear 
    
    glm::vec3 axb = glm::cross(a, b);

    float epsilon = 1e-5 ;    
    float len_axb = glm::length(axb); 

    glm::vec3 axis = len_axb < epsilon ? nglmext::pick_transverse_direction(a, dump) : axb ; 
    // collinear a and b prevents obtaining a rotation axis from their cross product 
    // so pick some arbitrary transverse direction to act as the axis


    if(dump)
    {
        LOG(info) 
            << " len_axb " << len_axb 
            << " axis " << glm::to_string(axis) 
            ;
    }

    glm::mat4 rot = glm::rotate(angle_radians, axis );  
    // matrix to rotate a->b from angle_radians and axis 
    
    return rot ;
}



glm::mat4 nglmext::make_rotate_a2b_then_translate( const glm::vec3& a, const glm::vec3& b, const glm::vec3& tlat, bool dump )
{
    glm::mat4 rotate= make_rotate_a2b(a,b, dump);         
    glm::mat4 translate = glm::translate(glm::mat4(1.0f), tlat );
    glm::mat4 rotate_then_translate = translate * rotate ; 
    return rotate_then_translate  ;   
}







void nglmext::_pick_up( glm::vec3& up, const glm::vec3& dir ) // static
{
    std::vector<glm::vec3> axes ; 
    axes.push_back( glm::vec3(1,0,0) );
    axes.push_back( glm::vec3(0,1,0) );
    axes.push_back( glm::vec3(0,0,1) );

    for(unsigned i=0 ; i < axes.size() ; i++)
    {   
       glm::vec3 axis = axes[i] ;
       float aul = glm::length(glm::cross(dir, axis));
       if(aul > 0.f)
       {
           up = axis ; 
           return ; 
       }
    }
}

void nglmext::_define_uv_basis( const glm::vec3& perp, glm::vec3& udir, glm::vec3& vdir   ) // static
{
    glm::vec3 up ;     
    _pick_up(up, perp );

    udir = glm::normalize(glm::cross(perp, up));
    vdir = glm::normalize(glm::cross(udir, perp)); 

    // see okc- View::getFocalBasis View::handleDegenerates
}

void nglmext::_define_uv_basis( const std::vector<glm::vec4>& perps, std::vector<glm::vec3>& udirs, std::vector<glm::vec3>& vdirs   ) // static
{
    for(unsigned i=0 ; i < perps.size() ; i++)
    {
        glm::vec3 perp(perps[i]);

        glm::vec3 udir ; 
        glm::vec3 vdir ; 
        nglmext::_define_uv_basis(perp, udir, vdir) ;

        udirs.push_back(udir);
        vdirs.push_back(vdir);
    }
}
 

glm::mat4 nglmext::make_yzflip(){ return make_flip(1,2) ; }  // static

glm::mat4 nglmext::make_flip(unsigned axa, unsigned axb) // static
{
    assert( axa < 3 && axb < 3 && axa != axb ); 

    glm::mat4 m(1.f) ; 

    m[axa][axa] = 0.f ; 
    m[axb][axb] = 0.f ; 
    m[axb][axa] = 1.f ; 
    m[axa][axb] = 1.f ; 

    return m ; 
}















void nglmext::transform_planes( NPY<float>* plan_buffer, const glm::mat4& placement ) //static
{
    // Formerly all geometry that required planes (eg trapezoids) 
    // was part of instanced solids... so this was not needed.
    // BUT for debugging it is useful to be able to operate in global mode
    // whilst testing small subsets of geometry
    //   
    //if(nplane > 0 ) assert(0 && "plane placement not implemented" );
    assert(plan_buffer->hasShape(-1,4));
    unsigned num_plane = plan_buffer->getNumItems();
    for(unsigned i=0 ; i < num_plane ; i++ )
    {    
        glm::vec4 pl = plan_buffer->getQuad_(i);
        nplane*  plane = make_plane(pl);
        glm::vec4 tpl = plane->make_transformed(placement);

        plan_buffer->setQuad( tpl, i ); 
    }    
}






std::ostream& operator<< (std::ostream& out, const glm::ivec3& v) 
{
    out << "{" 
        << " " << std::setw(4) << v.x 
        << " " << std::setw(4) << v.y 
        << " " << std::setw(4) << v.z
        << "}";
    return out;
}




std::ostream& operator<< (std::ostream& out, const glm::vec3& v) 
{
    out << "{" 
        << " " << std::fixed << std::setprecision(4) << std::setw(9) << v.x 
        << " " << std::fixed << std::setprecision(4) << std::setw(9) << v.y
        << " " << std::fixed << std::setprecision(4) << std::setw(9) << v.z 
        << "}";

    return out;
}



std::ostream& operator<< (std::ostream& out, const glm::vec4& v) 
{
    out << "{" 
        << " " << std::setprecision(2) << std::setw(7) << v.x 
        << " " << std::setprecision(2) << std::setw(7) << v.y
        << " " << std::setprecision(2) << std::setw(7) << v.z 
        << " " << std::setprecision(2) << std::setw(7) << v.w 
        << "}";
    return out;
}

std::ostream& operator<< (std::ostream& out, const glm::mat4& v) 
{
    out << "( "
        << " " << v[0]
        << " " << v[1]
        << " " << v[2]
        << " " << v[3]
        << " )"
        ; 

    return out;
}


std::ostream& operator<< (std::ostream& out, const glm::mat3& v) 
{
    out << "( "
        << " " << v[0]
        << " " << v[1]
        << " " << v[2]
        << " )"
        ; 

    return out;
}


template NPY_API glm::tmat4x4<float> nglmext::average_to_inverse_transpose_( const glm::tmat4x4<float>& m );
template NPY_API glm::tmat4x4<double> nglmext::average_to_inverse_transpose_( const glm::tmat4x4<double>& m );

template NPY_API void nglmext::polar_decomposition_( const glm::tmat4x4<float>& trs, ndeco_<float>& d,  bool verbose );
template NPY_API void nglmext::polar_decomposition_( const glm::tmat4x4<double>& trs, ndeco_<double>& d,  bool verbose );

template NPY_API glm::tmat4x4<float> nglmext::make_transform_(const std::string& order, const glm::tvec3<float>& tlat, const glm::tvec4<float>& axis_angle, const glm::tvec3<float>& scal );
template NPY_API glm::tmat4x4<double> nglmext::make_transform_(const std::string& order, const glm::tvec3<double>& tlat, const glm::tvec4<double>& axis_angle, const glm::tvec3<double>& scal );

template NPY_API float nglmext::compDiff_(const glm::tmat4x4<float>& a , const glm::tmat4x4<float>& b );
template NPY_API double nglmext::compDiff_(const glm::tmat4x4<double>& a , const glm::tmat4x4<double>& b );

