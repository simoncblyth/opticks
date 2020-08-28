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





glm::mat4 nglmext::invert_trs( const glm::mat4& trs )
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

    ndeco d ;
    polar_decomposition( trs, d ) ;
    glm::mat4 isirit = d.isirit ; 
    glm::mat4 i_trs = glm::inverse( trs ) ; 

    NGLMCF cf(isirit, i_trs );

    if(!cf.match) 
    {
        LOG(warning) << cf.desc("ngmext::invert_trs polar_decomposition inverse and straight inverse are mismatched " );
    }

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

float nglmext::compDiff(const glm::mat4& a , const glm::mat4& b )
{
    glm::mat4 amb = a - b ; 

    glm::mat4 aamb ; 
    for(unsigned i=0 ; i < 4 ; i++) aamb[i] = glm::abs(amb[i]) ; 

    glm::vec4 colmax ; 
    for(unsigned i=0 ; i < 4 ; i++) colmax[i] = glm::compMax(aamb[i]) ; // compMax returns float, here using index addressing of vec4

    return glm::compMax(colmax) ; 
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

glm::mat4 nglmext::make_transform(const std::string& order)
{
    glm::vec3 tla(0,0,100) ; 
    glm::vec4 rot(0,0,1,45);
    glm::vec3 sca(1,1,1) ; 

    return make_transform(order, tla, rot, sca );
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













std::string nmat4pair::digest() 
{
    return SDigest::digest( (void*)this, sizeof(nmat4pair) );
}


std::string nmat4triple::digest() const 
{
    return SDigest::digest( (void*)this, sizeof(nmat4triple) );
}


nmat4pair* nmat4pair::clone()
{
    return new nmat4pair(t,v);
}

nmat4pair* nmat4pair::product(const std::vector<nmat4pair*>& pairs)
{
    unsigned npairs = pairs.size();
    if(npairs==0) return NULL ; 
    if(npairs==1) return pairs[0] ; 

    glm::mat4 t(1.0) ; 
    glm::mat4 v(1.0) ; 

    for(unsigned i=0,j=npairs-1 ; i < npairs ; i++,j-- )
    {
        const nmat4pair* ii = pairs[i] ; 
        const nmat4pair* jj = pairs[j] ; 

        t *= ii->t ; 
        v *= jj->v ; 
    }

    // guessed multiplication ordering 
    // is this the appropriate transform and inverse transform multiplication order ?
    // ... pairs order is from the leaf back to the root   

    return new nmat4pair(t, v) ; 
}




nmat4pair::nmat4pair(const glm::mat4& t_ ) 
     : 
     t(t_),
     v(nglmext::invert_trs(t))
{
}


nmat4triple::nmat4triple(const float* data ) 
     : 
     t(glm::make_mat4(data)),
     v(nglmext::invert_trs(t)),
     q(glm::transpose(v))
{
}

nmat4triple::nmat4triple(const glm::mat4& t_ ) 
     : 
     t(t_),
     v(nglmext::invert_trs(t)),
     q(glm::transpose(v))
{
}

const nmat4triple* nmat4triple::clone() const 
{
    return new nmat4triple(t,v,q);
}



glm::vec3 nmat4triple::apply_transform_t(const glm::vec3& p_, const float w) const 
{
    ntransformer tr(t, w);
    return tr(p_); 
}

glm::vec3 nmat4triple::apply_transform_v(const glm::vec3& p_, const float w) const 
{
    ntransformer tr(v, w);
    return tr(p_); 
}

void nmat4triple::apply_transform_t(std::vector<glm::vec3>& dst, const std::vector<glm::vec3>& src, float w) const 
{
    ntransformer tr(t, w);
    std::transform(src.begin(), src.end(), std::back_inserter(dst), tr );
}
void nmat4triple::apply_transform_v(std::vector<glm::vec3>& dst, const std::vector<glm::vec3>& src, float w) const 
{
    ntransformer tr(v, w);
    std::transform(src.begin(), src.end(), std::back_inserter(dst), tr );
}

bool nmat4triple::is_equal_to(const nmat4triple* other, float eps) const 
{
    assert( other ) ; 
    float dt = nglmext::compDiff(t, other->t);
    float dv = nglmext::compDiff(v, other->v);
    float dq = nglmext::compDiff(q, other->q);
    return dt < eps && dv < eps && dq < eps ; 
}




glm::vec3 nmat4triple::get_translation() const 
{
    glm::vec3 tla(t[3]) ;  
    return tla ; 
}

bool nmat4triple::is_translation_only(float eps) const 
{
    const glm::mat3 i3(1.f); 
    const glm::mat3 t3(t) ; 
    float dt = nglmext::compDiff(t3, i3);
    return dt < eps ; 
}

bool nmat4triple::is_identity(float eps) const 
{
    glm::mat4 id(1.0) ; 
    float dt = nglmext::compDiff(t, id);
    float dv = nglmext::compDiff(v, id);
    float dq = nglmext::compDiff(q, id);
    return dt < eps && dv < eps && dq < eps ; 
}



const nmat4triple* nmat4triple::make_transform( 
           const float x0, const float y0, const float z0, const float w0,
           const float x1, const float y1, const float z1, const float w1, 
           const float x2, const float y2, const float z2, const float w2, 
           const float x3, const float y3, const float z3, const float w3 
       )  // static
{
    glm::mat4 t(1);

    t[0] = glm::vec4(x0,y0,z0,w0); 
    t[1] = glm::vec4(x1,y1,z1,w1); 
    t[2] = glm::vec4(x2,y2,z2,w2); 
    t[3] = glm::vec4(x3,y3,z3,w3); 

    return new nmat4triple(t);
}


const nmat4triple* nmat4triple::make_translate( const glm::vec3& tlate )
{
    glm::mat4 t = nglmext::make_translate(tlate);
    return new nmat4triple(t);
}
const nmat4triple* nmat4triple::make_scale( const glm::vec3& tsca )
{
    glm::mat4 s = nglmext::make_scale(tsca);
    return new nmat4triple(s);
}
const nmat4triple* nmat4triple::make_rotate( const glm::vec4& trot )
{
    glm::mat4 r = nglmext::make_rotate(trot);
    return new nmat4triple(r);
}


const nmat4triple* nmat4triple::make_translate( const float x, const float y, const float z)
{
    glm::vec3 tmp(x,y,z);
    return make_translate(tmp);
}
const nmat4triple* nmat4triple::make_scale( const float x, const float y, const float z)
{
    glm::vec3 tmp(x,y,z);
    return make_scale(tmp);
}
const nmat4triple* nmat4triple::make_rotate( const float x, const float y, const float z, const float w)
{
    glm::vec4 tmp(x,y,z,w);
    return make_rotate(tmp);
}





const nmat4triple* nmat4triple::product(const nmat4triple* a, const nmat4triple* b, bool reverse)
{
    std::vector<const nmat4triple*> triples ; 
    triples.push_back(a);
    triples.push_back(b);
    return nmat4triple::product( triples, reverse );
}

const nmat4triple* nmat4triple::product(const nmat4triple* a, const nmat4triple* b, const nmat4triple* c, bool reverse)
{
    std::vector<const nmat4triple*> triples ; 
    triples.push_back(a);
    triples.push_back(b);
    triples.push_back(c);
    return nmat4triple::product( triples, reverse );
}

const nmat4triple* nmat4triple::product(const std::vector<const nmat4triple*>& triples, bool reverse )
{
/*
    Use *reverse=true* when the triples are in reverse heirarchical order, ie when
    they have been collected by starting from the leaf node and then following parent 
    links back up to the root node. 
*/
    unsigned ntriples = triples.size();
    if(ntriples==0) return NULL ; 
    if(ntriples==1) return triples[0] ; 

    glm::mat4 t(1.0) ; 
    glm::mat4 v(1.0) ; 

    for(unsigned i=0,j=ntriples-1 ; i < ntriples ; i++,j-- )
    {
        // inclusive indices:
        //     i: 0 -> ntriples - 1      ascending 
        //     j: ntriples - 1 -> 0      descending (from last transform down to first)
        //
        const nmat4triple* ii = triples[reverse ? j : i] ;  // with reverse: start from the last (ie root node)
        const nmat4triple* jj = triples[reverse ? i : j] ;  // with reverse: start from the first (ie leaf node)

        t *= ii->t ;   
        v *= jj->v ;  // inverse-transform product in opposite order
    }

    // is this the appropriate transform and inverse transform multiplication order ?
    // ... triples order is from the leaf back to the root   

    glm::mat4 q = glm::transpose(v);
    return new nmat4triple(t, v, q) ; 
}


const nmat4triple* nmat4triple::make_identity()
{
    glm::mat4 identity(1.f); 
    return new nmat4triple(identity);
}


const nmat4triple* nmat4triple::make_translated(const glm::vec3& tlate, bool reverse, const char* user ) const 
{
    // reverse:true means the tlate happens at the root 
    // reverse:false means the tlate happens at the leaf
    return make_translated(this, tlate, reverse, user );
}

const nmat4triple* nmat4triple::make_translated(const nmat4triple* src, const glm::vec3& tlate, bool reverse, const char* user)
{ 
    glm::mat4 tra = glm::translate(glm::mat4(1.f), tlate);
    return make_transformed(src, tra, reverse, user );
}

const nmat4triple* nmat4triple::make_transformed(const nmat4triple* src, const glm::mat4& txf, bool reverse, const char* /*user*/)
{
    // reverse:true means the transform ordering is from leaf to root 
    // so when wishing to extend the hierarchy with a higher level root transform, 
    // that means just pushing another transform on the end of the existing vector

/*
    LOG(info) << "nmat4triple::make_transformed" 
              << " user " << user
              ;
*/

    nmat4triple perturb( txf );
    std::vector<const nmat4triple*> triples ; 

    // HMM its confusing to reverse here 
    // because reversal is also done in nmat4triple::product
    // so they will counteract ??
    // Who uses this ?
    
    // used by GMergedMesh::mergeSolidAnalytic/GParts::applyPlacementTransform
    //assert(0);

    if(reverse)
    { 
        triples.push_back(src);    
        triples.push_back(&perturb);
    }
    else
    {
        triples.push_back(&perturb);
        triples.push_back(src);    
    }

    const nmat4triple* transformed = nmat4triple::product( triples, reverse );  
    return transformed ; 
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
        glm::vec4 pl = plan_buffer->getQuad(i);
        nplane*  plane = make_plane(pl);
        glm::vec4 tpl = plane->make_transformed(placement);

        plan_buffer->setQuad( tpl, i ); 
    }    
}


void nmat4triple::dump( const NPY<float>* buf, const char* msg)
{
    LOG(info) << msg ; 
    assert(buf->hasShape(-1,3,4,4));
    unsigned ni = buf->getNumItems();  
    for(unsigned i=0 ; i < ni ; i++)
    {
        nmat4triple* tvq = buf->getMat4TriplePtr(i) ;
        std::cout << std::setw(3) << i << " tvq " << *tvq << std::endl ;  
    }
}


void nmat4triple::dump( const float* data4x4, const char* msg )
{
    LOG(info) << msg ; 
    nmat4triple* tvq = new nmat4triple(data4x4)  ;
    std::cout << " tvq " << *tvq << std::endl ;  
}



std::ostream& operator<< (std::ostream& out, const nmat4pair& pair)
{
    out 
       << std::endl 
       << gpresent( "pair.t",   pair.t ) 
       << std::endl 
       << gpresent( "pair.v", pair.v )
       << std::endl 
       ; 

    return out;
}


std::ostream& operator<< (std::ostream& out, const nmat4triple& triple)
{
    out 
       << std::endl 
       << gpresent( "triple.t",  triple.t ) 
       << std::endl 
       << gpresent( "triple.v",  triple.v ) 
       << std::endl 
       << gpresent( "triple.q",  triple.q ) 
       << std::endl 
       ; 

    return out;
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









