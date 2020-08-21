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

#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

// trying to fwd declare leads to linker errors for static NPY methods with some tests : G4StepNPYTest.cc, HitsNPYTest.cc see tests/CMakeLists.txt
//template <typename T> class NPY ; 
#include "NPY.hpp"

#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"


template<typename T>
void sincos_(const T angle, T& s, T& c)
{
#ifdef __APPLE__
    __sincos( angle, &s, &c);
#elif __linux
    sincos( angle, &s, &c);
#else
    s = sin(angle);
    c = cos(angle) ;
#endif

}


struct NPY_API nmat4pair 
{
    static nmat4pair* product(const std::vector<nmat4pair*>& tt);

    nmat4pair* clone();
    nmat4pair( const glm::mat4& transform ); 
    nmat4pair( const glm::mat4& transform, const glm::mat4& inverse ) : t(transform), v(inverse) {} ;
    std::string digest();

    glm::mat4 t ; 
    glm::mat4 v ; 
};


struct NPY_API nmat4triple
{
    static const nmat4triple* make_translate(const glm::vec3& tlate);
    static const nmat4triple* make_rotate(   const glm::vec4& trot);
    static const nmat4triple* make_scale(    const glm::vec3& tsca);

    static const nmat4triple* make_transform( 
           const float x0, const float y0, const float z0, const float w0,
           const float x1, const float y1, const float z1, const float w1, 
           const float x2, const float y2, const float z2, const float w2, 
           const float x3, const float y3, const float z3, const float w3 
       );

    static const nmat4triple* make_translate( const float x, const float y, const float z);
    static const nmat4triple* make_rotate(    const float x, const float y, const float z, const float w);
    static const nmat4triple* make_scale(     const float x, const float y, const float z);

    static const nmat4triple* product(const nmat4triple* a, const nmat4triple* b, bool reverse);
    static const nmat4triple* product(const nmat4triple* a, const nmat4triple* b, const nmat4triple* c, bool reverse);
    static const nmat4triple* product(const std::vector<const nmat4triple*>& tt, bool reverse );
    static const nmat4triple* make_translated(const nmat4triple* src, const glm::vec3& tlate, bool reverse, const char* user  );
    static const nmat4triple* make_transformed(const nmat4triple* src, const glm::mat4& txf, bool reverse, const char* user);
    static const nmat4triple* make_identity();
    static void dump( const NPY<float>* buf, const char* msg="nmat4triple::dump");
    static void dump( const float* data4x4, const char* msg="nmat4triple::dump");

    const nmat4triple* clone() const ;
    const nmat4triple* make_translated(const glm::vec3& tlate, bool reverse, const char* user ) const ;

    nmat4triple( const glm::mat4& transform ); 
    nmat4triple( const float* data ); 
    nmat4triple( const glm::mat4& transform, const glm::mat4& inverse, const glm::mat4& inverse_T ) 
         : 
            t(transform), 
            v(inverse), 
            q(inverse_T) 
         {} ;

    std::string digest() const ;

    glm::vec3 apply_transform_t( const glm::vec3& p, const float w=1.0f ) const ;
    glm::vec3 apply_transform_v( const glm::vec3& p, const float w=1.0f ) const ;

    void apply_transform_t(std::vector<glm::vec3>& dst, const std::vector<glm::vec3>& src, float w=1.0f) const ;
    void apply_transform_v(std::vector<glm::vec3>& dst, const std::vector<glm::vec3>& src, float w=1.0f) const ;

    glm::vec3 get_translation() const ; 
    bool is_translation_only(float epsilon=1e-5) const ; 
    bool is_identity(float epsilon=1e-5) const ; 
    bool is_equal_to(const nmat4triple* other, float epsilon=1e-5) const ; 


    glm::mat4 t ; 
    glm::mat4 v ; 
    glm::mat4 q ; 
};



struct NPY_API ntransformer
{
    ntransformer( const glm::mat4& t_, const float w_ ) : t(t_),w(w_) {} ;

    glm::vec3 operator()(const glm::vec3& p_) const 
    {
        glm::vec4 p(p_, w) ; 
        p = t * p ; 
        return glm::vec3(p);
    }

    const glm::mat4 t ; 
    const float w ; 
};



struct NPY_API ndeco
{  
    glm::mat4 t ; 
    glm::mat4 r ; 
    glm::mat4 s ; 

    glm::mat4 it ; 
    glm::mat4 ir ; 
    glm::mat4 is ; 

    glm::mat4 rs ;

    glm::mat4 tr ;
    glm::mat4 trs ;
    glm::mat4 isirit ;
};


struct NPY_API nglmext 
{ 
    static void GetEyeUVW(
          const glm::vec4& ce, 
          const glm::vec3& eye_m, 
          const glm::vec3& look_m, 
          const glm::vec3& up_m, 
          const unsigned width, 
          const unsigned height, 
          glm::vec3& eye, 
          glm::vec3& U, 
          glm::vec3& V, 
          glm::vec3& W );

    static glm::mat4 make_yzflip() ;
    static glm::mat4 make_flip(unsigned axa, unsigned axb) ;

    static void copyTransform( std::array<float,16>& dst, const glm::mat4& src );
    static std::string xform_string( const std::array<float, 16>& xform );
    static std::array<float, 16> _float4x4_mul( const std::array<float, 16>& a, const std::array<float, 16>& b) ;
    static const std::array<float, 16> _identity_float4x4 ;

    static glm::vec3 least_parallel_axis( const glm::vec3& dir );
    static glm::vec3 pick_transverse_direction( const glm::vec3& dir, bool dump=false) ;

    static glm::mat4 invert_tr( const glm::mat4& tr ); 
    static glm::mat4* invert_tr( const glm::mat4* tr ); 

    static float compDiff2(const glm::mat4& a , const glm::mat4& b, bool fractional=false, float epsilon=1e-5, float epsilon_translation=1e-3);
    static float compDiff2(const float a_     , const float b_    , bool fractional=false, float epsilon=1e-5);

    // maximum absolute componentwise difference between a and b 
    static float compDiff(const glm::mat4& a , const glm::mat4& b );
    static float compDiff(const glm::mat3& a , const glm::mat3& b );
    static float compDiff(const glm::vec4& a , const glm::vec4& b );
    static float compDiff(const glm::vec3& a , const glm::vec3& b );
    static float compDiff(const glm::vec2& a , const glm::vec2& b );

    static glm::mat4 average_to_inverse_transpose( const glm::mat4& m );
    static void polar_decomposition( const glm::mat4& trs, ndeco& deco, bool verbose=false );
    static glm::vec3 pluck_scale( const ndeco& d );
    static bool has_scale( const glm::vec3& scale, float epsilon=1e-3 ); 

    static glm::vec3 pluck_translation( const glm::mat4& t ); 
    static bool is_identity(const glm::mat4& t, float eps=1e-5); 


    static glm::mat4 invert_trs( const glm::mat4& trs ); 
    static glm::mat4 make_transform(const std::string& order, const glm::vec3& tlat, const glm::vec4& axis_angle, const glm::vec3& scal );
    static glm::mat4 make_transform(const std::string& order);

    static float angle_radians(float angle_degrees);
    static glm::mat4 make_translate(const glm::vec3& tlat);
    static glm::mat4 make_rotate(const glm::vec4& axis_angle);
    static glm::mat4 make_scale(const glm::vec3& scal);

    static glm::mat4 make_transpose(const glm::mat4& t );
    static glm::mat4 make_translate(const float x, const float y, const float z);
    static glm::mat4 make_rotate(const float x, const float y, const float z, const float w);
    static glm::mat4 make_scale(const float x, const float y, const float z);


    static void transform_planes( NPY<float>* plan_buffer, const glm::mat4& placement );


    static void _define_uv_basis( const glm::vec3& perp, glm::vec3& udir, glm::vec3& vdir   ) ; 
    static void _define_uv_basis( const std::vector<glm::vec4>& perps, std::vector<glm::vec3>& udirs, std::vector<glm::vec3>& vdirs   ); 

    static void _pick_up( glm::vec3& up, const glm::vec3& dir ); // static



};


NPY_API std::ostream& operator<< (std::ostream& out, const nmat4triple& triple); 
NPY_API std::ostream& operator<< (std::ostream& out, const nmat4pair& pair); 

NPY_API std::ostream& operator<< (std::ostream& out, const glm::mat3& v);
NPY_API std::ostream& operator<< (std::ostream& out, const glm::mat4& v); 
NPY_API std::ostream& operator<< (std::ostream& out, const glm::vec4& v); 
NPY_API std::ostream& operator<< (std::ostream& out, const glm::vec3& v);
NPY_API std::ostream& operator<< (std::ostream& out, const glm::ivec3& v);
 

