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
#include "ndeco.hpp"

#include "plog/Severity.h"
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


struct NPY_API nglmext 
{ 
    static const plog::Severity LEVEL ; 

    static constexpr double epsilon = 1e-5 ; 
    static constexpr double epsilon_translation = 1e-3 ; 

    static void GetEyeUVW(
          const glm::vec4& ce, 
          const glm::vec3& eye_m, 
          const glm::vec3& look_m, 
          const glm::vec3& up_m, 
          const unsigned width, 
          const unsigned height, 
          const float tanYfov, // reciprocal of camera zoom  
          glm::vec3& eye, 
          glm::vec3& U, 
          glm::vec3& V, 
          glm::vec3& W, 
          const bool dump 
       );

    static int HandleDegenerateGaze( glm::vec3& up, const glm::vec3& gaze, const float epsilon, const bool dump ) ; 


    static glm::dmat4 upconvert( const glm::mat4& ft );
    static glm::mat4  downconvert( const glm::dmat4& dt );


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

    static float compDiff2(const glm::mat4& a , const glm::mat4& b, bool fractional );
    static float compDiff2(const float a_     , const float b_    , bool fractional, float u_epsilon );

    template<typename T> static T compDiff2_(const glm::tmat4x4<T>& a , const glm::tmat4x4<T>& b, bool fractional );
    template<typename T> static T compDiff2_(const T a_     , const T b_    , bool fractional, T u_epsilon  );


    static glm::mat4 compDiff2_check( const glm::mat4& a_ , const glm::mat4& b_, bool fractional );

    template<typename T>
    static glm::tmat4x4<T> compDiff2_check_(const glm::tmat4x4<T>& a , const glm::tmat4x4<T>& b, bool fractional );


    // maximum absolute componentwise difference between a and b 
    static float compDiff(const glm::mat4& a , const glm::mat4& b );
    static float compDiff(const glm::mat3& a , const glm::mat3& b );
    static float compDiff(const glm::vec4& a , const glm::vec4& b );
    static float compDiff(const glm::vec3& a , const glm::vec3& b );
    static float compDiff(const glm::vec2& a , const glm::vec2& b );

    template<typename T>
    static T compDiff_(const glm::tmat4x4<T>& a , const glm::tmat4x4<T>& b );


    template<typename T>
    static T abs_(T v) ;



    static glm::mat4 average_to_inverse_transpose( const glm::mat4& m );

    template<typename T>
    static glm::tmat4x4<T> average_to_inverse_transpose_( const glm::tmat4x4<T>& m );



    static void polar_decomposition( const glm::mat4& trs, ndeco& deco, bool verbose );

    template<typename T>
    static void polar_decomposition_(const glm::tmat4x4<T>& trs, ndeco_<T>& d,  bool verbose ); 

    static glm::vec3 pluck_scale( const ndeco& d );
    static bool has_scale( const glm::vec3& scale, float epsilon=1e-3 ); 

    static glm::vec3 pluck_translation( const glm::mat4& t ); 
    static bool is_identity(const glm::mat4& t, float eps=1e-5); 


    static glm::mat4 invert_trs( const glm::mat4& trs, bool& match ); 

    template<typename T>
    static glm::tmat4x4<T> invert_trs_( const glm::tmat4x4<T>& trs, bool& match ); 

    static glm::mat4 make_transform(const std::string& order);

    static glm::mat4 make_transform(const std::string& order, const glm::vec3& tlat, const glm::vec4& axis_angle, const glm::vec3& scal );

    template<typename T>
    static glm::tmat4x4<T> make_transform_(const std::string& order, const glm::tvec3<T>& tlat, const glm::tvec4<T>& axis_angle, const glm::tvec3<T>& scal );

    template<typename T>
    static glm::tmat4x4<T> make_transform_(const char* s, char delim=' '); 

    template<typename T>
    static T ato_( const char* a ) ;

    static float angle_radians(float angle_degrees);

    template<typename T> 
    static T angle_radians_(T angle_degrees);

    static glm::mat4 make_translate(const glm::vec3& tlat);
    static glm::mat4 make_rotate(const glm::vec4& axis_angle);
    static glm::mat4 make_scale(const glm::vec3& scal);

    static glm::mat4 make_transpose(const glm::mat4& t );
    static glm::mat4 make_translate(const float x, const float y, const float z);
    static glm::mat4 make_rotate(const float x, const float y, const float z, const float w);
    static glm::mat4 make_scale(const float x, const float y, const float z);


    // used by okc/SphereOfTransforms
    static glm::mat4 make_rotate_a2b(const glm::vec3& a, const glm::vec3& b, bool dump=false );
    static glm::mat4 make_rotate_a2b_then_translate( const glm::vec3& a, const glm::vec3& b, const glm::vec3& tlat, bool dump=false );


    static void transform_planes( NPY<float>* plan_buffer, const glm::mat4& placement );


    static void _define_uv_basis( const glm::vec3& perp, glm::vec3& udir, glm::vec3& vdir   ) ; 
    static void _define_uv_basis( const std::vector<glm::vec4>& perps, std::vector<glm::vec3>& udirs, std::vector<glm::vec3>& vdirs   ); 

    static void _pick_up( glm::vec3& up, const glm::vec3& dir ); // static



};


NPY_API std::ostream& operator<< (std::ostream& out, const nmat4pair& pair); 

NPY_API std::ostream& operator<< (std::ostream& out, const glm::mat3& v);
NPY_API std::ostream& operator<< (std::ostream& out, const glm::mat4& v); 
NPY_API std::ostream& operator<< (std::ostream& out, const glm::vec4& v); 
NPY_API std::ostream& operator<< (std::ostream& out, const glm::vec3& v);
NPY_API std::ostream& operator<< (std::ostream& out, const glm::ivec3& v);
 

