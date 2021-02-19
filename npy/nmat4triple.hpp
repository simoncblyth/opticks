#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include "NPY.hpp"

#include "plog/Severity.h"
#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"


struct NPY_API nmat4triple
{
    static const plog::Severity LEVEL ; 

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
    static const nmat4triple* make_translated(const nmat4triple* src, const glm::vec3& tlate, bool reverse, const char* user, bool& match  );
    static const nmat4triple* make_transformed(const nmat4triple* src, const glm::mat4& txf, bool reverse, const char* user, bool& match );
    static const nmat4triple* make_identity();
    static void dump( const NPY<float>* buf, const char* msg="nmat4triple::dump");
    static void dump( const float* data4x4, const char* msg="nmat4triple::dump");

    const nmat4triple* clone() const ;
    const nmat4triple* make_translated(const glm::vec3& tlate, bool reverse, const char* user, bool& match ) const ;

    nmat4triple( const glm::mat4& transform ); 
    nmat4triple( const float* data ); 
    nmat4triple( const glm::mat4& transform, const glm::mat4& inverse, const glm::mat4& inverse_T ) ;


    std::string digest() const ;

    glm::vec3 apply_transform_t( const glm::vec3& p, const float w=1.0f ) const ;
    glm::vec3 apply_transform_v( const glm::vec3& p, const float w=1.0f ) const ;

    void apply_transform_t(std::vector<glm::vec3>& dst, const std::vector<glm::vec3>& src, float w=1.0f) const ;
    void apply_transform_v(std::vector<glm::vec3>& dst, const std::vector<glm::vec3>& src, float w=1.0f) const ;

    glm::vec3 get_translation() const ; 
    bool is_translation_only(float epsilon=1e-5) const ; 
    bool is_identity(float epsilon=1e-5) const ; 
    bool is_equal_to(const nmat4triple* other, float epsilon=1e-5) const ; 

    bool match ; 
    glm::mat4 t ; 
    glm::mat4 v ; 
    glm::mat4 q ; 
};

NPY_API std::ostream& operator<< (std::ostream& out, const nmat4triple& triple); 

