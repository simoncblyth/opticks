#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

template <typename T> class NPY ; 

#include "plog/Severity.h"
#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"

template<typename T>
struct NPY_API nmat4triple_
{
    static const plog::Severity LEVEL ; 

    static const nmat4triple_<T>* make_translate(const glm::tvec3<T>& tlate);
    static const nmat4triple_<T>* make_rotate(   const glm::tvec4<T>& trot);
    static const nmat4triple_<T>* make_scale(    const glm::tvec3<T>& tsca);

    static const nmat4triple_<T>* make_transform( 
           const T x0, const T y0, const T z0, const T w0,
           const T x1, const T y1, const T z1, const T w1, 
           const T x2, const T y2, const T z2, const T w2, 
           const T x3, const T y3, const T z3, const T w3 
       );

    static const nmat4triple_<T>* make_translate( const T x, const T y, const T z);
    static const nmat4triple_<T>* make_rotate(    const T x, const T y, const T z, const T w);
    static const nmat4triple_<T>* make_scale(     const T x, const T y, const T z);

    static const nmat4triple_<T>* product(const nmat4triple_<T>* a, const nmat4triple_<T>* b, bool reverse);
    static const nmat4triple_<T>* product(const nmat4triple_<T>* a, const nmat4triple_<T>* b, const nmat4triple_<T>* c, bool reverse);
    static const nmat4triple_<T>* product(const std::vector<const nmat4triple_<T>*>& tt, bool reverse );
    static const nmat4triple_<T>* make_translated(const nmat4triple_<T>* src, const glm::tvec3<T>& tlate, bool reverse, const char* user, bool& match  );
    static const nmat4triple_<T>* make_transformed(const nmat4triple_<T>* src, const glm::tmat4x4<T>& txf, bool reverse, const char* user, bool& match );
    static const nmat4triple_<T>* make_identity();
    static void dump( const NPY<T>* buf, const char* msg="nmat4triple_::dump");
    static void dump( const T* data4x4, const char* msg="nmat4triple_::dump");

    const nmat4triple_<T>* clone() const ;
    const nmat4triple_<T>* make_translated(const glm::tvec3<T>& tlate, bool reverse, const char* user, bool& match ) const ;

    nmat4triple_( const glm::tmat4x4<T>& transform ); 
    nmat4triple_( const T* data ); 
    nmat4triple_( const glm::tmat4x4<T>& transform, const glm::tmat4x4<T>& inverse, const glm::tmat4x4<T>& inverse_T ) ;


    std::string digest() const ;

    glm::tvec3<T> apply_transform_t( const glm::tvec3<T>& p, const T w=T(1) ) const ;
    glm::tvec3<T> apply_transform_v( const glm::tvec3<T>& p, const T w=T(1) ) const ;

    void apply_transform_t(std::vector<glm::tvec3<T> >& dst, const std::vector<glm::tvec3<T> >& src, T w=T(1)) const ;
    void apply_transform_v(std::vector<glm::tvec3<T> >& dst, const std::vector<glm::tvec3<T> >& src, T w=T(1)) const ;

    glm::tvec3<T> get_translation() const ; 
    bool is_translation_only(T epsilon=1e-5) const ; 
    bool is_identity(T epsilon=1e-5) const ; 
    bool is_equal_to(const nmat4triple_<T>* other, T epsilon=1e-5) const ; 

    bool match ; 
    glm::tmat4x4<T> t ; 
    glm::tmat4x4<T> v ; 
    glm::tmat4x4<T> q ; 
};


template<typename T>
NPY_API std::ostream& operator<< (std::ostream& out, const nmat4triple_<T>& triple); 



