#pragma once

#include "NOpenMeshType.hpp"

template <typename T> struct  NOpenMeshProp ;

template <typename T>
struct NPY_API  NOpenMeshDesc
{
    typedef typename T::EdgeHandle          EH ; 
    typedef typename T::HalfedgeHandle      HEH ; 
    typedef typename T::VertexHandle        VH ; 
    typedef typename T::FaceHandle          FH ; 


    NOpenMeshDesc( const T& mesh, const NOpenMeshProp<T>& prop );

    std::string operator()(const std::vector<typename T::HalfedgeHandle> loop, unsigned mx=10u) const ;
    std::string operator()(const typename T::FaceHandle fh) const ;
    std::string operator()(const typename T::HalfedgeHandle heh) const ;
    std::string operator()(const typename T::VertexHandle vh) const ;
    std::string operator()(const typename T::EdgeHandle vh) const ;
    std::string operator()(const typename T::Point& pt) const ;

    static std::string desc_point(const typename T::Point& pt, int w, int p) ;

    std::string desc() const ;
    std::string desc_euler() const ; 
    int euler_characteristic() const ;


    std::string vertices() const ;
    std::string faces() const ;
    std::string edges() const ;

    void dump_faces(const char* msg="NOpenMeshDesc::dump_faces") const ;
    void dump_vertices(const char* msg="NOpenMeshDesc::dump_vertices") const  ;


    const T& mesh  ;
    const NOpenMeshProp<T>& prop ;

};
 


