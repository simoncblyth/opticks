#pragma once

#include <vector>
#include <map>

struct NTriSource ; 
struct nnode ; 

#include "NOpenMeshType.hpp"

template <typename T> struct NOpenMesh ;
template <typename T> struct NOpenMeshProp ;
template <typename T> struct NOpenMeshDesc ;
template <typename T> struct NOpenMeshFind ;

template <typename T>
struct NPY_API  NOpenMeshBuild
{
    typedef typename T::VertexHandle VH ; 
    typedef typename T::FaceHandle   FH ; 
    typedef typename T::Point         P ; 

    NOpenMeshBuild( T& mesh, 
                    NOpenMeshProp<T>& prop, 
                    const NOpenMeshDesc<T>& desc, 
                    const NOpenMeshFind<T>& find,
                    int verbosity
                  );

    VH add_vertex_unique(typename T::Point pt, bool& added, const float epsilon) ;  

    void add_face_(VH v0, VH v1, VH v2, VH v3 );
    FH   add_face_(VH v0, VH v1, VH v2, int identity=-1 );
    bool is_consistent_face_winding(VH v0, VH v1, VH v2);

    void add_parametric_primitive(const nnode* node, int level, int ctrl, float epsilon )  ;
    void euler_check(const nnode* node, int level );

    void copy_faces(const NOpenMesh<T>* other, int facemask, float epsilon );
    void copy_faces(const NTriSource*   other, float epsilon  );

    void mark_faces(const nnode* other);
    void mark_face(FH fh, const nnode* other);
    std::string desc_facemask();


    void add_hexpatch(bool inner_only);
    void add_tetrahedron();
    void add_cube();


    T& mesh  ;

    NOpenMeshProp<T>& prop ;
    const NOpenMeshDesc<T>& desc ;
    const NOpenMeshFind<T>& find ;
    int verbosity ; 

    std::map<int,int> f_inside_other_count ; 

};


 


