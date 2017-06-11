#pragma once

#include <string>

#include "NBBox.hpp"
#include "NOpenMeshType.hpp"

#include "NPY_API_EXPORT.hh"

template <typename T> struct NOpenMesh ; 

class NParameters ; 
class NTrianglesNPY ; 
struct nnode ; 
struct nbbox ; 

class NPY_API NHybridMesher
{
    public:
         typedef NOpenMesh<NOpenMeshType> MESH ; 
     public:
        NHybridMesher(const nnode* node, NParameters* meta, const char* treedir=NULL );
        NTrianglesNPY* operator()();
        std::string desc();
    private:
        void init(); 
    private:
        NOpenMesh<NOpenMeshType>*  m_mesh ;
        nbbox*                     m_bbox ; 
        const char*                m_treedir ; 
  

};
