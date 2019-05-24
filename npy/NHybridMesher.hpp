#pragma once

#include <string>

#include "NBBox.hpp"
#include "NOpenMeshType.hpp"

#include "NPY_API_EXPORT.hh"

template <typename T> struct NOpenMesh ; 

#ifdef OLD_PARAMETERS
class X_BParameters ; 
#else
class NMeta ;
#endif

class NTrianglesNPY ; 
struct nnode ; 
struct nbbox ; 

class NPY_API NHybridMesher
{
    public:
         typedef NOpenMesh<NOpenMeshType> MESH ; 
     public:
#ifdef OLD_PARAMETERS
        NHybridMesher(const nnode* node, X_BParameters* meta, const char* treedir=NULL );
#else
        NHybridMesher(const nnode* node, NMeta* meta, const char* treedir=NULL );
#endif
        NTrianglesNPY* operator()();
        std::string desc();
    private:
        void init(); 
    private:
        NOpenMesh<NOpenMeshType>*  m_mesh ;
        nbbox*                     m_bbox ; 
        const char*                m_treedir ; 
  

};
