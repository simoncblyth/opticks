#pragma once

#include <functional>
#include <vector>
#include <string>

#include "NBBox.hpp"
#include "NOpenMeshType.hpp"

#include "NPY_API_EXPORT.hh"


template <typename T> struct NOpenMesh ; 

class Timer ; 
class NTrianglesNPY ; 
struct nnode ; 
struct nbbox ; 

class NPY_API NHybridMesher
{
    public:
        static NOpenMesh<NOpenMeshType>* make_mesh( const nnode* node, int level , int verbosity, int ctrl );
    public:
        typedef std::function<float(float,float,float)> FUNC ; 
     public:
        NHybridMesher(const nnode* node, int level=5, int verbosity=1, int ctrl=0);
        NTrianglesNPY* operator()();
        std::string desc();
    private:
        void init(); 
    private:
        Timer*                     m_timer ; 
        NOpenMesh<NOpenMeshType>*  m_mesh ;
        nbbox*                     m_bbox ; 
        int                        m_verbosity ; 
        int                        m_ctrl ; 

};
