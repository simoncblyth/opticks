#pragma once

/**
NTreeBuilder
=============

cf ../analytic/treebuilder.py 

**/

#include "NPY_API_EXPORT.hh"
#include "OpticksCSG.h"
#include <vector>
#include <string>

template <typename T> struct NNodeAnalyse ; 

template <typename T>
class NPY_API NTreeBuilder 
{
    public:
        static int FindBinaryTreeHeight(unsigned num_leaves); 
        static T* UnionTree(const std::vector<T*>& prims ); 
        std::string desc() const ;
    private:
        static T* CommonTree(const std::vector<T*>& prims, OpticksCSG_t operator_ ); 
    private:
        NTreeBuilder(const std::vector<T*>& prims, OpticksCSG_t operator_ ); 
        void   init();
        void   analyse();
        T*     build_r(int elevation);
        void   populate();
        void   prune();
        void   prune_r(T* node);

        T*     root() const ;
        void   setRoot(T* root); 
    private:
        const std::vector<T*>& m_prims ; 
        std::vector<T*>        m_cprims ; 
        int                    m_height ;
        OpticksCSG_t           m_operator ; 
        std::string            m_optag ; 
        T*                     m_root ; 
        NNodeAnalyse<T>*       m_ana ; 
        int                    m_verbosity ; 


};
 
