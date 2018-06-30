#pragma once

#include "NPY_API_EXPORT.hh"
#include "OpticksCSG.h"
#include <vector>
#include <string>

/**
NTreeBuilder
=============

Ported from ../analytic/treebuilder.py 

Used by:

1. extg4/X4Solid.cc to populate a tree with polycone primitives
2. npy/NTreeBalance.cpp to reduce analytic geometry part counts

**/

template <typename T>
class NPY_API NTreeBuilder 
{
    public:
        static int FindBinaryTreeHeight(unsigned num_leaves); 
        static T* UnionTree(const std::vector<T*>& prims ); 
        static T* CommonTree(const std::vector<T*>& prims, OpticksCSG_t operator_ ); 
        static T* BileafTree(const std::vector<T*>& bileafs, OpticksCSG_t operator_ ); 
        std::string desc() const ;
    private:
    private:
        NTreeBuilder(const std::vector<T*>& subs, OpticksCSG_t operator_, bool bileaf=false ); 
        void   init();
        T*     build_r(int elevation);
        void   populate();
        void   prune();
        void   prune_r(T* node);

        T*     root() const ;
        void   setRoot(T* root); 
    private:
        const std::vector<T*>& m_subs ; 
        std::vector<T*>        m_csubs ; 
        OpticksCSG_t           m_operator ; 
        std::string            m_optag ; 
        bool                   m_bileaf ; 

        int                    m_height ;
        T*                     m_root ; 
        int                    m_verbosity ; 


};
 
