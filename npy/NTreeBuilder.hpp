#pragma once

/**
NTreeBuilder
=============

cf ../analytic/treebuilder.py 

**/

#include "NPY_API_EXPORT.hh"
#include "OpticksCSG.h"
#include <vector>
struct nnode ; 

class NPY_API NTreeBuilder 
{
    public:
        static int FindBinaryTreeHeight(unsigned num_leaves); 
        static nnode* UnionTree(const std::vector<nnode*>& prims ); 
        std::string desc() const ;
    private:
        static nnode* CommonTree(const std::vector<nnode*>& prims, OpticksCSG_t operator_ ); 
    private:
        NTreeBuilder(const std::vector<nnode*>& prims, OpticksCSG_t operator_ ); 
        void   init();
        nnode* build(int height);
        nnode* build_r(int elevation);
        void   populate();
        void   prune();
        nnode* root(); 
    private:
        const std::vector<nnode*>& m_prims ; 
        std::vector<nnode*>        m_cprims ; 
        int                        m_height ;
        OpticksCSG_t               m_operator ; 
        OpticksCSG_t               m_placeholder ; 
        nnode*                     m_root ; 

};
 
