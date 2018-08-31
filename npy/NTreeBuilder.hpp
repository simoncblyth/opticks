#pragma once

#include "NPY_API_EXPORT.hh"
#include "OpticksCSG.h"
#include "plog/Severity.h"
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
        static const plog::Severity LEVEL ; 
        typedef enum {  PRIM, BILEAF, MIXED } NTreeBuilderMode_t ; 
        static const char* PRIM_ ; 
        static const char* BILEAF_ ; 
        static const char* MIXED_ ; 
        static const char* BuilderMode( NTreeBuilderMode_t mode ); 
    public:
        static int FindBinaryTreeHeight(unsigned num_leaves); 
        static T* UnionTree(const std::vector<T*>& prims ); 
        static T* CommonTree(const std::vector<T*>& prims, OpticksCSG_t operator_ ); 
        static T* BileafTree(const std::vector<T*>& bileafs, OpticksCSG_t operator_ ); 
        static T* MixedTree(const std::vector<T*>& bileafs, const std::vector<T*>& others, OpticksCSG_t operator_ ); 
        std::string desc() const ;
    private:
    private:
        NTreeBuilder(const std::vector<T*>& subs, const std::vector<T*>& others, OpticksCSG_t operator_, NTreeBuilderMode_t mode=PRIM ); 
        void   init();
        unsigned getNumPrim() const ;

        T*     build_r(int elevation);
        void   populate(std::vector<T*>& src);
        void   prune();
        void   prune_r(T* node);

        T*     root() const ;
        void   setRoot(T* root); 
    private:
        const std::vector<T*>& m_subs ; 
        const std::vector<T*>& m_otherprim ; 
        std::vector<T*>        m_subs_copy ; 
        std::vector<T*>        m_otherprim_copy ; 

        OpticksCSG_t           m_operator ; 
        std::string            m_optag ; 
        NTreeBuilderMode_t     m_mode ; 

        unsigned               m_num_prim ;
        int                    m_height ;
        T*                     m_root ; 
        int                    m_verbosity ; 


};
 
