#pragma once

#include "NPY_API_EXPORT.hh"
#include <vector>

/**
NTreeBalance
==============

NB this is **NOT A GENERAL TREE BALANCER** it does 
however succeed to balance trees that Geant4 
boolean solids often result in.

Ported from ../analytic/csg.py ../analytic/treebuilder.py

**/

template <typename T>
struct NPY_API NTreeBalance
{
    NTreeBalance(T* root_); 

    T* create_balanced(); 

    void init(); 
    static unsigned depth_r(T* node, unsigned depth, bool label);
    static void     subdepth_r(T* node, unsigned depth );

    unsigned operators(unsigned minsubdepth=0) const ;
    std::string operatorsDesc(unsigned minsubdepth=0) const ;

    static void operators_r(const T* node, unsigned& mask, unsigned minsubdepth );

    void subtrees(std::vector<T*>& subs, unsigned subdepth );
    static void subtrees_r(T* node, std::vector<T*>& subs, unsigned subdepth );

    bool is_positive_form() const ;  
    bool is_mono_form()     const ;  

    T*           root ; 
    unsigned     height0 ; 

};



