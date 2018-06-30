#pragma once

/**
NTreePositive
=============

cf ../analytic/csg.py 

**/

#include "NPY_API_EXPORT.hh"
#include "OpticksCSG.h"
#include <vector>
#include <string>


template <typename T>
class NPY_API NTreePositive
{
    public:
        std::string desc() const ;
        NTreePositive(T* root); 
        T*    root() const ;
    private:
        void  init() ; 
        static void positivize_r(T* node, bool negate, unsigned depth);
        static int  fVerbosity ; 
    private:
        T*                     m_root ; 


};

