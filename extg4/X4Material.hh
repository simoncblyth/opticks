#pragma once

#include <string>
#include "X4_API_EXPORT.hh"

class G4Material ; 

template <typename T> class NPY ;

/**
X4Material
===========

**/

class X4_API X4Material
{
    public:
        X4Material(const G4Material* material); 
        std::string desc() const  ; 
    private:
        void init();
    private:
        const G4Material* m_material ;  

};

