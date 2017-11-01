#pragma once

#include <string>
#include <glm/fwd.hpp>

template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"

/**
NPho
======

Wrapper for an (n,4,4) photon buffer providing higher level
accessors and dumping.

**/

class NPY_API NPho {
    public:  
        NPho(NPY<float>* photons); 
    private:
        void init();   
    public:  
        NPY<float>*           getPhotons() const ;
        unsigned              getNumPhotons() const ;
    public:  
        glm::vec4             getPositionTime(unsigned i) const ; 
        glm::vec4             getDirectionWeight(unsigned i) const ; 
        glm::vec4             getPolarizationWavelength(unsigned i) const ;
        glm::uvec4            getFlags(unsigned i) const ;
        std::string           desc(unsigned i) const ;
        std::string           desc() const ;
        void                  dump(unsigned modulo, unsigned margin, const char* msg="NPho::dump") const ;
   private:
       NPY<float>*            m_photons ; 
       unsigned               m_num_photons ; 

};

