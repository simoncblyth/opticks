#pragma once

#include <string>
#include <glm/fwd.hpp>

template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"

/**
NPri
======

Wrapper for an (n,4,4) primary particle buffer providing higher level
accessors and dumping.

**/

class NPY_API NPri {
    public:  
        static void Dump(NPY<float>* ox, unsigned modulo, unsigned margin, const char* msg) ;
    public:  
        unsigned getNumG4Event() const ;
        NPri(const NPY<float>* primaries); 
    private:
        void init();   
    public:  
        const NPY<float>*     getPrimaries() const ;
        unsigned              getNumPri() const ;
    public:  
        glm::vec4             getPositionTime(unsigned i) const ; 
        glm::vec4             getDirectionWeight(unsigned i) const ; 
        glm::vec4             getPolarizationKineticEnergy(unsigned i) const ;
        glm::ivec4            getFlags(unsigned i) const ;

        int                   getEventIndex(unsigned i) const ; 
        int                   getVertexIndex(unsigned i) const ; 
        int                   getParticleIndex(unsigned i) const ;  // within the vertex 
        int                   getPDGCode(unsigned i) const ; 

        std::string           desc(unsigned i) const ;
        std::string           desc() const ;
        void                  dump(unsigned modulo, unsigned margin, const char* msg="NPri::dump") const ;
   private:
        const NPY<float>*     m_primaries ; 

};

