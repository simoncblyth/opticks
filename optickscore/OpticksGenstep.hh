#pragma once

#include <string>
#include <glm/fwd.hpp>

template <typename T> class NPY ; 
#include "OKCORE_API_EXPORT.hh"

/**
OpticksGenstep
==================

Wrapper for an (n,6,4) NPY<float> genstep buffer providing higher level
accessors and dumping. Only generic functionality working for all 
types of gensteps is provided.  Types of gensteps::

    Cerenkov
    Scintillation
    Natural (mix of Cernenkov and Scintillaton)
    Torch (fabricated)

Recall that different types of gensteps can be mixed together
in the same buffer, being identified by an identifier in hdr.
The first three quads are common to all types of gensteps, the 
last three have different meanings depending on the
type of genstep.

TODO: incorporate npy.G4StepNPY ?


**/

class OKCORE_API OpticksGenstep {
    public:  
        static void Dump(const NPY<float>* gs, unsigned modulo, unsigned margin, const char* msg) ;
    public:  
        OpticksGenstep(const NPY<float>* gs); 
    private:
        void init();   
    public:  
        const NPY<float>*     getGensteps() const ;
        int                   getContentVersion() const ;
        unsigned              getNumGensteps() const ;
        unsigned              getNumPhotons() const ;
        float                 getAvgPhotonsPerGenstep() const ;
    public:  
        unsigned              getGencode(unsigned idx) const ; 
    public:  
        glm::ivec4            getHdr(unsigned i) const ;
        glm::vec4             getPositionTime(unsigned i) const ; 
        glm::vec4             getDeltaPositionStepLength(unsigned i) const ; 

        glm::vec4             getQ3(unsigned i) const ; 
        glm::vec4             getQ4(unsigned i) const ; 
        glm::vec4             getQ5(unsigned i) const ; 

        glm::ivec4            getI3(unsigned i) const ; 
        glm::ivec4            getI4(unsigned i) const ; 
        glm::ivec4            getI5(unsigned i) const ; 
    public:  
        std::string           desc(unsigned i) const ;
        std::string           desc() const ;
        void                  dump(unsigned modulo, unsigned margin, const char* msg="OpticksGenstep::dump") const ;
    private:
        const NPY<float>*     m_gs ; 

};

