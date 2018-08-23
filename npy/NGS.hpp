#pragma once

#include <string>
#include <glm/fwd.hpp>

template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"

/**
NGS
======

Wrapper for an (n,6,4) genstep buffer providing higher level
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

**/

class NPY_API NGS {
    public:  
        static void Dump(NPY<float>* gs, unsigned modulo, unsigned margin, const char* msg) ;
    public:  
        NGS(NPY<float>* gs); 
    private:
        void init();   
    public:  
        NPY<float>*           getGensteps() const ;
        unsigned              getNumGensteps() const ;
    public:  
        glm::ivec4            getHdr(unsigned i) const ;
        glm::vec4             getPositionTime(unsigned i) const ; 
        glm::vec4             getDeltaPositionStepLength(unsigned i) const ; 
        glm::vec4             getQ3(unsigned i) const ; 
        glm::vec4             getQ4(unsigned i) const ; 
        glm::vec4             getQ5(unsigned i) const ; 

        std::string           desc(unsigned i) const ;
        std::string           desc() const ;
        void                  dump(unsigned modulo, unsigned margin, const char* msg="NGS::dump") const ;
   private:
       NPY<float>*            m_gensteps ; 

};

