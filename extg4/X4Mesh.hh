#pragma once

#include "X4_API_EXPORT.hh"

class G4VSolid ; 
class G4Polyhedron ; 

template <typename T> class NPY ;

/**
X4Mesh
========

TODO: give birth to GMesh instances

**/

class X4_API X4Mesh
{
    public:
        X4Mesh(const G4VSolid* solid); 
        std::string desc() const  ; 
    private:
        void init();
        void polygonize();
        void collect();
    private:
        void collect_vtx(int ivert);
        void collect_raw(int iface);
        void collect_tri();
    private:
        const G4VSolid* m_solid ;  
        G4Polyhedron*   m_polyhedron ;
        NPY<float>*     m_vtx ; 
        NPY<unsigned>*  m_raw ; // tris or quads
        NPY<unsigned>*  m_tri ; // by splitting quads  
        int             m_verbosity ; 

};

