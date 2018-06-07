#pragma once

#include "X4_API_EXPORT.hh"

class G4VSolid ; 
class G4Sphere ; 
class G4Polyhedron ; 

template <typename T> class NPY ;

/**
X4Solid
==========

hmm: would be better for this to give birth to instances
     without any G4 dependency (eg GGeo classes)


**/

class X4_API X4Solid
{
    public:
        static X4Solid* Create(const G4Sphere* sphere); 
    public:
        X4Solid(const G4VSolid* solid); 
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

};

