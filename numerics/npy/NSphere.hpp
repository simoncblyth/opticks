#pragma once

#include <glm/glm.hpp>
template <typename T> class NPY ;

/*

 * attempt to apply subdiv and spherical projection  
   to get tesselated partial (z-sliced) spheres is hindered by 
   the need to come up with basis polyhedra

 * suggests lat/lon tesselation may be easiest route

*/

// hmm maybe eliminate this class, just passing thru to NTrianglesNPY ? 

class NSphere {
    public:
        static NPY<float>* icosahedron(unsigned int nsubdiv) ; 
        static NPY<float>* octahedron(unsigned int nsubdiv) ; 
        static NPY<float>* hemi_octahedron(unsigned int nsubdiv) ; 
        static NPY<float>* hemi_octahedron(unsigned int nsubdiv, glm::mat4& m) ; 
        static NPY<float>* cube(unsigned int nsubdiv) ; 
        static NPY<float>* latlon(unsigned int npolar=24, unsigned int nazimuthal=24) ; 
        static NPY<float>* latlon(float zmin, float zmax, unsigned int npolar=24, unsigned int nazimuthal=24) ;
        // z in polar direction, (theta,z) (0,+1) (pi,-1)

};



