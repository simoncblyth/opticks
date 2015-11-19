#pragma once
template <typename T> class NPY ;

/*

 * attempt to apply subdiv and spherical projection  
   to get tesselated partial (z-sliced) spheres is hindered by 
   the need to come up with basis polyhedra

 * suggests lat/lon tesselation may be easiest route

*/

class NSphere {
    public:
        static NPY<float>* icosahedron(unsigned int nsubdiv) ; 
        static NPY<float>* octahedron(unsigned int nsubdiv) ; 
        static NPY<float>* hemi_octahedron(unsigned int nsubdiv) ; 
        static NPY<float>* cube(unsigned int nsubdiv) ; 
        static NPY<float>* latlon(unsigned int npolar=24, unsigned int nazimuthal=24) ; 

};



