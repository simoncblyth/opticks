#pragma once

#include "GGEO_API_EXPORT.hh"

class GMesh ; 
template <typename T> class NPY ; 


class GGEO_API GMeshMaker 
{
    public:
        static GMesh* make_spherelocal_mesh(NPY<float>* triangles, unsigned int meshindex=0);  
        static GMesh* make_mesh(NPY<float>* triangles, unsigned int meshindex=0);

        // this one tries to avoid vertex duplication, but may have problems with normals
        // as a result, the GMesh assumption of same numbers of normals as verts aint so good  
        static GMesh* make_mesh(NPY<float>* vtx3, NPY<unsigned>* tri3, unsigned int meshindex=0);


};
 
