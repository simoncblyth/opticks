#pragma once

#include "GGEO_API_EXPORT.hh"

class GMesh ; 
template <typename T> class NPY ; 

struct nbbox ; 


class GGEO_API GMeshMaker 
{
    public:
        static GMesh* Make( nbbox& bb ) ;
        static GMesh* MakeSphereLocal(NPY<float>* triangles, unsigned meshindex=0);  
        static GMesh* Make(NPY<float>* triangles, unsigned meshindex=0);

        // this one tries to avoid vertex duplication, but may have problems with normals
        // as a result, the GMesh assumption of same numbers of normals as verts aint so good  
        static GMesh* Make(NPY<float>* vtx3, NPY<unsigned>* tri3, unsigned meshindex=0);


};
 
