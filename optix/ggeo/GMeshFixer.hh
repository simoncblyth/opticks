#pragma once

#include <string>
#include <map>
class GMesh ; 

/*
GMeshFixer
=============

In production it makes more sense to fix meshes earlier 
ie before the cache and even before GGeo gets involved at 
assimp import level. 

BUT during development it is convenient to operate
at a later stage as it is then easy to visualize the meshes.

*/

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GMeshFixer {
    public:
        GMeshFixer(GMesh* src);
        ~GMeshFixer();
        void copyWithoutVertexDuplicates();

        GMesh* getSrc();
        GMesh* getDst();

    private:
        void mapVertices();
        void copyDedupedVertices();

    private:
        GMesh* m_src ; 
        GMesh* m_dst ; 

        int*   m_old2new ; 
        int*   m_new2old ; 

        unsigned int m_num_deduped_vertices ; 

        std::map<std::string, unsigned int> m_vtxmap ;

};

#include "GGEO_TAIL.hh"


