#pragma once

#include <map>

class GGeo ; 
class GCache ; 
class GMergedMesh ; 
class GNode ; 

class GGeoLib {
    public:
        static const char* GMERGEDMESH ; 
        enum { MAX_MERGED_MESH = 10 } ;
    public:
        static GGeoLib* load(GCache* cache);
    public:
        GGeoLib(GCache* cache);
        void setMeshVersion(const char* mesh_version);
        const char* getMeshVersion();
    public:
        void loadFromCache();
        void saveToCache();
        GMergedMesh* makeMergedMesh(GGeo* ggeo, unsigned int index=0, GNode* base=NULL);
    private:
        void loadMergedMeshes(const char* idpath);
        void removeMergedMeshes(const char* idpath);
        void saveMergedMeshes(const char* idpath);
    public:
        unsigned int getNumMergedMesh();
        GMergedMesh* getMergedMesh(unsigned int index);
        void setMergedMesh(unsigned int index, GMergedMesh* mm);
        void eraseMergedMesh(unsigned int index);
        void clear();
    private:
        GGeo*   m_ggeo ; 
        GCache* m_cache ; 
        char*   m_mesh_version ;
        std::map<unsigned int,GMergedMesh*>  m_merged_mesh ; 
};

inline GGeoLib::GGeoLib(GCache* cache) 
     :
     m_cache(cache),
     m_mesh_version(NULL)
{
}


inline unsigned int GGeoLib::getNumMergedMesh()
{
    return m_merged_mesh.size();
}

inline GMergedMesh* GGeoLib::getMergedMesh(unsigned int index)
{
    if(m_merged_mesh.find(index) == m_merged_mesh.end()) return NULL ;
    return m_merged_mesh[index] ;
}






inline void GGeoLib::setMeshVersion(const char* mesh_version)
{
    m_mesh_version = mesh_version ? strdup(mesh_version) : NULL ;
}
inline const char* GGeoLib::getMeshVersion()
{
    return m_mesh_version ;
}


