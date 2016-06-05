#pragma once

#include <cstring>
#include <map>

class GGeo ; 
class GMergedMesh ; 
class GNode ; 

class Opticks ; 

class GGeoLib {
    public:
        static const char* GMERGEDMESH ; 
        enum { MAX_MERGED_MESH = 10 } ;
    public:
        static GGeoLib* load(Opticks* opticks);
    public:
        GGeoLib(Opticks* opticks);
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
        Opticks* m_opticks ; 
        char*   m_mesh_version ;
        std::map<unsigned int,GMergedMesh*>  m_merged_mesh ; 
};

inline GGeoLib::GGeoLib(Opticks* opticks) 
     :
     m_opticks(opticks),
     m_mesh_version(NULL)
{
}

inline unsigned int GGeoLib::getNumMergedMesh()
{
    return m_merged_mesh.size();
}
inline void GGeoLib::setMeshVersion(const char* mesh_version)
{
    m_mesh_version = mesh_version ? strdup(mesh_version) : NULL ;
}
inline const char* GGeoLib::getMeshVersion()
{
    return m_mesh_version ;
}

