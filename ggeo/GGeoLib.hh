#pragma once

#include <string>
#include <map>
class Opticks ; 

//class GGeo ; 
class GMergedMesh ; 
class GNode ; 

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/*
GGeoLib
==========

Container for GMergedMesh

*/

class GGEO_API GGeoLib {
    public:
        static const char* GMERGEDMESH ; 
        enum { MAX_MERGED_MESH = 10 } ;
    public:
        static GGeoLib* load(Opticks* opticks);
    public:
        GGeoLib(Opticks* opticks);
        std::string desc() const ; 
        void setMeshVersion(const char* mesh_version);
        const char* getMeshVersion() const ;
    public:
        void loadFromCache();
        void saveToCache();
        GMergedMesh* makeMergedMesh(unsigned index, GNode* base, GNode* root, unsigned verbosity );
    private:
        void loadMergedMeshes(const char* idpath);
        void removeMergedMeshes(const char* idpath);
        void saveMergedMeshes(const char* idpath);
    public:
        unsigned getNumMergedMesh() const ;
        GMergedMesh* getMergedMesh(unsigned int index);
        void setMergedMesh(unsigned int index, GMergedMesh* mm);
        void eraseMergedMesh(unsigned int index);
        void clear();
    private:
        Opticks* m_opticks ; 
        char*   m_mesh_version ;
        std::map<unsigned,GMergedMesh*>  m_merged_mesh ; 
};

#include "GGEO_TAIL.hh"

