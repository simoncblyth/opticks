#pragma once

#include <string>
#include <map>
class Opticks ; 

class GBndLib ; 
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
        static const char* GPARTS ; 
        enum { MAX_MERGED_MESH = 10 } ;
    public:
        static GGeoLib* Load(Opticks* ok, bool analytic, GBndLib* bndlib);
    public:
        GGeoLib(Opticks* ok, bool analytic, GBndLib* bndlib);
        std::string desc() const ; 
        void setMeshVersion(const char* mesh_version);
        const char* getMeshVersion() const ;
    public:
        void loadFromCache();
        void save();
        GMergedMesh* makeMergedMesh(unsigned index, GNode* base, GNode* root, unsigned verbosity );
    private:
        const char* getRelDir(const char* name);
        void loadConstituents(const char* idpath);
        void removeConstituents(const char* idpath);
        void saveConstituents(const char* idpath);
    public:
        void dump(const char* msg="GGeoLib::dump");
        unsigned getNumMergedMesh() const ;
        GMergedMesh* getMergedMesh(unsigned int index);
        void setMergedMesh(unsigned int index, GMergedMesh* mm);
        void eraseMergedMesh(unsigned int index);
        void clear();
    private:
        Opticks* m_ok ; 
        bool     m_analytic ; 
        GBndLib* m_bndlib ; 
        char*   m_mesh_version ;
        int     m_verbosity ;
        std::map<unsigned,GMergedMesh*>  m_merged_mesh ; 
};

#include "GGEO_TAIL.hh"

