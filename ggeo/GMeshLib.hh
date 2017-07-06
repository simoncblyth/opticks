#pragma once

#include <map>
#include <string>
#include <vector>

class Opticks ; 

class GItemIndex ; 
class GMesh ; 

#include "GGEO_API_EXPORT.hh"

/*

GMeshLib
===========

::

    op --dsst -G 


*/

class GGEO_API GMeshLib 
{
        friend class GGeo ; 
        friend class GScene ; 
    public:
        static const unsigned MAX_MESH  ; 

        static const char*    GITEMINDEX ; 
        static const char*    GMESHLIB_INDEX ; 
        static const char*    GMESHLIB_INDEX_ANALYTIC ; 
        static const char*    GetRelDirIndex(bool analytic);

        static const char*    GMESHLIB ; 
        static const char*    GMESHLIB_ANALYTIC ; 
        static const char*    GetRelDir(bool analytic);

        static GMeshLib* load(Opticks* ok, bool analytic);
    public:
        GMeshLib(Opticks* opticks, bool analytic); 
        bool isAnalytic() const ; 
        void add(GMesh* mesh);
        void dump(const char* msg="GMeshLib::dump") const;
    public:
        // methods working from the index, so work prior to loading meshes
        unsigned    getMeshIndex(const char* name, bool startswith) const ;
        const char* getMeshName(unsigned aindex) ; 
    public:
        //std::string desc() const ; 
        GItemIndex* getMeshIndex() ;
        unsigned    getNumMeshes() const ; 
        GMesh*      getMesh(unsigned aindex) const ;
        GMesh*      getMesh(const char* name, bool startswith) const ;
    private:
        void        loadFromCache();
        void        save() const ; 
    private:
        void saveMeshes(const char* idpath) const ;
        void loadMeshes(const char* idpath ) ;
        void removeMeshes(const char* idpath ) const ;
    private:
        Opticks*                      m_ok ; 
        bool                          m_analytic ; 
        const char*                   m_reldir ; 
        GItemIndex*                   m_meshindex ; 
        unsigned                      m_missing ; 
        std::vector<GMesh*>           m_meshes ; 


};
