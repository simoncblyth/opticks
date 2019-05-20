#pragma once

#include <string>
#include <vector>

#include "NPY_API_EXPORT.hh"

class NCSG ; 
class NGeoTestConfig ; 
class BTxt ; 

#include "NBBox.hpp"

class NPY_API NCSGList 
{
    public:
        static const char* FILENAME ; 
        static NCSGList* Load(const char* csgpath, int verbosity=-1, bool checkmaterial=true) ;
        static bool      ExistsDir(const char* dir);
        static const char* MakeUniverseBoundary( const char* boundary0 );
        static NCSGList* Create(std::vector<NCSG*>& trees,  const char* csgpath, int verbosity ); 
    public:
        // from GGeoTest
        void autoTestSetup(NGeoTestConfig* config);
    private:
        NCSGList(const char* csgpath, int verbosity);
        void init() ;
        void load() ;
        void checkMaterialConsistency() const  ;

        NCSG* loadTree(unsigned idx, const char* boundary) const ;
        NCSG* createUniverse(float scale, float delta) const ;

    public:
        void savesrc() const ;
    private:
        void add(NCSG* tree) ; 

    public:
        void dump(const char* msg="NCSGList::dump") const ;
        void dumpDesc(const char* msg="NCSGList::dumpDesc") const ;
        void dumpMeta(const char* msg="NCSGList::dumpMeta") const ;
        void dumpUniverse(const char* msg="NCSGList::dumpUniverse")  ; // not-const as may create

    public:
        std::vector<NCSG*>& getTrees(); 
        std::string  getTreeDir(unsigned idx) const ;
        NCSG*        getTree(unsigned index) const ;
        unsigned     getNumTrees() const ;
    public:
        NCSG*        getUniverse() ;   // not-const as may create
        const char*  getBoundary(unsigned index) const ;
        NCSG*        findEmitter() const ;
        int          polygonize();

    private:
        const char*        m_csgpath ; 
        const char*        m_txtpath ; 
        int                m_verbosity ; 
        BTxt*              m_bndspec ; 
        NCSG*              m_universe ; 
        std::vector<NCSG*> m_trees ; 
        nbbox              m_container_bbox ; 

};
 
