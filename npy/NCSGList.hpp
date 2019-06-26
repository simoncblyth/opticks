#pragma once

#include <string>
#include <vector>

#include "NPY_API_EXPORT.hh"

class NCSG ; 
class NGeoTestConfig ; 
class BTxt ; 

#include "NBBox.hpp"
#include "plog/Severity.h"

/**
NCSGList
===========

Used by GGeoTest and many other tests to load 
persisted trees.  


**/


class NPY_API NCSGList 
{
        static const plog::Severity LEVEL ; 
    public:
        typedef enum { PROXY, EMITTER, CONTAINER } NCSG_t ;
  
        static const char* FILENAME ; 
        static NCSGList* Load(const char* csgpath, int verbosity=-1 ) ;
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
        void postload(bool checkmaterial=true);
        void checkMaterialConsistency() const  ;

        NCSG* loadTree(unsigned idx) const ;
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
        void         setTree(unsigned index, NCSG* replacement );
        unsigned     getNumTrees() const ;
    public:
        NCSG*        getUniverse() ;   // not-const as may create
        const char*  getBoundary(unsigned index) const ;
    public:
        NCSG*        findEmitter() const ;
        NCSG*        findContainer() const ;
        NCSG*        findProxy() const ;
    public:
        int          findEmitterIndex() const ;
        int          findContainerIndex() const ;
        int          findProxyIndex() const ;
    public:
        bool         hasContainer() const ;  
        bool         hasProxy() const ;  
        bool         hasEmitter() const ;  
    public:
        void         update() ; 
        int          polygonize();
    private:
        NCSG*        find( NCSG_t type ) const ;
        int          findIndex( NCSG_t type ) const ;
        void         adjustContainerSize(); 
        void         updateBoundingBox() ; 

    private:
        const char*        m_csgpath ; 
        const char*        m_txtpath ; 
        int                m_verbosity ; 
        BTxt*              m_bndspec ; 
        NCSG*              m_universe ; 
        std::vector<NCSG*> m_trees ; 
        nbbox              m_bbox ; 

};
 
