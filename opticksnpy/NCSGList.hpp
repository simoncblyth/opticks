#pragma once

#include <string>
#include <vector>

#include "NPY_API_EXPORT.hh"

class NCSG ; 
class NTxt ; 

#include "NBBox.hpp"

class NPY_API NCSGList 
{
    public:
        static const char* FILENAME ; 
        static NCSGList* Load(const char* csgpath, int verbosity, bool checkmaterial=true) ;
        static bool      ExistsDir(const char* dir);
        static const char* MakeUniverseBoundary( const char* boundary0 );

    private:
        NCSGList(const char* csgpath, int verbosity);
        void init() ;
        void load() ;
        void checkMaterialConsistency() const  ;

        NCSG* loadTree(unsigned idx, const char* boundary) const ;
        NCSG* createUniverse(float scale, float delta) const ;
    public:
        void dump(const char* msg="NCSGList::dump") const ;
        void dumpDesc(const char* msg="NCSGList::dumpDesc") const ;
        void dumpMeta(const char* msg="NCSGList::dumpMeta") const ;
        void dumpUniverse(const char* msg="NCSGList::dumpUniverse")  ; // not-const as may create
    public:
        std::string getTreeDir(unsigned idx) const ;

        NCSG*        getUniverse() ;   // not-const as may create
        NCSG*        getTree(unsigned index) const ;
        const char*  getBoundary(unsigned index) const ;
        NCSG*        findEmitter() const ;
        unsigned     getNumTrees() const ;
        int          polygonize();

        std::vector<NCSG*>& getTrees(); 

    private:
        const char*        m_csgpath ; 
        int                m_verbosity ; 
        NTxt*              m_bndspec ; 
        NCSG*              m_universe ; 
        std::vector<NCSG*> m_trees ; 
        nbbox              m_container_bbox ; 

};
 
