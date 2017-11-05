#pragma once

#include <string>
#include <vector>
#include "NCSG.hpp"

#include "NPY_API_EXPORT.hh"

class NPY_API NCSGList 
{
    public:
        static const char* FILENAME ; 
        static NCSGList* Load(const char* csgpath, int verbosity) ;
        static bool      ExistsDir(const char* dir);

        NCSGList(const char* csgpath, int verbosity);
        void load() ;
        void dump(const char* msg="NCSGList::dump") const ;
    public:
        std::string getTreeDir(unsigned idx);
        NCSG*    getTree(unsigned index) const ;
        NCSG*    findEmitter() const ;
        unsigned getNumTrees() const ;
        int      polygonize();

        std::vector<NCSG*>& getTrees(); 

    private:
        const char*        m_csgpath ; 
        int                m_verbosity ; 
        std::vector<NCSG*> m_trees ; 

};
 
