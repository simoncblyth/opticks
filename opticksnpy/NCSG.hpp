#pragma once

#include <string>
#include <vector>

template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"

/**
NCSG
======

cf dev/csg/csg.py 

* this can currently only Deserialize from a python written directory 

**/

struct nvec4 ; 
struct nnode ; 

class NPY_API NCSG {
    public:
        enum { NJ = 4, NK = 4, MAX_HEIGHT = 10 };
        static const char* FILENAME ; 
        static unsigned NumNodes(unsigned height);
        static int Deserialize(const char* base, std::vector<NCSG*>& trees);
        static NCSG* FromNode(nnode* root, const char* boundary);
    public:
        void dump(const char* msg="NCSG::dump");
        std::string desc();
    public:
        const char* getPath();
        unsigned getIndex();
        const char* getBoundary();
        NPY<float>* getBuffer();
        unsigned getNumNodes();
        unsigned getHeight();
        nnode* getRoot();
    private:
        // Deserialize
        NCSG(const char* path, unsigned index=0u);
        void setBoundary(const char* boundary);
        unsigned getTypeCode(unsigned idx);
        nvec4 getQuad(unsigned idx, unsigned j);
        void load();
        void import();
        nnode* import_r(unsigned idx);
    private:
         // FromNode
        void importRoot(nnode* root);
    private:
        const char* m_path ; 
        unsigned    m_index ; 
        const char* m_boundary ; 
        NPY<float>* m_data ; 
        unsigned    m_num_nodes ; 
        unsigned    m_height ; 
        nnode*      m_root ;  

};


