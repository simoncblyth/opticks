#pragma once

#include <string>
#include <vector>

template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"

/**
NCSG
======

cf dev/csg/csg.py 

**/

struct nvec4 ; 
struct nnode ; 

class NPY_API NCSG {
    public:
        enum { MAX_HEIGHT = 10 };
        static const char* FILENAME ; 
        static int Deserialize(const char* base,  std::vector<NCSG*>& trees);
    public:
        NCSG(const char* path);
        void dump(const char* msg="NCSG::dump");
        void setBoundary(const char* boundary);

        void load();
        unsigned getTypeCode(unsigned idx);
        nvec4 getQuad(unsigned idx, unsigned j);
        void import();
        nnode* import_r(unsigned idx);

        std::string desc();
    private:
        const char* m_path ; 
        const char* m_boundary ; 
        NPY<float>* m_data ; 
        unsigned    m_num_nodes ; 
        unsigned    m_height ; 
        nnode*      m_root ;  

};


