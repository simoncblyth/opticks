#pragma once

#include "NPY_API_EXPORT.hh"

struct nnode ; 

class NPY_API NNodeDump2
{
    public:
        NNodeDump2(const nnode* node);

        void dump() const  ;

        void dump_label(const char* pfx) const ;
        void dump_base() const ;
        void dump_prim() const ;
        void dump_transform() const ;
        void dump_gtransform() const ;
        void dump_planes() const ;

    private:
        const nnode* m_node ; 

};




 
