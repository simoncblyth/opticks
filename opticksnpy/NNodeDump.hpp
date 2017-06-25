#pragma once

#include "NPY_API_EXPORT.hh"

struct nnode ; 

class NPY_API NNodeDump
{
    public:
        NNodeDump(const nnode& node);

        void dump(const char* msg) const  ;
        void dump_prim( const char* msg) const ;
        void dump_transform( const char* msg) const ;
        void dump_gtransform( const char* msg) const ;

    private:
        const nnode& m_node ; 

};




 
