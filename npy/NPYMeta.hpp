#pragma once

#include <string>
#include <map>

class BParameters ; 

#include "NPY_API_EXPORT.hh"

/**
NPYMeta
=========

**/

class NPY_API NPYMeta
{
    public:
        static BParameters* LoadMetadata(const char* treedir, int item=-1);
        static bool         ExistsMeta(const char* treedir, int item=-1);
    private:
        static const char*  META ; 
        static const char*  ITEM_META ; 
        static std::string  MetaPath(const char* treedir, int item=-1);
        enum { NUM_ITEM = 16 } ;  // default number of items to look for
    public:
        // item -1 corresponds to global metadata 
        NPYMeta(); 
        BParameters*  getMeta(int item=-1) const ;   
        bool          hasMeta(int idx) const ;
    public:
        template<typename T> T    getValue(const char* key, const char* fallback, int item=-1 ) const ;
        template<typename T> void setValue(const char* key, T value, int item=-1);
    public:
        void load(const char* dir, int num_item = NUM_ITEM ) ;
        void save(const char* dir) const ;
    private:
        std::map<int, BParameters*>    m_meta ;    
        // could be a complete binary tree with loada nodes, so std::array not appropriate

};



