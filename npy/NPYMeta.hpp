#pragma once

#include <string>
#include <map>

#ifdef OLD_PARAMETERS
class X_BParameters ; 
#else
class NMeta ; 
#endif

#include "NPY_API_EXPORT.hh"

/**
NPYMeta
=========

**/

class NPY_API NPYMeta
{
    public:
#ifdef OLD_PARAMETERS
        static X_BParameters* LoadMetadata(const char* treedir, int item=-1);
#else
        static NMeta*       LoadMetadata(const char* treedir, int item=-1);
#endif
        static bool         ExistsMeta(const char* treedir, int item=-1);
    private:
        static const char*  META ; 
        static const char*  ITEM_META ; 
        static std::string  MetaPath(const char* treedir, int item=-1);
        enum { NUM_ITEM = 16 } ;  // default number of items to look for
    public:
        // item -1 corresponds to global metadata 
        NPYMeta(); 
#ifdef OLD_PARAMETERS
        X_BParameters*  getMeta(int item=-1) const ;   
#else
        NMeta*  getMeta(int item=-1) const ;   
#endif
        bool          hasMeta(int idx) const ;
    public:
        int                       getIntFromString(const char* key, const char* fallback, int item=-1 ) const ;
        template<typename T> T    getValue(const char* key, const char* fallback, int item=-1 ) const ;
        template<typename T> void setValue(const char* key, T value, int item=-1);
    public:
        void load(const char* dir, int num_item = NUM_ITEM ) ;
        void save(const char* dir) const ;
    private:
#ifdef OLD_PARAMETERS
        std::map<int, X_BParameters*>    m_meta ;    
#else
        std::map<int, NMeta*>    m_meta ;    
#endif
        // could be a complete binary tree with loada nodes, so std::array not appropriate

};



