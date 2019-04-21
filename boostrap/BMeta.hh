#pragma once

/**
BMeta
=======

For simple key,value string lists metadata persisting beneath NPY level, 
ie without using NMeta (which uses the nlohman::json). 


**/

#include <map>
#include <vector>
#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API BMeta 
{
    public: 
        static BMeta* Load(const char* dir, const char* label );  
    private:
        static const char* EXT ; 
        static std::string Name(const char* label); 
    public:
        BMeta(const char* label="BMeta"); 
        void add( const char* k, const char* v); 
        void addEnvvar( const char* k ); 
        void dump(const char* msg="BMeta::dump") const ; 
        void save(const char* dir) ;
        void load(const char* dir) ;
    private:
        typedef std::pair<std::string, std::string>  SS ; 
        typedef std::vector<SS>                      VSS ; 
    private:
        const char* m_label ;  
        VSS         m_kv ; 
};

#include "BRAP_TAIL.hh"

