#pragma once

#include <string>
#include <map>
#include <vector>

#include "BRAP_API_EXPORT.hh"

// intended as simple alternative to BCfg, for usage example see NSceneConfig

struct BRAP_API BConfig
{
    typedef std::pair<std::string,std::string> KV ;
    typedef std::pair<std::string,int*>        KI ;
    const char*     cfg ; 

    std::vector<KV> ekv ; 
    std::vector<KI> eki ; 

    void addInt(const char* k, int* ptr);
//    template <typename T> void add<T>(const char* k, T* ptr);


    BConfig(const char* cfg);
    void parse();

    void dump(const char* msg="BConfig::dump") const ;
    void dump_ekv() const ; 

};

