#pragma once

#include <string>
#include <map>
#include <vector>

#include "BRAP_API_EXPORT.hh"

// intended as simple alternative to BCfg, for usage example see NSceneConfig

struct BRAP_API BConfig
{
    typedef std::pair<std::string,std::string> KV ;

    typedef std::pair<std::string,std::string*> KS ;
    typedef std::pair<std::string,int*>        KI ;
    typedef std::pair<std::string,float*>      KF ;
    const char*     cfg ; 

    std::vector<KV> ekv ; 
    std::vector<KI> eki ; 
    std::vector<KF> ekf ; 
    std::vector<KS> eks ; 

    void addInt(  const char* k, int* ptr);
    void addFloat(const char* k, float* ptr);
    void addString(const char* k, std::string* ptr);


    BConfig(const char* cfg);
    void parse();

    void dump(const char* msg="BConfig::dump") const ;
    void dump_ekv() const ; 
    void dump_eki() const ; 
    void dump_ekf() const ; 
    void dump_eks() const ; 

    std::string desc() const ; 


};

