#pragma once

#include <string>
#include <map>
#include <vector>

#include "BRAP_API_EXPORT.hh"


/**

BConfig
==========

Intended as simple alternative to BCfg, for usage examples
see NSnapConfig, NSceneConfig


Lifecycle:

1. instanciated as ctor member of a configuration class, 
   such as NSnapConfig, with a const char* argument
2. the holding class invokes the addInt/addFloat/addString 
   methods with key names and pointers to corresponding member variables.
3. BConfig::parse is invoked which looks for keys that 
   correspond to those setup with the add methods, and thus 
   BConfig sets the corresponding member variables using the pointers  


Note that BConfig does not store any values only their key strings
and pointers to where the values can be found.

**/

struct BRAP_API BConfig
{
    static const char* DEFAULT_KVDELIM ; 

    typedef std::pair<std::string,std::string> KV ;

    typedef std::pair<std::string,std::string*> KS ;
    typedef std::pair<std::string,int*>        KI ;
    typedef std::pair<std::string,float*>      KF ;


    const char*     cfg ; 
    char            edelim ; 
    const char*     kvdelim ; 

    std::vector<KV> ekv ; 
    std::vector<KI> eki ; 
    std::vector<KF> ekf ; 
    std::vector<KS> eks ; 

    void addInt(  const char* k, int* ptr);
    void addFloat(const char* k, float* ptr);
    void addString(const char* k, std::string* ptr);


    BConfig(const char* cfg, char delim=',', const char* kvdelim=NULL);
    void parse();

    void dump(const char* msg="BConfig::dump") const ;
    void dump_ekv() const ; 
    void dump_eki() const ; 
    void dump_ekf() const ; 
    void dump_eks() const ; 

    std::string desc() const ; 


};

