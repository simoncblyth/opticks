#pragma once
#include "string.h"

#include <string>
#include <map>
#include <vector>


class DynamicDefine {
    public:
       DynamicDefine(const char* dir, const char* name);  
    public:
       template<typename T>
       void add(const char* name, T value);

       void write();
    private:
       const char* m_dir ;
       const char* m_name ;

       std::vector<std::pair<std::string, std::string> > m_defines ; 

};


inline DynamicDefine::DynamicDefine(const char* dir, const char* name) 
     :
     m_dir(strdup(dir)),
     m_name(strdup(name))
{
}    


