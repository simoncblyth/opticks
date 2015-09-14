#pragma once
#include "string.h"

#include <string>
#include <map>
#include <vector>


class DynamicDefine {
    public:
       DynamicDefine();  
    public:
       template<typename T>
       void add(const char* name, T value);

       void write(const char* dir, const char* name);
    private:
       std::vector<std::pair<std::string, std::string> > m_defines ; 

};


inline DynamicDefine::DynamicDefine() 
{
}    


