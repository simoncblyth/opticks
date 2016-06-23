#pragma once

#include <string>
#include <map>
#include <vector>

#include "OGLRAP_API_EXPORT.hh"
#include "OGLRAP_HEAD.hh"

class OGLRAP_API DynamicDefine {
    public:
       DynamicDefine();  
    public:
       template<typename T>
       void add(const char* name, T value);
       void write(const char* dir, const char* name);

    private:
       std::vector<std::pair<std::string, std::string> > m_defines ; 

};

#include "OGLRAP_TAIL.hh"

