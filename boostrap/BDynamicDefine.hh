#pragma once

#include <string>
#include <map>
#include <vector>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API BDynamicDefine {
    public:
       BDynamicDefine();  
    public:
       template<typename T>
       void add(const char* name, T value);
       void write(const char* dir, const char* name);

    private:
       std::vector<std::pair<std::string, std::string> > m_defines ; 

};

#include "BRAP_TAIL.hh"

