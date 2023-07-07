#pragma once

#include <vector>
#include <string>

struct snam
{
    static const char* get(const std::vector<std::string>& names, int idx ) ;  
};

inline const char* snam::get(const std::vector<std::string>& names, int idx)
{
    return idx > -1 && idx < int(names.size()) ? names[idx].c_str() : nullptr ; 
}

