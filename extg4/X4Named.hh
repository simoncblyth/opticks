#pragma once

#include "X4_API_EXPORT.hh"
#include <string>

struct X4_API X4Named
{
    X4Named(const char* name_) : name(name_) {} 
    std::string name ; 
    const std::string& GetName() const { return name ; } 
};


