#pragma once

#include <string>
#include <vector>


#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"
#include "NYJSON.hpp"

/**
NJS
====

Convenience wrapper for JSON read/write 

**/


class NPY_API NJS {
   public:
       NJS(); 
       NJS(const nlohmann::json& js ); 
   public:
       nlohmann::json& get();
       void read(const char* path);
       void write(const char* path) const ;
       void dump(const char* msg="NJS::dump") const ; 
   private:
       nlohmann::json  m_js ;  
};

#include "NPY_TAIL.hh"

