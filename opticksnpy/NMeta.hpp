#pragma once

#include <vector>
#include <string>

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"
#include "NJS.hpp"

//class NJS ; 

class NPY_API NMeta {
   public:
       static NMeta* Load(const char* path);
       static NMeta* Load(const char* dir, const char* name);
   public:
       NMeta();
       nlohmann::json& js();
   public:
       const char* getKey(unsigned idx) const ;
       unsigned    getNumKeys() ; // non-const may updateKeys
       std::string desc(unsigned wid=0);
   private:
       void        updateKeys();
   public:
       void   setObj(const char* name, NMeta* obj); 
       NMeta* getObj(const char* name);
   public:
       template <typename T> void set(const char* name, T value);
       template <typename T> T get(const char* name) const ;
       template <typename T> T get(const char* name, const char* fallback) const ;
   public:
       void save(const char* path) const ;
       void save(const char* dir, const char* name) const ;
       void dump() const ; 
   public:
       void load(const char* path);
       void load(const char* dir, const char* name);
   private:
       NJS* m_js ; 
       std::vector<std::string> m_keys ; 

};

#include "NPY_TAIL.hh"


