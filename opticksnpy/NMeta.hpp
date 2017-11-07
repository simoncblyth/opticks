#pragma once

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
       void set(const char* name, NMeta* obj); 
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

};

#include "NPY_TAIL.hh"


