#pragma once

#include <vector>
#include <string>

#include "NPY_API_EXPORT.hh"
#include "NYJSON.hpp"
#include "NPY_HEAD.hh"


class NPY_API NMeta {
   public:
       static NMeta* Load(const char* path);
       static NMeta* Load(const char* dir, const char* name);
   public:
       NMeta();
       NMeta(const NMeta& other);

       nlohmann::json& js();
       const nlohmann::json& cjs() const ;
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
       void dump(const char* msg) const ; 
   public:
       void load(const char* path);
       void load(const char* dir, const char* name);

   private:
       // formerly used separate NJS, but that makes copy-ctor confusing 
       void read(const char* path0, const char* path1=NULL);
       void write(const char* path0, const char* path1=NULL) const ;
 
   private:
       nlohmann::json  m_js ;  
       std::vector<std::string> m_keys ; 

};

#include "NPY_TAIL.hh"


