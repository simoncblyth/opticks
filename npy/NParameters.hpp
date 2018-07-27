#pragma once

#include <map>
#include <string>
#include <vector>

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API NParameters {
   public:
       static NParameters* load(const char* path);
       static NParameters* load(const char* dir, const char* name);
   public:
       typedef std::pair<std::string, std::string>   SS ; 
       typedef std::vector<SS>                      VSS ; 
       typedef std::vector<std::string>              VS ; 
   public:
       NParameters();
      const std::vector<std::pair<std::string,std::string> >& getVec() ;

       std::string getStringValue(const char* name) const ;
   public:
       void append(NParameters* other);
   public:

       template <typename T> 
       void add(const char* name, T value);

       template <typename T> 
       void set(const char* name, T value);

       template <typename T> 
       void append(const char* name, T value, const char* delim=" ");

       template <typename T> 
       T get(const char* name) const ;

       template <typename T> 
       T get(const char* name, const char* fallback) const ;

       template <typename T> 
       T get_fallback(const char* fallback) const ;

   public:
       unsigned getNumItems();
       void dump();
       void dump(const char* msg);
       std::string desc();
       void prepLines();
       std::vector<std::string>& getLines();
   public:
       void save(const char* path);
       void save(const char* dir, const char* name);
   public:
       void load_(const char* path);
       void load_(const char* dir, const char* name);
   private:
       VSS m_parameters ; 
       VS  m_lines ;  

};

#include "NPY_TAIL.hh"


