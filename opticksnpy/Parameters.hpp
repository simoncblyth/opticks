#pragma once

#include <map>
#include <string>
#include <vector>

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API Parameters {
   public:
       static Parameters* load(const char* path);
       static Parameters* load(const char* dir, const char* name);
   public:
       typedef std::pair<std::string, std::string>   SS ; 
       typedef std::vector<SS>                      VSS ; 
       typedef std::vector<std::string>              VS ; 
   public:
       Parameters();
       const std::vector<std::pair<std::string,std::string> >& getVec() ;

       std::string getStringValue(const char* name);
   public:
       void append(Parameters* other);
   public:
       template <typename T> 
       void add(const char* name, T value);

       template <typename T> 
       void set(const char* name, T value);

       template <typename T> 
       T get(const char* name);

       template <typename T> 
       T get(const char* name, const char* fallback);
   public:
       void dump();
       void dump(const char* msg);
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


