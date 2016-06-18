#pragma once

#include <map>
#include <string>
#include <vector>

#include "NPY_API_EXPORT.hh"

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
       std::string getStringValue(const char* name);

       template <typename T> 
       void add(const char* name, T value);

       template <typename T> 
       T get(const char* name);

       template <typename T> 
       T get(const char* name, const char* fallback);


       void dump(const char* msg="Parameters::dump");
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



/*

https://social.msdn.microsoft.com/Forums/vstudio/en-US/4fd49664-e28e-4f23-b1eb-b669d35ad264/function-template-instantation-export-from-dll?forum=vcgeneral


https://social.msdn.microsoft.com/Forums/vstudio/en-US/0d613b65-52ac-4fb7-bf65-8a543dfbcc6e/visual-c-error-lnk2019-unresolved-external-symbol?forum=vcgeneral





*/

