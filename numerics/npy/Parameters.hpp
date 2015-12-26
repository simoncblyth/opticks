#pragma once

#include <map>
#include <string>
#include <vector>


class Parameters {
   public:
       static Parameters* load(const char* path);
       static Parameters* load(const char* dir, const char* name);
   public:
       typedef std::pair<std::string, std::string>   SS ; 
       typedef std::vector<SS>                      VSS ; 
       typedef std::vector<std::string>              VS ; 
   public:
       Parameters();

       template <typename T> 
       void add(const char* name, T value);

       std::string getStringValue(const char* name);

       template <typename T> 
       T get(const char* name);

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


inline Parameters::Parameters()
{
}

inline std::vector<std::string>& Parameters::getLines()
{
    if(m_lines.size() == 0 ) prepLines();
    return m_lines ;
}
