#pragma once

#include <map>
#include <string>
#include <vector>


class Parameters {
   public:
       typedef std::pair<std::string, std::string>   SS ; 
       typedef std::vector<SS>                      VSS ; 
       typedef std::vector<std::string>              VS ; 
   public:
       Parameters();

       template <typename T>
       void add(const char* name, T value);

       void dump(const char* msg="Parameters::dump");
       void prepLines();

       std::vector<std::string>& getLines();


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
