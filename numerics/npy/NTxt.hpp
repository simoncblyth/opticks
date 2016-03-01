#pragma once

#include <cstring>
#include <string>
#include <vector>

class NTxt {
   public:
       NTxt(const char* path); 
       void read();
       const char* getLine(unsigned int num);
       unsigned int getNumLines();
   private:
       const char* m_path ; 
       std::vector<std::string> m_lines ; 

};

inline NTxt::NTxt(const char* path)
   :
   m_path(strdup(path))
{
}

inline const char* NTxt::getLine(unsigned int num)
{
   return num < m_lines.size() ? m_lines[num].c_str() : NULL ; 
}
inline unsigned int  NTxt::getNumLines()
{
   return m_lines.size() ; 
}



