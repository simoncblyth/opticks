#pragma once

#include <cstring>
#include <climits>
#include <string>
#include <vector>

class NTxt {
   public:
       NTxt(const char* path); 
       void read();
       const char* getLine(unsigned int num);
       unsigned int getIndex(const char* line); // index of line or UINT_MAX if not found
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

inline unsigned int NTxt::getIndex(const char* line)
{
   std::string s(line);
   for(unsigned int i=0 ; i < m_lines.size() ; i++) if(m_lines[i].compare(s)==0) return i ;
   return UINT_MAX ; 
}

