#pragma once

#include <vector>
#include <string>

class Report {
   public:
      typedef std::vector<std::string> VS ; 
   public:
      static const char* TIMEFORMAT ;  
      static std::string timestamp();
      static std::string name(const char* typ, const char* tag);
   public:
      Report();
   public:
      void add(const VS& lines);
      void save(const char* dir, const char* name);

   private:
      VS          m_lines ; 

};

inline Report::Report()
{
}

inline void Report::add(const VS& lines)
{
    for(VS::const_iterator it=lines.begin() ; it!=lines.end() ; it++) m_lines.push_back(*it);
}

