#pragma once

#include <string>
#include <vector>


#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API NTxt {
   public:
       typedef std::vector<std::string> VS_t ; 
   public:
       NTxt(const char* path); 
       void read();
       const char* getLine(unsigned int num);
       unsigned int getIndex(const char* line); // index of line or UINT_MAX if not found
       unsigned int getNumLines();
   public:
       void addLine(const char* line); 
       void write();
   private:
       const char* m_path ; 
       VS_t m_lines ; 

};

#include "NPY_TAIL.hh"

