#pragma once

#include <string>
#include <vector>


#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API BTxt {
   public:
       typedef std::vector<std::string> VS_t ; 
   public:
       BTxt(const char* path = NULL); 
       void read();
   public:
       std::string desc() const ; 
       void dump(const char* msg="BTxt::dump") const ;
       const char* getLine(unsigned int num) const ; 
       unsigned int getNumLines() const ;
       unsigned int getIndex(const char* line) const ; // index of line or UINT_MAX if not found
       void write(const char* path=NULL) const ;
       void prepDir(const char* path=NULL) const ; 
   public:
       void addLine(const std::string& line); 
       void addLine(const char* line); 
       template<typename T> void addValue(T value); 
       const std::vector<std::string>& getLines() const ; 
   private:
       const char* m_path ; 
       VS_t m_lines ; 

};

#include "BRAP_TAIL.hh"

