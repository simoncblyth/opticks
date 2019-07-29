#pragma once

#include <string>
#include <vector>


#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BTxt ; 

class BRAP_API BLog {
   public:
       static BLog* Load(const char* path); 
       static int ParseLine( const std::string& line,  std::string& k, std::string& v ); 

   public:
       BLog(); 

       void add( const char* key, double value ); 

       unsigned    getNumKeys() const ;  
       const char* getKey(unsigned i) const ; 
       double      getValue(unsigned i) const ; 
       void        dump(const char* msg="BLog::dump") const ; 

   private:
       void init();  

   private:
       std::vector<std::string>  m_keys ; 
       std::vector<double>       m_values ; 

};

#include "BRAP_TAIL.hh"

