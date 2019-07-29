#pragma once

#include <string>
#include <vector>


#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BTxt ; 

class BRAP_API BLog {
   public:
       static const double TOLERANCE ; 
       static BLog* Load(const char* path); 
       static int ParseLine( const std::string& line,  std::string& k, std::string& v ); 
       static int Compare( const BLog* a , const BLog* b ) ; 
   public:
       BLog(); 

       void setSequence(const std::vector<double>*  sequence);   

       void add( const char* key, double value ); 

       unsigned    getNumKeys() const ;  
       const char* getKey(unsigned i) const ; 
       double      getValue(unsigned i) const ; 
       int         getSequenceIndex(unsigned i ) const ;
       const std::vector<double>&  getValues() const ; 

       void        dump(const char* msg="BLog::dump") const ; 

       std::string  makeLine(unsigned i) const ; 
       BTxt*        makeTxt() const ; 
       void         write(const char* path) const ; 

   private:
       void init();  

   private:
       std::vector<std::string>     m_keys ; 
       std::vector<double>          m_values ; 
       const std::vector<double>*   m_sequence ;    


};

#include "BRAP_TAIL.hh"

