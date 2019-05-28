#pragma once
/**
SASCII
========

**/


#include <string>
#include <cstddef>

#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SASCII 
{
  public:
      static const char UPPER[] ;  
      static const char LOWER[] ;  
      static const char NUMBER[] ;  
      static const char OTHER[] ;  
      static const char ALLOWED[] ;  
  public:
      static unsigned Count( char c, const char* list );  
      static bool IsUpper( char c );  
      static bool IsLower( char c );  
      static bool IsNumber( char c );  
      static bool IsOther( char c );  
      static bool IsAllowed( char c );
      static char Classify( char c); 

      static void DumpAllowed();
      static void Dump(const char* s);

  public:
      SASCII( const char* s_); 
      std::string getFirst(unsigned n) const ; 
      std::string getFirstUpper(unsigned n) const ; 
      std::string getFirstLast() const ; 
      std::string getTwoChar(unsigned first, unsigned second) const ; 
  private:
      void init();   
  public:
      const char* s ; 
      unsigned len ; 
      unsigned upper; 
      unsigned lower; 
      unsigned number; 
      unsigned other ; 
      unsigned allowed ; 
};




