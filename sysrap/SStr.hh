#pragma once


#include <cstddef>

#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SStr {

    typedef unsigned long long ULL ;
  public:
      static void FillFromULL( char* dest, unsigned long long value, char unprintable='.') ; 
      static const char* FromULL(unsigned long long value, char unprintable='.'); 
      static unsigned long long ToULL(const char* s8 ); 
};



