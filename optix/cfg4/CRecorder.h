#pragma once


#include "CFG4_API_EXPORT.hh"

struct CFG4_API short4 
{  
   short x ; 
   short y ; 
   short z ; 
   short w ; 
};

struct CFG4_API ushort4 
{  
   unsigned short x ; 
   unsigned short y ; 
   unsigned short z ; 
   unsigned short w ; 
};

union CFG4_API hquad
{   
   short4   short_ ;
   ushort4  ushort_ ;
};  

struct CFG4_API char4
{
   char x ; 
   char y ; 
   char z ; 
   char w ; 
};

struct CFG4_API uchar4
{
   unsigned char x ; 
   unsigned char y ; 
   unsigned char z ; 
   unsigned char w ; 
};

union CFG4_API qquad
{   
   char4   char_   ;
   uchar4  uchar_  ;
};  

union CFG4_API uifchar4
{
   unsigned int u ; 
   int          i ; 
   float        f ; 
   char4        char_   ;
   uchar4       uchar_  ;
};



