#pragma once

struct short4 
{  
   short x ; 
   short y ; 
   short z ; 
   short w ; 
};

struct ushort4 
{  
   unsigned short x ; 
   unsigned short y ; 
   unsigned short z ; 
   unsigned short w ; 
};

union hquad
{   
   short4   short_ ;
   ushort4  ushort_ ;
};  

struct char4
{
   char x ; 
   char y ; 
   char z ; 
   char w ; 
};

struct uchar4
{
   unsigned char x ; 
   unsigned char y ; 
   unsigned char z ; 
   unsigned char w ; 
};

union qquad
{   
   char4   char_   ;
   uchar4  uchar_  ;
};  

union uifchar4
{
   unsigned int u ; 
   int          i ; 
   float        f ; 
   char4        char_   ;
   uchar4       uchar_  ;
};



