#pragma once

#include "NPY_API_EXPORT.hh"
#include <string>

/**
NGrid
======

* (row, column) grid of T struct pointers, initally all NULL
* set/get pointers in the grid, battleship style
* string display assuming only that the T struct has a "label" 
  member that provides brief identification eg 3 characters only.

**/

template <typename T>
struct NPY_API NGrid 
{
   NGrid(unsigned nr_, unsigned nc_, unsigned width_=4, const char* unset_="", const char* rowjoin_="\n\n" );
   ~NGrid();

   void init();
   void clear();

   unsigned idx(unsigned r, unsigned c) const  ; 
   void     set(unsigned r, unsigned c, T* ptr);
   T*       get(unsigned r, unsigned c) const ; 

   std::string desc() ;

   unsigned    nr ;  
   unsigned    nc ;  
   unsigned    width ; 
   const char* unset ;
   const char* rowjoin ;

   T**      grid ;   // linear array of pointers to T 
   
   // Judged that the hassles of 2d arrays are not worth the bother 
   // for the minor benefit of 2d indexing,
   // when can trivially compute a linear index : in a way that 
   // easily scales to 3,4,5,.. dimensions.

};
