#pragma once

#include "NPY_API_EXPORT.hh"

/**
NSlice
=========

TODO: move to the py names : start/stop/step 


**/

struct NPY_API NSlice {

     unsigned int low ; 
     unsigned int high ; 
     unsigned int step ; 
     const char*  _description ; 

     NSlice(const char* slice, const char* delim=":");
     NSlice(unsigned int low, unsigned int high, unsigned int step=1);

     const char* description();
     unsigned int count();

     bool isHead(unsigned index, unsigned window);
     bool isTail(unsigned index, unsigned window);
     bool isMargin(unsigned index, unsigned window);


};

