#pragma once

template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"

class NPY_API NLoad {
   public:
       static NPY<float>* Gensteps( const char* det, const char* typ, const char* tag );

};
