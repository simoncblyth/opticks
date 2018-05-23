#pragma once

#include <string>
template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"

class NPY_API NLoad {
   public:
       static NPY<float>* Gensteps( const char* det, const char* typ, const char* tag );
       static std::string GenstepsPath( const char* det, const char* typ, const char* tag );
       static std::string directory(const char* det, const char* typ, const char* tag, const char* anno=NULL);

};
