#pragma once

#include "NPY_API_EXPORT.hh"

class NPY_API NSequence {
  public:
      virtual ~NSequence(){}

      virtual unsigned int getNumKeys() const = 0;
      virtual const char* getKey(unsigned int idx) const  = 0;
      virtual unsigned int getIndex(const char* key) const = 0;
};


