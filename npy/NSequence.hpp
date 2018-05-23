#pragma once

#include "NPY_API_EXPORT.hh"

class NPY_API NSequence {
  public:
      virtual ~NSequence(){}

      virtual unsigned int getNumKeys() = 0;
      virtual const char* getKey(unsigned int idx) = 0;
      virtual unsigned int getIndex(const char* key) = 0;
};


