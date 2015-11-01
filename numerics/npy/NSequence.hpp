#pragma once

class NSequence {
  public:
      virtual ~NSequence(){}

      virtual unsigned int getNumKeys() = 0;
      virtual const char* getKey(unsigned int idx) = 0;
      virtual unsigned int getIndex(const char* key) = 0;
};


