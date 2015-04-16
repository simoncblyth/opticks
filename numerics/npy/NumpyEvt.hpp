#pragma once
#include <string>
class NPY ;

class NumpyEvt {
   public:
       NumpyEvt();

       void setNPY(NPY* npy);
       NPY* getNPY();
       bool hasNPY();

       std::string description(const char* msg);
   private:
       NPY* m_npy ;
};



