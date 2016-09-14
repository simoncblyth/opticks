#pragma once

// for both non-CUDA and CUDA compilation
typedef enum {
   F_UNDEF,
   F_POINT,
   F_NUM_TYPE
}  Fab_t ;

#ifndef __CUDACC__

#include <string>
#include "NGLM.hpp"

template<typename T> class NPY ; 

#include "GenstepNPY.hpp"
#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API FabStepNPY : public GenstepNPY 
{
   public:
        FabStepNPY(unsigned code, unsigned num_step, unsigned num_photons_per_step);
        void update();
   private: 
        void init();
   private:
        unsigned m_num_photons_per_step ;

};

#include "NPY_TAIL.hh"
#endif


