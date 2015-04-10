#pragma once

// /usr/local/env/cuda/OptiX_370b2_sdk/sutil/GLUTDisplay.cpp

#include <optixu/optixpp_namespace.h>

class OptiXEngine ; 

class OptiXDisplay {
   public:
       OptiXDisplay(OptiXEngine* engine);

   private:
   //    static void displayFrame();   

       OptiXEngine* m_engine ;



};
