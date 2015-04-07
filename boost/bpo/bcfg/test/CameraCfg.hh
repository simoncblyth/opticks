#pragma once
#include "Cfg.hh"

template <class Listener>
class CameraCfg : public Cfg {
public:
   CameraCfg(const char* name, Listener* listener) : Cfg(name) 
   {
      addOptionF<Listener>(listener, "yfov",     "Vertical Field of view in degrees");
      addOptionF<Listener>(listener, "near",     "Near distance");
      addOptionF<Listener>(listener, "far",      "Far distance" );
      addOptionF<Listener>(listener, "parallel", "Parallel or perspective");
   } 
};



