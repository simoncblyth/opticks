#pragma once
#include "Cfg.hh"

template <class Listener>
class CameraCfg : public Cfg {
public:
   CameraCfg(const char* name, Listener* listener, bool live) : Cfg(name, live) 
   {
       addOptionI<Listener>(listener, Listener::PRINT,    "Print");

       addOptionF<Listener>(listener, Listener::YFOV,     "Vertical Field of view in degrees");
       addOptionF<Listener>(listener, Listener::NEAR,     "Near distance");
       addOptionF<Listener>(listener, Listener::FAR,      "Far distance" );
       addOptionF<Listener>(listener, Listener::PARALLEL, "Parallel or perspective");
   }
};


