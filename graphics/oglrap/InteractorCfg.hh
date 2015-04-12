#pragma once
#include "Cfg.hh"

template <class Listener>
class InteractorCfg : public Cfg {
public:
   InteractorCfg(const char* name, Listener* listener, bool live) : Cfg(name, live) 
   {   
       addOptionF<Listener>(listener, Listener::DRAGFACTOR, "Drag factor");
       addOptionI<Listener>(listener, Listener::OPTIXMODE, "OptiX mode");
   }   
};


