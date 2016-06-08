#pragma once
#include "BCfg.hh"

template <class Listener>
class InteractorCfg : public BCfg {
public:
   InteractorCfg(const char* name, Listener* listener, bool live) : BCfg(name, live) 
   {   
       addOptionF<Listener>(listener, Listener::DRAGFACTOR, "Drag factor");
       addOptionI<Listener>(listener, Listener::OPTIXMODE, "OptiX mode");
   }   
};


