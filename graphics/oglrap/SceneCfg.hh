#pragma once
#include "BCfg.hh"

template <class Listener>
class SceneCfg : public BCfg {
public:
   SceneCfg(const char* name, Listener* listener, bool live) : BCfg(name, live) 
   {   
       addOptionI<Listener>(listener, Listener::TARGET, "Absolute index of target solid, 0 is treated differently corresponding to entire geometry");


   }   
};


