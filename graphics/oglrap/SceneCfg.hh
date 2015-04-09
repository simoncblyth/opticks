#pragma once
#include "Cfg.hh"

template <class Listener>
class SceneCfg : public Cfg {
public:
   SceneCfg(const char* name, Listener* listener, bool live) : Cfg(name, live) 
   {
       addOptionI<Listener>(listener, Listener::PRINT,    "Print");
   }
};


