#pragma once
#include "Cfg.hh"

template <class Listener>
class CompositionCfg : public Cfg {
public:
   CompositionCfg(const char* name, Listener* listener, bool live) : Cfg(name, live) 
   {
       addOptionI<Listener>(listener, Listener::PRINT,    "Print");
   }
};


