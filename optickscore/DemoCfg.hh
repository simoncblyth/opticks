#pragma once
#include "Cfg.hh"




template <class Listener>
class DemoCfg : public Cfg {
public:
   DemoCfg(const char* name, Listener* listener, bool live) : Cfg(name, live) 
   {
      addOptionF<Listener>(listener, Listener::A, "A");
      addOptionF<Listener>(listener, Listener::B, "B");
      addOptionF<Listener>(listener, Listener::C, "C");
   } 
};



