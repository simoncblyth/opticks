#pragma once
#include "BCfg.hh"




template <class Listener>
class DemoCfg : public BCfg {
public:
   DemoCfg(const char* name, Listener* listener, bool live) : BCfg(name, live) 
   {
      addOptionF<Listener>(listener, Listener::A, "A");
      addOptionF<Listener>(listener, Listener::B, "B");
      addOptionF<Listener>(listener, Listener::C, "C");
   } 
};



