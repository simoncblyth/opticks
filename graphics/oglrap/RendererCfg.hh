#pragma once
#include "BCfg.hh"

template <class Listener>
class RendererCfg : public BCfg {
public:
   RendererCfg(const char* name, Listener* listener, bool live) : BCfg(name, live) 
   {
       addOptionI<Listener>(listener, Listener::PRINT,    "Print");
   }
};


