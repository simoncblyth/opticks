#pragma once
#include "Cfg.hh"
#include <string>

template <class Listener>
class numpydelegateCfg : public Cfg {
public:
   numpydelegateCfg(const char* name, Listener* listener) : Cfg(name)
   {
       addOptionS<Listener>(listener, "zmqbackend",  "ZMQ Backend");
       addOptionI<Listener>(listener, "udpport",     "UDP Port");
       addOptionI<Listener>(listener, "npyecho",     "NPY Echo");
   }
};


