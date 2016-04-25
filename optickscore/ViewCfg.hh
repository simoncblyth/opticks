#pragma once
#include "Cfg.hh"

template <class Listener>
class ViewCfg : public Cfg {
public:
   ViewCfg(const char* name, Listener* listener, bool live) : Cfg(name, live) 
   {
       addOptionS<Listener>(listener, "eye", "Comma delimited eye position in model-extent coordinates, eg 0,0,-1  ");
       addOptionS<Listener>(listener, "look","Comma delimited look position in model-extent coordinates, eg 0,0,0  ");
       addOptionS<Listener>(listener, "up",  "Comma delimited up direction in model-extent frame, eg 0,1,0 " );
   }
};


