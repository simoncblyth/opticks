#pragma once
#include "Cfg.hh"

template <class Listener>
class TrackballCfg : public Cfg {
public:
   TrackballCfg(const char* name, Listener* listener, bool live) : Cfg(name, live) 
   {
       addOptionF<Listener>(listener, Listener::RADIUS,          "Trackball radius");
       addOptionF<Listener>(listener, Listener::TRANSLATEFACTOR, "Translation factor");

       addOptionS<Listener>(listener, Listener::ORIENTATION,     "Comma delimited theta,phi in degress");
       addOptionS<Listener>(listener, Listener::TRANSLATE,       "Comma delimited x,y,z translation triplet");
   }
};


