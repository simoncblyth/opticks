#pragma once
#include "Cfg.hh"

template <class Listener>
class TrackballCfg : public Cfg {
public:
   TrackballCfg(const char* name, Listener* listener, bool live) : Cfg(name, live) 
   {
       addOptionF<Listener>(listener, "radius",          "Trackball radius");
       addOptionF<Listener>(listener, "translatefactor", "Translation factor");

       addOptionS<Listener>(listener, "orientation",     "Comma delimited theta,phi in degress");
       addOptionS<Listener>(listener, "translate",       "Comma delimited x,y,z translation triplet");
   }
};


