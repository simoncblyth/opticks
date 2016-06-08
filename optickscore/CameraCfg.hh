#pragma once
#include "BCfg.hh"

template <class Listener>
class CameraCfg : public BCfg {
public:
   CameraCfg(const char* name, Listener* listener, bool live) : BCfg(name, live) 
   {
       addOptionI<Listener>(listener, Listener::PRINT,    "Print");

       addOptionF<Listener>(listener, Listener::ZOOM,     "Zoom factor");
       addOptionF<Listener>(listener, Listener::SCALE,    "Screen Scale");
       addOptionF<Listener>(listener, Listener::NEAR,     "Near distance");
       addOptionF<Listener>(listener, Listener::FAR,      "Far distance" );
       addOptionF<Listener>(listener, Listener::PARALLEL, "Parallel or perspective");
   }
};


