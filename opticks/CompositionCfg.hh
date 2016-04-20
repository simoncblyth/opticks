#pragma once
#include "Cfg.hh"

template <class Listener>
class CompositionCfg : public Cfg {
public:
   CompositionCfg(const char* name, Listener* listener, bool live) : Cfg(name, live) 
   {
       addOptionI<Listener>(listener, Listener::PRINT,    "Print");
       addOptionS<Listener>(listener, Listener::SELECT,   "Selection, four comma delimited integers");

       addOptionS<Listener>(listener, Listener::EYEW,    "Three Comma delimited world space eye coordinates, eg 300,300,0 " );
       addOptionS<Listener>(listener, Listener::LOOKW,   "Three Comma delimited world space look coordinates, eg 0,0,300 " );


       addOptionS<Listener>(listener, Listener::PICKPHOTON, 
           "[UDP only], up to 4 comma delimited integers, eg:\n"
           "10000   : target view at the center extent \n" 
           "10000,1 : as above but hide other records \n" 
           "\n"
           "see CompositionCfg.hh\n"
      );


   }
};


