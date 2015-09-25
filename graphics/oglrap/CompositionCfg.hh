#pragma once
#include "Cfg.hh"

template <class Listener>
class CompositionCfg : public Cfg {
public:
   CompositionCfg(const char* name, Listener* listener, bool live) : Cfg(name, live) 
   {
       addOptionI<Listener>(listener, Listener::PRINT,    "Print");
       addOptionS<Listener>(listener, Listener::SELECT,   "Selection, four comma delimited integers");
       addOptionS<Listener>(listener, Listener::PICKPHOTON, 
           "PickPhoton, up to 4 comma delimited integers, eg:\n"
           "10000   : target view at the center extent \n" 
           "10000,1 : as above but hide other records \n" 
      );

       addOptionS<Listener>(listener, Listener::PICKFACE, 
           "PickFace, up to 4 comma delimited integers, eg:\n"
           "10,3158,0  : target face index 10 of solid inde 3158 in mesh index 0 \n" 
      );

   }
};


