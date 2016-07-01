#include "NGLM.hpp"
#include "Demo.hh"
#include "DemoCfg.hh"


template OKCORE_API void BCfg::addOptionF<Demo>(Demo*, const char*, const char* );
template OKCORE_API void BCfg::addOptionI<Demo>(Demo*, const char*, const char* );
template OKCORE_API void BCfg::addOptionS<Demo>(Demo*, const char*, const char* );






template <class Listener>
DemoCfg<Listener>::DemoCfg(const char* name, Listener* listener, bool live) 
    : 
    BCfg(name, live) 
{
      addOptionF<Listener>(listener, Listener::A, "A");
      addOptionF<Listener>(listener, Listener::B, "B");
      addOptionF<Listener>(listener, Listener::C, "C");
} 



template class OKCORE_API DemoCfg<Demo> ;


