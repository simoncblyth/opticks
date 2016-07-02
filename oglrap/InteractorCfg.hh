#pragma once
#include "Interactor.hh"
#include "BCfg.hh"
#include "OGLRAP_API_EXPORT.hh"

template <class Listener>
class OGLRAP_API InteractorCfg : public BCfg {
public:
   InteractorCfg(const char* name, Listener* listener, bool live) : BCfg(name, live) 
   {   
       addOptionF<Listener>(listener, Listener::DRAGFACTOR, "Drag factor");
       addOptionI<Listener>(listener, Listener::OPTIXMODE, "OptiX mode");
   }   
};


template class OGLRAP_API InteractorCfg<Interactor> ;

