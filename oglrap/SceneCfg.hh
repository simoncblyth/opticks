#pragma once
#include "Scene.hh"
#include "BCfg.hh"
#include "OGLRAP_API_EXPORT.hh"

template <class Listener>
class OGLRAP_API SceneCfg : public BCfg {
public:
   SceneCfg(const char* name, Listener* listener, bool live) : BCfg(name, live) 
   {   
       addOptionI<Listener>(listener, Listener::TARGET, "Absolute index of target solid, 0 is treated differently corresponding to entire geometry");


   }   
};

template class OGLRAP_API SceneCfg<Scene> ;

