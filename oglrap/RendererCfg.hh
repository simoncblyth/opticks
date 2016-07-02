#pragma once

#include "Renderer.hh"
#include "BCfg.hh"
#include "OGLRAP_API_EXPORT.hh"

template <class Listener>
class OGLRAP_API RendererCfg : public BCfg {
public:
   RendererCfg(const char* name, Listener* listener, bool live) : BCfg(name, live) 
   {
       addOptionI<Listener>(listener, Listener::PRINT,    "Print");
   }
};

template class OGLRAP_API RendererCfg<Renderer> ;

