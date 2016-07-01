#pragma once
#include "BCfg.hh"
#include "OKCORE_API_EXPORT.hh"

template <class Listener>
class OKCORE_API DemoCfg : public BCfg {
public:
   DemoCfg(const char* name, Listener* listener, bool live);
};



