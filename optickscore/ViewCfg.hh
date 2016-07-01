#pragma once
#include "BCfg.hh"

#include "OKCORE_API_EXPORT.hh"

template <class Listener>
class OKCORE_API ViewCfg : public BCfg {
public:
   ViewCfg(const char* name, Listener* listener, bool live);
};


