#pragma once

#include "BCfg.hh"
#include "OKCORE_API_EXPORT.hh"

template <class Listener>
class OKCORE_API TrackballCfg : public BCfg {
public:
   TrackballCfg(const char* name, Listener* listener, bool live);
};


