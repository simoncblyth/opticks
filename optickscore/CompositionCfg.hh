#pragma once
#include "BCfg.hh"
#include "OKCORE_API_EXPORT.hh"

template <class Listener>
class OKCORE_API CompositionCfg : public BCfg {
 public:
     CompositionCfg(const char* name, Listener* listener, bool live);
};


