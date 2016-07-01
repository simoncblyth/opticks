#pragma once

#include "BCfg.hh"
#include "OKCORE_API_EXPORT.hh"

template <class Listener>
class OKCORE_API ClipperCfg : public BCfg {
 public:
    ClipperCfg(const char* name, Listener* listener, bool live);
};


