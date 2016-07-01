#pragma once
#include "BCfg.hh"

#include "OKCORE_API_EXPORT.hh"

template <class Listener>
class OKCORE_API CameraCfg : public BCfg {
  public:
   CameraCfg(const char* name, Listener* listener, bool live);
};


