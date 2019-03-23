#pragma once

/**
SCtrl
======

Protocol base used to avoid dependency issue when communicating
between oglrap, opticksgeo and optickscore.

**/


class SCtrl {
   public:
      virtual void command(const char* cmd) = 0 ;
};


