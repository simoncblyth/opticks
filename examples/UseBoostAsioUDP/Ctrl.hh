#pragma once

#define API  __attribute__ ((visibility ("default")))
class API Ctrl {
   public:
      virtual void command(const char* cmd) = 0 ; 
};



