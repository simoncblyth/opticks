#include <iostream>
#include "PLOG.hh"
#include "BRAP_LOG.hh"
#include "BConfig.hh"


struct DemoConfig : BConfig
{
    DemoConfig(const char* cfg);

    int red ; 
    int green ; 
    int blue ; 

    float cyan ; 
};

DemoConfig::DemoConfig(const char* cfg_)
   : 
   BConfig(cfg_),

   red(0),
   green(0),
   blue(0),
   cyan(1.f)
{
   addInt("red",   &red); 
   addInt("green", &green); 
   addInt("blue",  &blue); 
   addFloat("cyan",  &cyan); 

   parse();
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    BRAP_LOG__ ; 

    DemoConfig cfg("red=1,green=2,blue=3,cyan=1.5");
    cfg.dump();


    std::cout << " cyan " << cfg.cyan << std::endl ; 

    return 0 ; 
}

