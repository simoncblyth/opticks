
#include "PLOG.hh"
#include "BConfig.hh"


struct DemoConfig : BConfig
{
    DemoConfig(const char* cfg);

    int red ; 
    int green ; 
    int blue ; 
};

DemoConfig::DemoConfig(const char* cfg)
   : 
   BConfig(cfg),

   red(0),
   green(0),
   blue(0)
{
   addInt("red",   &red); 
   addInt("green", &green); 
   addInt("blue",  &blue); 

   parse();
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    DemoConfig cfg("red=1,green=2,blue=3");
    cfg.dump();


    return 0 ; 
}

