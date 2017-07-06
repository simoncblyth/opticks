
#include "PLOG.hh"
#include "BConfig.hh"


struct DemoConfig 
{
    DemoConfig(const char* cfg);

    struct BConfig* bconfig ; 
    void dump() const  ;
    
    int red ; 
    int green ; 
    int blue ; 
};

DemoConfig::DemoConfig(const char* cfg)
   : 
   bconfig(new BConfig(cfg)),

   red(0),
   green(0),
   blue(0)
{
   bconfig->addInt("red",   &red); 
   bconfig->addInt("green", &green); 
   bconfig->addInt("blue",  &blue); 

   bconfig->parse();
}


void DemoConfig::dump() const 
{
    bconfig->dump();
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    DemoConfig cfg("red=1,green=2,blue=3");
    cfg.dump();


    return 0 ; 
}

