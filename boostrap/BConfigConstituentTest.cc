
struct DemoConstituentConfig 
{
    DemoConstituentConfig(const char* cfg);

    struct BConfig* 

    int red ; 
    int green ; 
    int blue ; 
};


struct DemoInheritConfig : BConfig
{
    DemoInheritConfig(const char* cfg);

    int red ; 
    int green ; 
    int blue ; 
};



DemoInheritConfig::DemoInheritConfig(const char* cfg)



DemoInheritConfig::DemoInheritConfig(const char* cfg)
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

    DemoInheritConfig cfg("red=1,green=2,blue=3");
    cfg.dump();


    return 0 ; 
}

