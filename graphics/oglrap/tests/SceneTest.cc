#include "Scene.hh"
#include <cstdio>


void test_mode_gymnastics()
{
#ifdef MODE_GYMNASTICS
    Scene scene ; 
    scene.dumpModes("default");

    scene.setMode(0, false);
    scene.dumpModes("after disable 0");

    scene.setMode(1, false);
    scene.dumpModes("after disable 1");

    scene.setMode(2, false);
    scene.dumpModes("after disable 2");
#endif
}



int main()
{
    return 0 ; 
}


