#include "Scene.hh"

void test_charstarstar()
{
    unsigned int nmode = 3 ; 
    const char** c = new const char*[nmode] ; 

    c[0] = Scene::PHOTON ; 
    c[1] = Scene::GENSTEP ; 
    c[2] = Scene::GEOMETRY ; 

    for(unsigned int i=0 ; i < nmode ; i++)
    {
        printf("%2d : %s \n", i, c[i] );
    }
}


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


