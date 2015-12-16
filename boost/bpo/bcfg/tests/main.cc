#include "CameraCfg.hh"
#include "Camera.hh"

int main(int argc, char** argv)
{
    Camera camera ; 
    camera.Print("initial default camera");

    CameraCfg<Camera> cfg("camera", &camera);

    cfg.configfile("demo.cfg");
    camera.Print("after configfile");

    cfg.commandline(argc,argv);
    camera.Print("after commandline");

    cfg.liveline("--yfov 123 --near 100 --far 1000 --parallel 1");
    camera.Print("after liveline");

};


