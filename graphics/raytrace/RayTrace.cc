#include <stdlib.h>
#include <libgen.h>
#include <sstream>
#include "AOScene.hh"

int main(int argc, char* argv[])
{
    printf("argv0 %s \n", argv[0]);

    GLUTDisplay::init( argc, argv );

    unsigned int width = 1080u, height = 720u;
    const char* key = "DAE_NAME_DYB_NOEXTRA" ; 
    const char* path = getenv(key);

    char* ptxdir = dirname(argv[0]);
    const char* target = "RayTrace" ; 

    std::stringstream title;
    title << "AOScene " << key ;
    try 
    {
        AOScene scene(path, ptxdir, target);
        scene.setDimensions( width, height );
        scene.Info();
        GLUTDisplay::run( title.str(), &scene );
    } 
    catch( optix::Exception& e )
    {
        sutilReportError( e.getErrorString().c_str() );
        exit(1);
    }
    return 0 ; 
}

