#include <stdlib.h>
#include <libgen.h>
#include <sstream>

#include "OptiXAssimpGeometry.hh"
#include "OptiXProgram.hh"
#include "OptiXScene.hh"

int main(int argc, char* argv[])
{
    const char* query = getenv("RAYTRACE_QUERY");
    if(!query) query = "__dd__Geometry__AD__lvOIL0xbf5e0b8" ;
    printf("argv0 %s query %s \n", argv[0], query );

    GLUTDisplay::init( argc, argv );

    unsigned int width = 1080u, height = 720u;

    const char* key = "DAE_NAME_DYB_NOEXTRA" ; 
    const char* path = getenv(key);

    std::stringstream title;
    title << "RayTrace " << key ;
    try 
    {
        OptiXScene scene;
        optix::Context context = scene.getContext();

        context->setRayTypeCount( 1 );
        context->setEntryPointCount( 1 );
        context->setStackSize( 4640 );

        char* ptxdir = dirname(argv[0]);     // alongside the executable
        OptiXProgram prog(ptxdir, "RayTrace");  // cmake target name
        prog.setContext(context);


        optix::GeometryGroup gg = context->createGeometryGroup();
        OptiXAssimpGeometry geom(path);
        geom.import();
        geom.setGeometryGroup(gg);

        // must setContext and setProgram before convert 
        geom.setContext(context);
        geom.setProgram(&prog);
        geom.convert(query); 
        geom.setupAcceleration();

        scene.setProgram(&prog);
        scene.setDimensions( width, height );
        scene.setGeometry(&geom);

        printf("calling GLUTDisplay::run \n");
 
        GLUTDisplay::run( title.str(), &scene );
    } 
    catch( optix::Exception& e )
    {
        sutilReportError( e.getErrorString().c_str() );
        exit(1);
    }
    return 0 ; 
}

