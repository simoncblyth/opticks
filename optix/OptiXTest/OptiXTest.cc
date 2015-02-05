#include <optix.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sutil.h>

#include "OptiXTestConfig.h"

int main(int argc, char* argv[])
{
    OptiXTestConfig cfg(argc, argv );

    RTcontext context;
    RT_CHECK_ERROR( rtContextCreate( &context ) );
    RT_CHECK_ERROR( rtContextSetRayTypeCount( context, 1 ) );
    RT_CHECK_ERROR( rtContextSetEntryPointCount( context, 1 ) );

    RTbuffer  buffer;
    RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_OUTPUT, &buffer ) );
    RT_CHECK_ERROR( rtBufferSetFormat( buffer, RT_FORMAT_FLOAT4 ) );
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, cfg.width, cfg.height ) );

    RTvariable result_buffer;
    RT_CHECK_ERROR( rtContextDeclareVariable( context, "result_buffer", &result_buffer ) );
    RT_CHECK_ERROR( rtVariableSetObject( result_buffer, buffer ) );

    RTprogram program;
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, cfg.path_to_ptx, "draw_solid_color", &program ) );
    RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, 0, program ) );

    RTvariable draw_color;
    RT_CHECK_ERROR( rtProgramDeclareVariable( program, "draw_color", &draw_color ) );
    RT_CHECK_ERROR( rtVariableSet3f( draw_color, 0.962f, 0.725f, 0.0f ) );

    // validate, compile, launch 
    RT_CHECK_ERROR( rtContextValidate( context ) );
    RT_CHECK_ERROR( rtContextCompile( context ) );
    RT_CHECK_ERROR( rtContextLaunch2D( context, 0, cfg.width, cfg.height ) );

    // display
    if( strlen( cfg.outfile ) == 0 ) 
    {
        RT_CHECK_ERROR( sutilDisplayBufferInGlutWindow( argv[0], buffer ) );
    } 
    else 
    {
        RT_CHECK_ERROR( sutilDisplayFilePPM( cfg.outfile, buffer ) );
    }

    RT_CHECK_ERROR( rtBufferDestroy( buffer ) );
    RT_CHECK_ERROR( rtProgramDestroy( program ) );
    RT_CHECK_ERROR( rtContextDestroy( context ) );

    return 0 ;
}




