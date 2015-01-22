#include <optix.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sutil.h>


struct Config 
{
    char path_to_ptx[512];
    char outfile[512];
    unsigned int width  ;
    unsigned int height ;
    int i ;
    int use_glut ;

    Config() 
          :  width(512u) 
          ,  height(384u) 
          ,  use_glut(1)
          ,  i(0)
    {
          outfile[0] = '\0';
          path_to_ptx[0] = '\0';

          const char* ptxdir = "." ;  
          const char* target = "OptiXTest" ; 
          const char* cu = "draw_color.cu" ; 
          sprintf( path_to_ptx, "%s/%s_generated_%s.ptx", ptxdir, target, cu );
          printf("path_to_ptx %s \n", path_to_ptx );
    }


    void ParseArgs( int argc, char* argv[] )
    {

        /* If "--file" is specified, don't do any GL stuff */
        for( i = 1; i < argc; ++i ) 
        {
            if( strcmp( argv[i], "--file" ) == 0 || strcmp( argv[i], "-f" ) == 0 ) use_glut = 0;
        }

        /* Process command line args */
        if(use_glut)
        {
            RT_CHECK_ERROR_NO_CONTEXT( sutilInitGlut( &argc, argv ) );
        } 

        for( i = 1; i < argc; ++i ) 
        {
            if( strcmp( argv[i], "--help" ) == 0 || strcmp( argv[i], "-h" ) == 0 ) 
            {
                printUsageAndExit( argv[0] );
            } 
            else if( strcmp( argv[i], "--file" ) == 0 || strcmp( argv[i], "-f" ) == 0 ) 
            {
                if( i < argc-1 ) 
                {
                    strcpy( outfile, argv[++i] );
                } 
                else 
                {
                    printUsageAndExit( argv[0] );
                }
            } 
            else if ( strncmp( argv[i], "--dim=", 6 ) == 0 ) 
            {
                const char *dims_arg = &argv[i][6];
                if ( sutilParseImageDimensions( dims_arg, &width, &height ) != RT_SUCCESS ) 
                {
                    fprintf( stderr, "Invalid window dimensions: '%s'\n", dims_arg );
                    printUsageAndExit( argv[0] );
                }
            } 
            else 
            {
                fprintf( stderr, "Unknown option '%s'\n", argv[i] );
                printUsageAndExit( argv[0] );
            }
        }
    }

    void printUsageAndExit( const char* argv0 )
    {
      fprintf( stderr, "Usage  : %s [options]\n", argv0 );
      fprintf( stderr, "Options: --file | -f <filename>      Specify file for image output\n" );
      fprintf( stderr, "         --help | -h                 Print this usage message\n" );
      fprintf( stderr, "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n" );
      exit(1);
    }


};



int main(int argc, char* argv[])
{
    Config cfg ; 
    cfg.ParseArgs(argc, argv );

    unsigned int num_devices;
    unsigned int version;

    RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetDeviceCount(&num_devices));
    RT_CHECK_ERROR_NO_CONTEXT(rtGetVersion(&version));
    printf("OptiX %d.%d.%d\n", version/1000, (version%1000)/10, version%10);
    printf("Number of Devices = %d\n\n", num_devices);

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
    RT_CHECK_ERROR( rtContextLaunch2D( context, 0 /*entry point*/, cfg.width, cfg.height ) );


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





