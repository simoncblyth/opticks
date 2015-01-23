#ifndef RAYTRACECONFIG
#define RAYTRACECONFIG

#include <stdlib.h>

struct RayTraceConfig 
{
    char path_to_ptx[512];
    char outfile[512];
    unsigned int width  ;
    unsigned int height ;
    int i ;
    int use_glut ;

    RayTraceConfig() 
          :  width(512u) 
          ,  height(384u) 
          ,  use_glut(1)
          ,  i(0)
    {
          outfile[0] = '\0';
          path_to_ptx[0] = '\0';

          const char* ptxdir = "." ;  
          const char* target = "RayTrace" ; 
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


#endif
