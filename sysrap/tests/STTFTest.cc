
/**
STTFTest.cc
=============

~/o/sysrap/tests/STTFTest.sh

Note that SIMG.h now has annotation methods using 
an internal STTF instance, so most annotation does 
need to directly using STTF.h 


**/

#include <string>
#include <sstream>

#include "spath.h"
#include "ssys.h"

#define STTF_IMPLEMENTATION 1 
#include "STTF.h"

#define SIMG_IMPLEMENTATION 1 
#include "SIMG.h"


std::string formTEXT()
{
    double value = 1.23456789 ; 
    std::stringstream ss ; 
    ss << "the quick brown fox jumps over the lazy dog " ; 
    ss << value  ;
    ss << std::endl ;   // <-- newline renders as an open rectangle  
    ss << "the quick brown fox jumps over the lazy dog " ; 
    ss << value  ;

    std::string text = ss.str();  
    return text ; 
}


int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : spath::Resolve("$TMP/STTFTest/STTFTest.jpg") ; 
    std::string _text = formTEXT(); 
    const char* text = _text.c_str(); 

    STTF* ttf = STTF::Create(); 

    
    int width = 1280;
    int height = 720; 
    int channels = 4 ; 
    int line_height = ssys::getenvint("LINE_HEIGHT", 32 ) ; 
    int offset = 0 ; 

    printf("STTFTest line_height %d \n", line_height ); 

    int magenta[4] = {255,0,255,0} ; 
    int black[4] = {0,0,0,0} ; 

    unsigned char* data = (unsigned char*)calloc(width * height * channels, sizeof(unsigned char));
    ttf->render_background( data,        channels, width, height,      magenta ) ;

    ttf->render_background( data+offset, channels, width, line_height, black ) ;
    ttf->render_text(       data+offset, channels, width, line_height, text ) ;

    offset = width*height*channels/2 ;   
    ttf->render_background( data+offset, channels, width, line_height, black ) ;
    ttf->render_text(       data+offset, channels, width, line_height, text ) ;

    bool lowlevel = false ; 

    if( lowlevel )
    {  
        offset = width*(height-line_height-1)*channels ;    // -1 to avoid stepping off the end and segmenting 
        ttf->render_background( data+offset, channels, width, line_height, black ) ;
        ttf->render_text(       data+offset, channels, width, line_height, text ) ;
    }    
    else
    {
        ttf->annotate( data , channels, width, height, line_height, text, true );  
        ttf->annotate( data , channels, width, height, line_height, text, false );  
    }


   

    SIMG img(width, height, channels, data ); 

    int quality = 50 ; 
    printf("STTFTest : writing to %s quality %d \n", path, quality ); 
    img.writeJPG(path, quality ); 
 
    free(data);

    return 0 ; 
}
