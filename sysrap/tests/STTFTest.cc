// name=STTFTest ; gcc $name.cc -I.. -std=c++11 -lstdc++ -o /tmp/$name &&  OPTICKS_STTF_PATH=/Library/Fonts/Arial.ttf /tmp/$name  

#include "STTF.hh"

#define SIMG_IMPLEMENTATION 1 
#include "SIMG.hh"

int main(int argc, char** argv)
{
    STTF sttf ; 
    
    int width = 1280;
    int height = 720; 

    unsigned char* data = (unsigned char*)calloc(width * height, sizeof(unsigned char));

    SIMG img(width, height, 1, data ); 

    const char* text = "STTFTest : the quick brown fox jumps over the lazy dog 0.123456789 "  ; 
    int line_height = 32 ; 

    if( 0 == sttf.write_to_bitmap( img.data, img.width, line_height, text ))
    {
        int quality = 50 ; 
        const char* path = "/tmp/out.jpg" ; 
        printf("STTFTest : writing to %s\n", path ); 
        img.writeJPG(path, quality ); 
    }
     
    free(data);

    return 0 ; 
}
