#pragma once

#include <sstream>
#include <iostream>

static void SPPM_write( const char* dir, const char* name, const uchar4* image, int width, int height, bool yflip )
{
    std::stringstream ss ; 
    ss << dir << "/" << name ; 
    std::string s = ss.str(); 
    const char* filename = s.c_str(); 

    FILE * fp; 
    fp = fopen(filename, "wb");

    fprintf(fp, "P6\n%d %d\n%d\n", width, height, 255);

    unsigned size = height*width*3 ; 
    unsigned char* data = new unsigned char[size] ; 

    for( int h=0; h < height ; h++ ) // flip vertically
    {   
        int y = yflip ? height - 1 - h : h ; 

        for( int x=0; x < width ; ++x ) 
        {
            *(data + (y*width+x)*3+0) = image[(h*width+x)].x ;   
            *(data + (y*width+x)*3+1) = image[(h*width+x)].y ;   
            *(data + (y*width+x)*3+2) = image[(h*width+x)].z ;   
        }
    }   
    fwrite(data, sizeof(unsigned char)*size, 1, fp);
    fclose(fp);  
    //std::cout << "Wrote file (uchar4) " << filename << std::endl  ;
    delete[] data;
}


