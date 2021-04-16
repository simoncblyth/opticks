#pragma once
/**
STTF.hh
=========

Based on https://github.com/justinmeiners/stb-truetype-example

**/

#include <stdio.h>
#include <stdlib.h>

#define STB_TRUETYPE_IMPLEMENTATION 
#include "stb_truetype.h" /* http://nothings.org/stb/stb_truetype.h */


struct STTF
{
    static const char* KEY ; 
    unsigned char* fontBuffer ;
    stbtt_fontinfo info ;

    STTF() ;   
    void load_ttf(const char* path); 
    virtual ~STTF(); 
 
    int write_to_bitmap( unsigned char* bitmap, int line_width, int line_height, const char* text );
};


const char* STTF::KEY = "OPTICKS_STTF_PATH" ; 

inline STTF::STTF()
    :
    fontBuffer(nullptr)
{
    char* path = getenv(KEY) ; 
    load_ttf(path); 
}

void STTF::load_ttf(const char* path)
{
    if(path == nullptr)
    {
        printf("STTF::load_ttf : Envvar %s with path to ttf font file is required \n", KEY);
        return ; 
    }

    long size ;
    FILE* fontFile = fopen(path, "rb");
    fseek(fontFile, 0, SEEK_END);
    size = ftell(fontFile); /* how long is the file ? */
    fseek(fontFile, 0, SEEK_SET); /* reset */
    
    fontBuffer = (unsigned char*)malloc(size);
    
    fread(fontBuffer, size, 1, fontFile);
    fclose(fontFile);

    /* prepare font */
    if (!stbtt_InitFont(&info, fontBuffer, 0))
    {
        printf("STTF::load_ttf failed\n");
    }
}


STTF::~STTF()
{
    free(fontBuffer);
}

inline int STTF::write_to_bitmap( unsigned char* bitmap, int line_width, int line_height, const char* text )
{
    if(fontBuffer == nullptr) return 1 ; 

    float scale = stbtt_ScaleForPixelHeight(&info, line_height );
    
    int x = 0;
       
    int ascent, descent, lineGap;
    stbtt_GetFontVMetrics(&info, &ascent, &descent, &lineGap);
    
    ascent = roundf(ascent * scale);
    descent = roundf(descent * scale);
    
    int i;
    for (i = 0; i < strlen(text); ++i)
    {
        /* how wide is this character */
        int ax;
		int lsb;
        stbtt_GetCodepointHMetrics(&info, text[i], &ax, &lsb);

        /* get bounding box for character (may be offset to account for chars that dip above or below the line */
        int c_x1, c_y1, c_x2, c_y2;
        stbtt_GetCodepointBitmapBox(&info, text[i], scale, scale, &c_x1, &c_y1, &c_x2, &c_y2);
        
        /* compute y (different characters have different heights */
        int y = ascent + c_y1;
        
        /* render character (stride and offset is important here) */
        int byteOffset = x + roundf(lsb * scale) + (y * line_width);
        stbtt_MakeCodepointBitmap(&info, bitmap + byteOffset, c_x2 - c_x1, c_y2 - c_y1, line_width , scale, scale, text[i]);

        /* advance x */
        x += roundf(ax * scale);
        
        /* add kerning */
        int kern;
        kern = stbtt_GetCodepointKernAdvance(&info, text[i], text[i + 1]);
        x += roundf(kern * scale);
    }
    return 0 ; 
}



