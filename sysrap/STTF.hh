#pragma once
/**
STTF.hh
=========

Based on:

* https://github.com/justinmeiners/stb-truetype-example
* https://github.com/nothings/stb
* https://github.com/nothings/stb/blob/master/stb_truetype.h
* http://nothings.org/stb/stb_truetype.h 

**/

#include <stdio.h>
#include <stdlib.h>
#include "OKConf.hh"

struct STTF
{
    static const char* KEY ; 
    static const char* GetFontPath(); 
    static unsigned char* Load(const char* path); 

    const    char* fontPath ;
    unsigned char* fontBuffer ;
    void* font_ ;   // stbtt_fontinfo*   good to keep foreign types out of definition when easy to do so 
    bool  valid ; 

    STTF() ;   
    virtual ~STTF(); 
 
    void init(); 
    int  render_background( unsigned char* bitmap, int channels, int width, int height,      int* color );
    int  render_text(       unsigned char* bitmap, int channels, int width, int line_height, const char* text );

    int   annotate(          unsigned char* bitmap, int channels, int width, int height, int line_height, const char* text, bool bottom );  

};




#ifdef __clang__


#elif defined(__GNUC__) || defined(__GNUG__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"

#elif defined(_MSC_VER)

#endif


#ifdef STTF_IMPLEMENTATION

#define STB_TRUETYPE_IMPLEMENTATION 
#include "stb_truetype.h" 

#endif

#ifdef __clang__
#elif defined(__GNUC__) || defined(__GNUG__)

#pragma GCC diagnostic pop

#elif defined(_MSC_VER)
#endif






const char* STTF::KEY = "OPTICKS_STTF_PATH" ; 

inline const char* STTF::GetFontPath() // static
{
    const char* dpath = OKConf::DefaultSTTFPath() ;
    const char* epath = getenv(KEY) ; 
    //printf("STTF::GetFontPath dpath %s epath %s \n", ( dpath ? dpath : "" ), ( epath ? epath : "" ) );    
    return epath ? epath : dpath ; 
}

inline unsigned char* STTF::Load(const char* path) // static 
{
    if(path == nullptr)
    {
        printf("STTF::Load : Envvar %s with path to ttf font file is required \n", KEY);
        return nullptr ; 
    }

#ifdef DEBUG
    printf("STTF::Load font from %s\n", path );  
#endif

    long size ;
    FILE* fontFile = fopen(path, "rb");
    fseek(fontFile, 0, SEEK_END);
    size = ftell(fontFile); /* how long is the file ? */
    fseek(fontFile, 0, SEEK_SET); /* reset */
    
    unsigned char* buffer = (unsigned char*)malloc(size);
    
    fread(buffer, size, 1, fontFile);
    fclose(fontFile);

    return buffer ; 
}



inline STTF::STTF()
    :
    fontPath(GetFontPath()),
    fontBuffer(Load(fontPath)),
    font_(nullptr),
    valid(false)
{
    init();
}



inline void STTF::init()
{
    if(fontBuffer == nullptr)
    {
        printf("STTF::init failed : no font file has been loaded \n");
        return ; 
    }

    stbtt_fontinfo* font = new stbtt_fontinfo ; 
    if (!stbtt_InitFont(font, fontBuffer, 0))
    {
        printf("STTF::init failed : loaded font file is not a valid TTF font ? \n");
        return ; 
    }

    font_ = (void*)font ; 
    valid = true ; 
}


STTF::~STTF()
{
    stbtt_fontinfo* font = (stbtt_fontinfo*)font_ ; 
    delete font ; 
    free(fontBuffer);
}


inline int STTF::render_background( unsigned char* bitmap, int channels, int width, int line_height, int* color )
{
    for(int y=0 ; y < line_height ; y++ ) 
    {
        for(int x=0 ; x < width ; x++) 
        {
            for(int c = 0 ; c < channels ; c++ )
            {
                bitmap[ (y*width + x)*channels + c] = color[c] ; 
            } 
        }  
    }
    return 0 ; 
}

inline int STTF::render_text( unsigned char* bitmap, int channels, int width, int line_height, const char* text )
{
    if(!valid) return 1 ; 
    stbtt_fontinfo* font = (stbtt_fontinfo*)font_ ; 

#ifdef DEBUG
    printf("STTF::render_text channels %d \n", channels ); 
#endif

    float pixels = float(line_height);   
    float scale = stbtt_ScaleForPixelHeight(font, pixels);
    // computes a scale factor to produce a font whose "height" is 'pixels' tall.
    // Height is measured as the distance from the highest ascender to the lowest
    // descender; in other words, it's equivalent to calling stbtt_GetFontVMetrics
    // and computing:
    //       scale = pixels / (ascent - descent)
    // so if you prefer to measure height by the ascent only, use a similar calculation.
    
       
    int ascent, descent, lineGap;
    stbtt_GetFontVMetrics(font, &ascent, &descent, &lineGap);
    // ascent: coordinate above the baseline the font extends; 
    // descent: coordinate below the baseline the font extends (i.e. it is typically negative)
    // lineGap: spacing between one row's descent and the next row's ascent...
    // so you should advance the vertical position by "*ascent - *descent + *lineGap"
    // these are expressed in unscaled coordinates, so you must multiply by
    // the scale factor for a given size
    
    ascent = roundf(ascent * scale);
    descent = roundf(descent * scale);
    
    int x = 0;

    // terminating with "x+pixels < width"  prevents overlong text wrapping around ontop of itself
    while(*text && x + pixels < width)   
    {
        int codepoint = *text ; 

        int ax;  // advanceWidth is the offset from the current horizontal position to the next horizontal position
		int lsb; // leftSideBearing is the offset from the current horizontal position to the left edge of the character
        stbtt_GetCodepointHMetrics(font, codepoint, &ax, &lsb);   // expressed in unscaled coordinates
        ax  = roundf(ax*scale) ; 
        lsb = roundf(lsb*scale) ; 


        int ix0,  iy0,  ix1,  iy1;
        stbtt_GetCodepointBitmapBox(font, codepoint, scale, scale, &ix0, &iy0, &ix1, &iy1);
        // get the bbox of the bitmap centered around the glyph origin; so the
        // bitmap width is ix1-ix0, height is iy1-iy0, and location to place
        // the bitmap top left is (leftSideBearing*scale,iy0).
        // (Note that the bitmap uses y-increases-down, but the shape uses
        // y-increases-up, so CodepointBitmapBox and CodepointBox are inverted.)

        int y = ascent + iy0 ;
        int offset = x + lsb + (y * width);

        unsigned char* output = bitmap + offset*channels ;
        int out_w = (ix1-ix0)*channels ;        
        int out_h = (iy1-iy0) ;   // <-- when multiply by channels get black boxes at y positions below the text  
        int out_stride = width*channels ;  

        float scale_x = scale*channels ;    // adhoc ? why is this needed 
        float scale_y = scale ;  

        stbtt_MakeCodepointBitmap(font, output, out_w, out_h, out_stride, scale_x, scale_y, codepoint );
        // the same as stbtt_GetCodepointBitmap, but you pass in storage for the bitmap
        // in the form of 'output', with row spacing of 'out_stride' bytes. the bitmap
        // is clipped to out_w/out_h bytes. Call stbtt_GetCodepointBitmapBox to get the
        // width and height and positioning info for it first.
        
        int kern;
        kern = stbtt_GetCodepointKernAdvance(font, *text, *(text+1));
        kern = roundf(kern * scale) ; 

        x += ax + kern ;

        //printf("[%c]%d\n", *text, x ); 
        text++ ; 
    }
    return 0 ; 
}


inline int STTF::annotate( unsigned char* bitmap, int channels, int width, int height, int line_height, const char* text, bool bottom )
{
    int rc = 1 ; 
    if(!valid) return rc ; 


    // black band with text annotation at base of image 
    int black[4] = {0,0,0,0} ;   // any color, so long as its black 
   
    int margin_bkg = 0 ; 
    int margin_txt = 1 ; 

    int offset_bkg = bottom ? width*(height-line_height-margin_bkg)*channels : 0 ;      
    int offset_txt = bottom ? width*(height-line_height-margin_txt)*channels : 0 ;      

    rc = render_background( bitmap+offset_bkg, channels, width, line_height, black ) ;  
    rc = render_text(       bitmap+offset_txt, channels, width, line_height, text  ) ;  
    return rc ; 
}

