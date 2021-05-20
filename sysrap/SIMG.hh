#pragma once

#include <string>
#include <cstring>
#include <sstream>
#include <iostream>
#include <cassert>

#ifdef WITH_PLOG
#include <plog/Severity.h>
#include "PLOG.hh"
#else
#define LOG(severity) if(false) std::cout    // just kill the logging when no PLOG
#endif


struct SIMG 
{
#ifdef WITH_PLOG
   static const plog::Severity LEVEL ; 
#endif

   int width ; 
   int height ; 
   int channels ; 
   unsigned char* data ; 
   const char* loadpath ; 
   const char* loadext ; 
   const bool owned ; 

   SIMG(const char* path, int desired_channels=0);   // 0:asis channels 
   SIMG(int width_, int height_, int channels_, unsigned char* data_ ); 
   void setData(unsigned char* data_) ; 
   virtual ~SIMG();  

   std::string desc() const ; 

   void annotate( const char* bottom_line=nullptr, const char* top_line=nullptr, int line_height=24 ) ;

   void writePNG() const ; 
   void writeJPG(int quality) const ; 

   void writePNG(const char* path) const ; 
   void writeJPG(const char* path, int quality) const ; 

   void writePNG(const char* dir, const char* name) const ; 
   void writeJPG(const char* dir, const char* name, int quality) const ; 

   static std::string FormPath(const char* dir, const char* name);
   static bool EndsWith( const char* s, const char* q);
   static const char* ChangeExt( const char* s, const char* x1, const char* x2);
   static const char* Ext(const char* path);
};


#ifdef __clang__


#elif defined(__GNUC__) || defined(__GNUG__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"

#elif defined(_MSC_VER)

#endif


#ifdef SIMG_IMPLEMENTATION

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#endif

#ifdef __clang__
#elif defined(__GNUC__) || defined(__GNUG__)

#pragma GCC diagnostic pop

#elif defined(_MSC_VER)
#endif




#define STTF_IMPLEMENTATION 1 
#include "STTF.hh"


#ifdef WITH_PLOG
const plog::Severity SIMG::LEVEL = PLOG::EnvLevel("SIMG", "DEBUG"); 
#endif


inline bool SIMG::EndsWith( const char* s, const char* q) // static 
{
    int pos = strlen(s) - strlen(q) ;
    return pos > 0 && strncmp(s + pos, q, strlen(q)) == 0 ; 
}

inline const char* SIMG::ChangeExt( const char* s, const char* x1, const char* x2)  // static
{
    assert( EndsWith(s, x1) );  

    std::string st = s ; 
    std::stringstream ss ; 

    ss << st.substr(0, strlen(s) - strlen(x1) ) ; 
    ss << x2 ;   
    std::string ns = ss.str() ; 
    return strdup(ns.c_str()); 
}


inline SIMG::SIMG(const char* path, int desired_channels) 
    :
    width(0),
    height(0),
    channels(0),
    data(stbi_load(path, &width, &height, &channels, desired_channels)),    
    loadpath(strdup(path)),
    loadext(Ext(loadpath)),
    owned(true)
{
}

inline SIMG::SIMG(int width_, int height_, int channels_, unsigned char* data_) 
    :
    width(width_),
    height(height_),
    channels(channels_),
    data(data_),
    loadpath("image.ppm"),
    loadext(Ext(loadpath)),
    owned(false)
{
}

inline void SIMG::setData(unsigned char* data_)
{
    data = data_ ; 
}


inline SIMG::~SIMG()
{
    // getting linker error with the below when using in CMake project, but not in standalone testing 
    //if(owned) stbi_image_free((void*)data);    
}

inline std::string SIMG::desc() const
{
    std::stringstream ss ;
    ss << "SIMG"
       << " width " << width 
       << " height " << height
       << " channels " << channels
       << " loadpath " << ( loadpath ? loadpath : "-" )
       << " loadext " << ( loadext ? loadext : "-" )
       ;
    std::string s = ss.str(); 
    return s ;
}

inline std::string SIMG::FormPath(const char* dir, const char* name)
{
    std::stringstream ss ;
    ss << dir << "/" << name ; 
    std::string s = ss.str(); 
    return s ; 
}

inline void SIMG::writePNG() const 
{
    assert(loadpath && loadext); 
    const char* pngpath = ChangeExt(loadpath, loadext, ".png" ); 
    if(strcmp(pngpath, loadpath)==0)
    {
        LOG(error) << "ERROR cannot overwrite loadpath " << loadpath ; 
    } 
    else
    {
         writePNG(pngpath);  
    }
}

inline void SIMG::writeJPG(int quality) const 
{
    assert(loadpath && loadext); 

    std::stringstream ss ; 
    ss << "_" << quality << ".jpg" ; 
    std::string x = ss.str(); 

    const char* jpgpath = ChangeExt(loadpath, loadext, x.c_str() ); 

    if(strcmp(jpgpath, loadpath)==0)
    {
        LOG(error) << "ERROR cannot overwrite loadpath " << loadpath ; 
    } 
    else
    {
        writeJPG(jpgpath, quality);  
    }
}


inline void SIMG::writePNG(const char* dir, const char* name) const 
{
    std::string s = FormPath(dir, name); 
    writePNG(s.c_str()); 
}
inline void SIMG::writePNG(const char* path) const 
{
    //LOG(LEVEL) << "stbi_write_png " << path ; 
    stbi_write_png(path, width, height, channels, data, width * channels);
}


inline void SIMG::writeJPG(const char* dir, const char* name, int quality) const 
{
    std::string s = FormPath(dir, name); 
    writeJPG(s.c_str(), quality); 
}
inline void SIMG::writeJPG(const char* path, int quality) const 
{
    //LOG(LEVEL) << "stbi_write_jpg " << path << " quality " << quality  ; 
    assert( quality > 0 && quality <= 100 ); 
    stbi_write_jpg(path, width, height, channels, data, quality );
}


inline const char* SIMG::Ext(const char* path)
{
    std::string s = path ; 
    std::size_t pos = s.find_last_of(".");
    std::string ext = s.substr(pos) ;  
    return strdup(ext.c_str()); 
}





void SIMG::annotate( const char* bottom_line, const char* top_line, int line_height )
{
    // Accessing ttf in the ctor rather than doing it here at point of use turns out to be flakey somehow ?
    // Possibly related to this being implemented in the header ?

#ifdef WITH_PLOG
    STTF* ttf = PLOG::instance ? PLOG::instance->ttf : nullptr ; 
#else
    STTF* ttf = nullptr ; 
#endif

    if( ttf == nullptr )
    {
        LOG(error) << "ttf NULL : cannot annotate  " ; 
        return ; 
 
    }
    if(!ttf->valid || line_height > int(height)) 
    {
        LOG(error) << "ttf invalid OR line_height too large " ; 
        return ; 
    } 

    if( top_line )
        ttf->annotate( data, int(channels), int(width), int(height), line_height, top_line, false );  

    if( bottom_line )
        ttf->annotate( data, int(channels), int(width), int(height), line_height, bottom_line, true );  
}






