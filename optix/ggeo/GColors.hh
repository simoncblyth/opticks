#pragma once

#include "GVector.hh"

#include <map>
#include <string>
#include <vector>

class GBuffer ; 

template <typename T> class NPY ; 


/*

*GColors* associates color names to RGB hexcodes

Usage from GLoader::

    m_colors = GColors::load("$HOME/.opticks","GColors.json");

simon:.opticks blyth$ cat GColors.json | tr "," "\n"
{"indigo": "#4B0082"
 "gold": "#FFD700"
 "hotpink": "#FF69B4"
 "firebrick": "#B22222"
 "indianred": "#CD5C5C"
 "yellow": "#FFFF00"
 "mistyrose": "#FFE4E1"
 "darkolivegreen": "#556B2F"
 "olive": "#808000"


The codes are obtained from matplotlib::

    import json, matplotlib.colors  
    json.dump(matplotlib.colors.cnames, open("/tmp/colors.json", "w"))


The composite color buffer is avaiable as a texture sampler on the GPU
this is put togther in GLoader::load


*/

class GColors {  
   public:
        enum {
           MATERIAL_COLOR_OFFSET    = 0,  
           FLAG_COLOR_OFFSET        = 64,
           PSYCHEDELIC_COLOR_OFFSET = 96,
           SPECTRAL_COLOR_OFFSET    = 256,
           COLORMAX                 = 256
       };
   public:
       static const char* NAME ; 
       static GColors* load(const char* dir, const char* name=NAME);
   public:
       GColors();

       void sort();
       void dump(const char* msg="GColors::dump");
       void test(const char* msg="GColors::test");
       bool operator() (const std::string& a, const std::string& b);

       static gfloat3 makeColor( unsigned int rgb );

       const char* getNamePsychedelic(unsigned int index);
       unsigned int getCode(const char* name, unsigned int missing=0xFFFFFF);
       const char* getHex( const char* name, const char* missing=NULL);
       const char* getName(const char* hex_, const char* missing=NULL);
       unsigned int parseHex(const char* hex_);

       gfloat3 getColor(const char* name, unsigned int missing=0xFFFFFF);
       gfloat3 getPsychedelic(unsigned int num);
       std::vector<unsigned int>& getPsychedelicCodes() ;
       std::vector<unsigned int>& getSpectralCodes() ;

       unsigned int getNumColors();
       unsigned int getNumBytes();
       unsigned int getBufferEntry(unsigned char* colors);
       GBuffer* make_uchar4_buffer();
       GBuffer* make_uchar4_buffer(std::vector<unsigned int>& codes);
       void dump_uchar4_buffer( GBuffer* buffer );
   public: 
       NPY<unsigned char>* make_buffer();
       NPY<unsigned char>* make_buffer(std::vector<unsigned int>& codes);
   public: 
       void setupCompositeColorBuffer(std::vector<unsigned int>&  material_codes, std::vector<unsigned int>& flag_codes);
       GBuffer* getCompositeBuffer();
       NPY<unsigned char>* getCompositeBuffer_();
       guint4   getCompositeDomain();
       void dumpCompositeBuffer(const char* msg="GColors::dumpCompositeBuffer");

   private:
       void initCompositeColorBuffer(unsigned int max_colors);
       void addColors(std::vector<unsigned int>& codes, unsigned int offset=0 );
       void loadMaps(const char* dir);
       //void make_spectral_codes();

   private:
       std::vector<std::string>            m_psychedelic_names ; 
       std::vector<unsigned int>           m_psychedelic_codes ;
       std::vector<unsigned int>           m_spectral_codes ;
       std::map<std::string, std::string>  m_name2hex ; 
       GBuffer*                            m_composite ; 
       NPY<unsigned char>*                 m_composite_ ; 
       guint4                              m_composite_domain ; 

};

inline GColors::GColors()  
    :
    m_composite(NULL),
    m_composite_(NULL),
    m_composite_domain(0,0,0,0)
{
}
inline GBuffer* GColors::getCompositeBuffer()
{
    return m_composite ;  
}

inline NPY<unsigned char>* GColors::getCompositeBuffer_()
{
    return m_composite_ ;  
}

