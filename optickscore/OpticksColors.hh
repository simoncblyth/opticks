#pragma once

#include "NQuad.hpp"

#include <map>
#include <string>
#include <vector>

template <typename T> class NPY ; 

/*

*OpticksColors* associates color names to RGB hexcodes

Usage from GLoader::

    m_colors = OpticksColors::load("$HOME/.opticks","GColors.json");

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



#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksColors {  
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
       static OpticksColors* load(const char* dir, const char* name=NAME);
   public:
       OpticksColors();

       void sort();
       void dump(const char* msg="OpticksColors::dump");
       void test(const char* msg="OpticksColors::test");
       bool operator() (const std::string& a, const std::string& b);

       static nvec3 makeColor( unsigned int rgb );

       const char* getNamePsychedelic(unsigned int index);
       unsigned int getCode(const char* name, unsigned int missing=0xFFFFFF);
       const char* getHex( const char* name, const char* missing=NULL);
       const char* getName(const char* hex_, const char* missing=NULL);
       unsigned int parseHex(const char* hex_);

       nvec3 getColor(const char* name, unsigned int missing=0xFFFFFF);
       nvec3 getPsychedelic(unsigned int num);
       std::vector<unsigned int>& getPsychedelicCodes() ;
       std::vector<unsigned int>& getSpectralCodes() ;

       unsigned int getNumColors();
       unsigned int getNumBytes();
       unsigned int getBufferEntry(unsigned char* colors);
   public: 
       NPY<unsigned char>* make_buffer();
       NPY<unsigned char>* make_buffer(std::vector<unsigned int>& codes);
   public: 
       void setupCompositeColorBuffer(std::vector<unsigned int>&  material_codes, std::vector<unsigned int>& flag_codes);
       NPY<unsigned char>* getCompositeBuffer();
       nuvec4   getCompositeDomain();
       void dumpCompositeBuffer(const char* msg="OpticksColors::dumpCompositeBuffer");

   private:
       void initCompositeColorBuffer(unsigned int max_colors);
       void addColors(std::vector<unsigned int>& codes, unsigned int offset=0 );
       void loadMaps(const char* dir);

   private:
       std::vector<std::string>            m_psychedelic_names ; 
       std::vector<unsigned int>           m_psychedelic_codes ;
       std::vector<unsigned int>           m_spectral_codes ;
       std::map<std::string, std::string>  m_name2hex ; 
       NPY<unsigned char>*                 m_composite ;
       nuvec4                              m_composite_domain ; 

};

#include "OKCORE_TAIL.hh"


