#pragma once


/*

  import json, matplotlib.colors  
  json.dump(matplotlib.colors.cnames, open("/tmp/colors.json", "w"))

*/

#include <map>
#include <string>
#include <vector>

class GBuffer ; 

class GColors {  
   public:
       static const char* NAME ; 
       static GColors* load(const char* dir, const char* name=NAME);
   public:
       GColors();
       void dump(const char* msg="GColors::dump");
       void test(const char* msg="GColors::test");
       bool operator() (const std::string& a, const std::string& b);

       unsigned int getCode(const char* name, unsigned int missing=0xFFFFFF);
       const char* getHex( const char* name, const char* missing=NULL);
       const char* getName(const char* hex_, const char* missing=NULL);
       unsigned int parseHex(const char* hex_);
       unsigned int getNumColors();
       unsigned int getNumBytes();
       unsigned int getBufferEntry(unsigned char* colors);
       GBuffer* make_uchar4_buffer();
       GBuffer* make_uchar4_buffer(std::vector<unsigned int>& codes);
       void dump_uchar4_buffer( GBuffer* buffer );

   public: 
       void initCompositeColorBuffer(unsigned int max_colors);
       void addColors(std::vector<unsigned int>& codes, unsigned int offset=0 );
       GBuffer* getCompositeBuffer();
       void dumpCompositeBuffer(const char* msg="GColors::dumpCompositeBuffer");

   private:
       void loadMaps(const char* dir);

   private:
       std::map<std::string, std::string>  m_name2hex ; 
       GBuffer*                            m_composite ; 

};

inline GColors::GColors()  
    :
    m_composite(NULL)
{
}


inline GBuffer* GColors::getCompositeBuffer()
{
   return m_composite ;  
}
