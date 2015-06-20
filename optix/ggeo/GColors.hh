#pragma once


/*

  import json, matplotlib.colors  
  json.dump(matplotlib.colors.cnames, open("/tmp/colors.json", "w"))

*/

#include <map>
#include <string>

class GColors {  
   public:
       static const char* NAME ; 
       static GColors* load(const char* dir);
   public:
       GColors();
       void dump(const char* msg="GColors::dump");
       void test(const char* msg="GColors::test");

       const char* getHex( const char* name, const char* missing=NULL);
       const char* getName(const char* hex_, const char* missing=NULL);
       unsigned int parseHex(const char* hex_);
       unsigned int getNumColors();
       unsigned int getNumBytes();
       unsigned int getBufferEntry(unsigned char* colors);
       unsigned char* make_uchar4_buffer();

   private:
       void loadMaps(const char* dir);

   private:
       std::map<std::string, std::string>  m_name2hex ; 

};

inline GColors::GColors() 
{
}


