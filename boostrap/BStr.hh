#pragma once

#include <vector>
#include <string>
#include <map>

//
// returning std::string is fussy wrt compiler details, making inconvenient 
// ... so prefer to rely on external allocation of output 
// and just fill in the content here, as done in removeField


#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API BStr {
  public:
     static int  atoi( const char* str, int fallback=0 );
     static const char* itoa( int i );
     static const char* negate(const char* tag);
     static bool listHasKey(const char* dlist, const char* key, const char* delim=",");
     static char* trimPointerSuffixPrefix(const char* origname, const char* prefix);
     static const char* uppercase( const char* str );
     static std::string patternPickField(std::string str, std::string ptn, int num );

     static void split( std::vector<std::string>& elem, const char* line, char delim );
     static std::vector<std::pair<std::string, std::string> > ekv_split( const char* line, char edelim=' ', const char* kvdelim=":" );


     static std::string join(std::vector<std::string>& elem, char delim );
     static void removeField(char* dest, const char* line, char delim, int index );
     static std::string insertField(const char* line, char delim, int index, const char* field);
     static unsigned char* make_uchar4_colors(unsigned int n);
};


#include "BRAP_TAIL.hh"




