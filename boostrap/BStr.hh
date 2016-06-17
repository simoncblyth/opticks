#pragma once

#include <vector>
#include <string>
#include <map>

//
// returning std::string is fussy wrt compiler details, making inconvenient 
// ... so prefer to rely on external allocation of output 
// and just fill in the content here, as done in removeField


#include "BRAP_API_EXPORT.hh"
#include "BRAP_FLAGS.hh"

class BRAP_API BStr {
  public:
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


/*
   moved to BHex

     template<typename T>
     static T hex_lexical_cast(const char* in) ;

     template<typename T>
     static std::string as_hex(T in);

     template<typename T>
     static std::string as_dec(T in);

*/

};





/*
// http://stackoverflow.com/questions/5419356/redirect-stdout-stderr-to-a-string

#include <streambuf>

struct cout_redirect {
    cout_redirect( std::streambuf * new_buffer ) 
        : old( std::cout.rdbuf( new_buffer ) ) 
    { } 

    ~cout_redirect( ) { 
        std::cout.rdbuf( old );
    }   

private:
    std::streambuf * old;
};


struct cerr_redirect {
    cerr_redirect( std::streambuf * new_buffer ) 
        : old( std::cerr.rdbuf( new_buffer ) ) 
    { } 

    ~cerr_redirect( ) { 
        std::cerr.rdbuf( old );
    }   

private:
    std::streambuf * old;
};

*/





