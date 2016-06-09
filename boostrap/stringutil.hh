#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <map>

#include <iostream>
#include <streambuf>

//
// returning std::string is fussy wrt compiler details, making inconvenient 
// ... so prefer to rely on external allocation of output 
// and just fill in the content here, as done in removeField

bool listHasKey(const char* dlist, const char* key, const char* delim=",");
char* trimPointerSuffixPrefix(const char* origname, const char* prefix);


const char* uppercase( const char* str );

void split( std::vector<std::string>& elem, const char* line, char delim );

std::vector<std::pair<std::string, std::string> > ekv_split( const char* line, char edelim=' ', const char* kvdelim=":" );


std::string patternPickField(std::string str, std::string ptn, int num );

std::string join(std::vector<std::string>& elem, char delim );
std::string insertField(const char* line, char delim, int index, const char* field);
std::string md5digest( const char* buffer, int len );

void removeField(char* dest, const char* line, char delim, int index );

unsigned char* make_uchar4_colors(unsigned int n);
void saveIndexJSON( std::map<unsigned int, std::string>& index, const char* path);


template<typename T>
inline T hex_lexical_cast(const char* in) {
    T out;
    std::stringstream ss; 
    ss <<  std::hex << in; 
    ss >> out;
    return out;
}

template<typename T>
inline std::string as_hex(T in) {
    std::stringstream ss; 
    ss <<  std::hex << in; 
    return ss.str();
}

template<typename T>
inline std::string as_dec(T in) {
    std::stringstream ss; 
    ss <<  std::dec << in; 
    return ss.str();
}






template<typename T>
std::string arraydigest( T* data, unsigned int n );


// http://stackoverflow.com/questions/5419356/redirect-stdout-stderr-to-a-string

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







