#pragma once

#include <sstream>
#include <iostream>
#include <string>
#include <boost/regex.hpp>
#include <vector>

typedef std::pair<std::string, std::string>   pair_t ;
typedef std::vector<pair_t>                   pairs_t ;

typedef std::pair<int, std::string>           ipair_t ;
typedef std::vector<ipair_t>                  ipairs_t ;

typedef std::pair<unsigned int, std::string>  upair_t ;
typedef std::vector<upair_t>                  upairs_t ;


void regexsearch( pairs_t& pairs, std::istream& is, boost::regex& e );

void enum_regexsearch( upairs_t& upairs, const char* path );


std::string os_path_expandvars(const char* s);
std::string regex_extract_quoted(const char* line);


void dump(  pairs_t& pairs, const char* msg="dump" );

//void dump( ipairs_t& pairs, const char* msg="dump" );
//void dump( upairs_t& pairs, const char* msg="dump" );


//template<typename T>
//void dump( std::vector<std::pair<T, std::string> >& pairs, const char* msg="dump");

void udump( std::vector<std::pair<unsigned int, std::string> >& pairs, const char* msg="udump");



template<typename T>
inline T hex_lexical_cast(const char* in) {
    T out;
    std::stringstream ss;
    ss <<  std::hex << in;
    ss >> out;
    return out;
}





