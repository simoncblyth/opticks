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

void regexsearch( pairs_t& pairs, std::istream& is, boost::regex& e );
void enum_regexsearch( ipairs_t& ipairs, const char* path );

std::string os_path_expandvars(const char* s);


void dump(  pairs_t& pairs, const char* msg="dump" );
void dump( ipairs_t& pairs, const char* msg="dump" );


template<typename T>
inline T hex_lexical_cast(const char* in) {
    T out;
    std::stringstream ss;
    ss <<  std::hex << in;
    ss >> out;
    return out;
}





