#pragma once

#include <sstream>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <boost/regex.hpp>

typedef std::pair<std::string, std::string>   pair_t ;
typedef std::vector<pair_t>                   pairs_t ;

typedef std::pair<int, std::string>           ipair_t ;
typedef std::vector<ipair_t>                  ipairs_t ;

typedef std::pair<unsigned int, std::string>  upair_t ;
typedef std::vector<upair_t>                  upairs_t ;


void regexsearch( pairs_t& pairs, std::istream& is, boost::regex& e );

void enum_read(std::map<std::string, unsigned int>& emap, const char* path);
void enum_regexsearch( upairs_t& upairs, const char* path );


std::string os_path_expandvars(const char* s, bool debug=false);
std::string regex_extract_quoted(const char* line);
std::string regex_matched_element(const char* line);


void dump(  pairs_t& pairs, const char* msg="dump" );

void udump( std::vector<std::pair<unsigned int, std::string> >& pairs, const char* msg="udump");






