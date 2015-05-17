#pragma once

#include <vector>
#include <string>

//
// returning std::string is fussy wrt compiler details, making inconvenient 
// ... so prefer to rely on external allocation of output 
// and just fill in the content here, as done in removeField

int getenvint( const char* envkey, int fallback=-1 );
const char* getenvvar( const char* envprefix, const char* envkey );

void split( std::vector<std::string>& elem, const char* line, char delim );

std::string join(std::vector<std::string>& elem, char delim );
std::string insertField(const char* line, char delim, int index, const char* field);
std::string md5digest( const char* buffer, int len );

void removeField(char* dest, const char* line, char delim, int index );


