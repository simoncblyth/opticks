#include "stringutil.hpp"
#include "md5digest.h"

#include <sstream>
#include <boost/lexical_cast.hpp>


int getenvint( const char* envkey, int fallback )
{
    char* val = getenv(envkey);
    int ival = val ? boost::lexical_cast<int>(val) : fallback ;
    return ival ; 
}


const char* getenvvar( const char* envprefix, const char* envkey )
{
    char envvar[128];
    snprintf(envvar, 128, "%s%s", envprefix, envkey );
    return getenv(envvar);
}



void split( std::vector<std::string>& elem, const char* line, char delim )
{
    if(line == NULL) return ; 
    std::istringstream f(line);
    std::string s;
    while (getline(f, s, delim)) elem.push_back(s);
}


std::string join(std::vector<std::string>& elem, char delim )
{
    typedef std::vector<std::string> Vec_t ;
    std::stringstream ss ;
    for(size_t i=0 ; i < elem.size() ; ++i)
    {
        ss << elem[i] ;
        if( i < elem.size() - 1) ss << delim ;
    }
    return ss.str();
}

void removeField(char* dest, const char* line, char delim, int index )
{
    //  
    //   split the line with the delim
    //   then reassemble skipping the field pointed to by rfield
    //  
    //    For example the below line with delim '.' and rfield -2
    //  
    //       /path/to/geometry.dae.noextra.abcdefghijklmnopqrstuvwxyz.dae
    //       /path/to/geometry.dae.noextra.dae
    //  

    std::vector<std::string> elem ;
    split(elem, line, delim);


    if(index >= 0 && index < elem.size())
    {
        elem.erase( elem.begin() + index);
    }
    else if( index < 0 && -index < elem.size())
    {
        elem.erase( elem.end() + index );
    }
    else
    {
        printf("removeField line %s delim %c index %d : invalid index \n", line, delim, index );
    }
    std::string j = join(elem, delim);

    strcpy( dest, j.c_str()) ;
}


std::string insertField(const char* line, char delim, int index, const char* field)
{
    std::vector<std::string> elem ;
    split(elem, line, delim);    

    std::string s(field);

    if(index >= 0 && index < elem.size())
    {   
        elem.insert( elem.begin() + index, s); 
    }   
    else if( index < 0 && -index < elem.size())
    {   
        elem.insert( elem.end() + index, s );
    }   
    else
    {   
        printf("insertField line %s delim %c index %d : invalid index \n", line, delim, index );
    }   
    return join(elem, delim); 
}



std::string md5digest( const char* buffer, int len )
{
    char* out = md5digest_str2md5(buffer, len);
    std::string digest(out);
    free(out);
    return digest;
}





