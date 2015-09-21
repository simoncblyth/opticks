#include "stringutil.hpp"
#include "md5digest.h"

#include <sstream>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace pt = boost::property_tree;


int getenvint( const char* envkey, int fallback )
{
    char* val = getenv(envkey);
    int ival = val ? boost::lexical_cast<int>(val) : fallback ;
    return ival ; 
}


const char* uppercase( const char* str )
{
    char* STR = strdup(str);
    char* p = STR ;
    while(*p)
    {
       if( *p >= 'a' && *p <= 'z') *p += 'A' - 'a' ;
       p++ ; 
    } 
    return STR ;
}


const char* getenvvar( const char* envprefix, const char* envkey )
{
    char envvar[128];
    snprintf(envvar, 128, "%s%s", envprefix, envkey );
    return getenv(envvar);
}


std::string patternPickField(std::string str, std::string ptn, int num )
{
    std::vector<std::string> result;
    boost::algorithm::split_regex( result, str, boost::regex(ptn) ) ;
    unsigned int size = result.size();

    //printf("patternPickField %u \n", size );
    if(num < 0) num+= size ; 

    assert(num > -1 && num < size);

    return result[num];
}




void split( std::vector<std::string>& elem, const char* line, char delim )
{
    if(line == NULL) return ; 
    std::istringstream f(line);
    std::string s;
    while (getline(f, s, delim)) elem.push_back(s);
}




std::vector<std::pair<std::string, std::string>> ekv_split( const char* line, char edelim, const char* kvdelim )
{
    typedef std::pair<std::string,std::string> KV ;  
    std::vector<KV> ekv ; 
    std::istringstream f(line);
    std::string s;
    while (getline(f, s, edelim))
    {
        std::vector<std::string> kv ;
        boost::split(kv, s, boost::is_any_of(kvdelim));
        if(kv.size() == 2)
        {
            ekv.push_back(KV(kv[0],kv[1]));
        }
        else
        {
            printf("stringutil.ekv_split ignoring malformed kv %s \n", s.c_str() );
        }
    }
    return ekv ;
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

#define RGBA(r,g,b,a) \
{ \
    colors[offset + 0] = (r) ; \
    colors[offset + 1] = (g) ; \
    colors[offset + 2] = (b);  \
    colors[offset + 3] = (a) ; \
} \




unsigned char* make_uchar4_colors(unsigned int n)
{
    unsigned char* colors = new unsigned char[n*4] ; 
    for(unsigned int i=0 ; i < n ; i++ )
    {   
        unsigned int offset = i*4 ; 
        switch(i % 5)
        {   
           case 0:RGBA(0xff,0x00,0x00,0xff);break; // R
           case 1:RGBA(0x00,0xff,0x00,0xff);break; // G 
           case 2:RGBA(0xff,0x00,0xff,0xff);break; // M 
           case 3:RGBA(0xff,0xff,0x00,0xff);break; // Y 
           case 4:RGBA(0x00,0xff,0xff,0xff);break; // C
        }   
    }   

   /*
In [1]: np.linspace(0.,1.,5 )
Out[1]: array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])
                
In [2]: np.linspace(0.,1.,5+1 )
Out[2]: array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
                RRRRRRRRGGGGGGMMMMMM

In [10]: (np.arange(0,5) + 0.5)/5              (i + 0.5)/5.0  lands mid-bin
Out[10]: array([ 0.1,  0.3,  0.5,  0.7,  0.9])


     // -0.10  R
     //  0.00  R 
     //  0.200 R
     ---------------
     //  0.201 G 
     //  0.205 G 
     //  0.250 G 
     //  0.300 G 
     //  0.400 G  
     ---------------
     //  0.401 M
     //  0.5   M
     //  0.6   M 
     ---------------
     //  0.601 Y
     //  0.7   Y
     //  0.8   Y
     --------------
     //  0.801 C
     //  0.9   C
     //  1.0   C
     --------------
     //  1.1   C
     //  1.5   C    
     //   
  */


    return colors ; 
}







