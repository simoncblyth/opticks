
#include <sstream>
#include <sstream>
#include <iostream>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <iterator>
#include <iostream>


#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/algorithm/string/replace.hpp>

#include "BStr.hh"

#include "PLOG.hh"


bool BStr::listHasKey(const char* dlist, const char* key, const char* delim)
{  
    std::vector<std::string> elems ; 
    boost::split(elems, dlist, boost::is_any_of(delim));
    bool haskey = false ;
    for(unsigned int i=0 ; i < elems.size() ; i++) 
    {    
       if(strcmp(elems[i].c_str(),key)==0) 
       {    
           haskey = true ;
           break ; 
       }    
    }    
    return haskey ; 
}


char* BStr::DAEIdToG4( const char* daeid)
{
    /**
        Convert daeid such as  "__dd__Geometry__PoolDetails__lvLegInIWSTub0xc400e40" 
        to G4 name                  /dd/Geometry/PoolDetails/lvLegInIWSTub
    **/

    std::string trimPtr = BStr::trimPointerSuffixPrefix(daeid, NULL);
    std::string rep__(trimPtr);
    boost::replace_all(rep__, "__", "/"); 

/*
    LOG(info) 
              << " daeid " << std::setw(40) << daeid 
              << " trimPtr " << std::setw(40) << trimPtr
              << " rep__  " << std::setw(40) << rep__
              ;
*/

    return strdup(rep__.c_str());
}


char* BStr::trimPointerSuffixPrefix(const char* origname, const char* prefix)
{
    //  __dd__Materials__ADTableStainlessSteel0xc177178    0x is 9 chars from the end
    const char* ox = "0x" ;
    char* name = strdup(origname);
    char* c = name + strlen(name) - 9 ;    
    if(strncmp(c, ox, strlen(ox)) == 0) *c = '\0';   // insert NULL to snip off the 0x tail
    if(prefix) name += strlen(prefix) ;
    return name ;   
}


char* BStr::afterLastOrAll(const char* orig, char delim)
{
    const char* p = strrchr(orig, delim) ;      // point at last delim, or NULL if no delim 
    bool lastchar = p && ( p == orig + strlen(orig) - 1 ) ;
    const char* name = p && !lastchar ? p + 1 : orig ; 
    return strdup(name) ;
}


const char* BStr::ctoa( char c )
{
    std::stringstream ss ; 
    ss << c ; 
    std::string s = ss.str() ;
    return strdup(s.c_str());
}

const char* BStr::itoa( int i )
{
    std::stringstream ss ; 
    ss << i ; 
    std::string s = ss.str() ;
    return strdup(s.c_str());
}


int BStr::atoi( const char* str, int fallback )
{
    int i(fallback) ;   
    if(!str) return i ; 
 
    try{ 
        i = boost::lexical_cast<int>(str) ;
    }   
    catch (const boost::bad_lexical_cast& e ) { 
        LOG(warning)  << "Caught bad lexical cast with error " << e.what() ;
    }   
    catch( ... ){
        LOG(warning) << "Unknown exception caught!" ;
    }   
    return i ;
}

float BStr::atof( const char* str, float fallback )
{
    float f(fallback) ;   
    if(!str) return f ; 
 
    try{ 
        f = boost::lexical_cast<float>(str) ;
    }   
    catch (const boost::bad_lexical_cast& e ) { 
        LOG(warning)  << "Caught bad lexical cast with error " << e.what() ;
    }   
    catch( ... ){
        LOG(warning) << "Unknown exception caught!" ;
    }   
    return f ;
}





const char* BStr::negate(const char* tag)
{
    int itag = BStr::atoi(tag) ;
    std::stringstream ss ;
    ss << -itag ; 
    std::string stag = ss.str();
    return strdup(stag.c_str());
}


const char* BStr::uppercase( const char* str )
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


std::string BStr::patternPickField(std::string str, std::string ptn, int num )
{
    std::vector<std::string> result;
    boost::algorithm::split_regex( result, str, boost::regex(ptn) ) ;
    int size = result.size();

    //printf("patternPickField %u \n", size );
    if(num < 0) num+= size ; 

    assert(num > -1 && num < size);

    return result[num];
}


void BStr::split( std::vector<std::string>& elem, const char* line, char delim )
{
    if(line == NULL) return ; 
    std::istringstream f(line);
    std::string s;
    while (getline(f, s, delim)) elem.push_back(s);
}

void BStr::isplit( std::vector<int>& elem, const char* line, char delim )
{
    if(line == NULL) return ; 
    std::istringstream f(line);
    std::string s;
    while (getline(f, s, delim))
    {
        int i = BStr::atoi(s.c_str(), -1);
        elem.push_back(i);
    }
}


void BStr::usplit( std::vector<unsigned>& elem, const char* line, char delim )
{
    if(line == NULL) return ; 
    std::istringstream f(line);
    std::string s;
    while (getline(f, s, delim))
    {
        int i = BStr::atoi(s.c_str(), -1);
        unsigned u = i ; 
        elem.push_back(u);
    }
}

void BStr::fsplit( std::vector<float>& elem, const char* line, char delim )
{
    if(line == NULL) return ; 
    std::istringstream f(line);
    std::string s;
    while (getline(f, s, delim))
    {
        float i = BStr::atof(s.c_str(), 0);
        elem.push_back(i);
    }
}





std::string BStr::ijoin( std::vector<int>& elem, char delim)
{
    char delimiter[2] ;
    delimiter[0] = delim ;
    delimiter[1] = '\0' ;

    std::stringstream ss ;    
    std::copy( elem.begin(), elem.end(), std::ostream_iterator<int>(ss,delimiter));
    std::string s_ = ss.str();
    std::string s = s_.substr(0, s_.size()-1) ;  // remove trailing delimiter
    return s ;
}



std::vector<std::pair<std::string, std::string> > 
BStr::ekv_split( const char* line_, char edelim, const char* kvdelim )
{
    const char* line = strdup(line_);
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
    

std::string BStr::join(std::vector<std::string>& elem, char delim )
{
    std::stringstream ss ;
    for(size_t i=0 ; i < elem.size() ; ++i)
    {
        ss << elem[i] ;
        if( i < elem.size() - 1) ss << delim ;
    }
    return ss.str();
}

void BStr::removeField(char* dest, const char* line, char delim, int index )
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
    int size = elem.size();

    if(index >= 0 && index < size)
    {
        elem.erase( elem.begin() + index);
    }
    else if( index < 0 && -index < size )
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


std::string BStr::insertField(const char* line, char delim, int index, const char* field)
{
    std::vector<std::string> elem ;
    split(elem, line, delim);    
    int size = elem.size();
    std::string s(field);

    if(index >= 0 && index < size)
    {   
        elem.insert( elem.begin() + index, s); 
    }   
    else if( index < 0 && -index < size)
    {   
        elem.insert( elem.end() + index, s );
    }   
    else
    {   
        printf("insertField line %s delim %c index %d : invalid index \n", line, delim, index );
    }   
    return join(elem, delim); 
}





#define RGBA(r,g,b,a) \
{ \
    colors[offset + 0] = (r) ; \
    colors[offset + 1] = (g) ; \
    colors[offset + 2] = (b);  \
    colors[offset + 3] = (a) ; \
} \




unsigned char* BStr::make_uchar4_colors(unsigned int n)
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

template <typename T>
const char* BStr::concat( const char* head, T body_, const char* tail )
{
    std::string body = boost::lexical_cast<std::string>(body_) ;
    std::stringstream ss ; 

    if(head) ss << head ; 
    ss << body ; 
    if(tail) ss << tail ; 

    std::string s = ss.str();
    return strdup(s.c_str());
}


template BRAP_API const char* BStr::concat(const char*, int        , const char* );
template BRAP_API const char* BStr::concat(const char*, unsigned   , const char* );
template BRAP_API const char* BStr::concat(const char*, const char*, const char* );

