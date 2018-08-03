
#include <sstream>
#include <sstream>
#include <iostream>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <iterator>
#include <iostream>


//#include <boost/regex.hpp>
#include "BRegex.hpp"
#include <boost/algorithm/string/regex.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "BStr.hh"

#include "PLOG.hh"



void BStr::rtrim(std::string& s )
{
    boost::trim_right(s);
}

void BStr::replace_all(std::string& s, const std::string& fr, const std::string& to )
{
    boost::replace_all(s, fr, to);
} 
 
bool BStr::HasChar(const std::string& s, char c)
{
    return s.find(c) != std::string::npos ; 
}

bool BStr::EndsWith(const char* s, const char* q)
{
    return boost::ends_with(s, q);
}

const char* BStr::WithoutEnding(const char* s, const char* q)
{
    assert( EndsWith(s,q) ); 
    int n = std::strlen(s) - std::strlen(q) ; 
    assert( n > 0 ) ; 
    std::string fc( s, std::min<size_t>( n, std::strlen( s ) ));
    return strdup(fc.c_str()); 
}

bool BStr::StartsWith(const char* s, const char* q)
{
    return boost::starts_with(s, q);
}

int BStr::index_first( const std::vector<std::string>& elem, const char* item )
{
    typedef std::vector<std::string> VS ; 
    VS::const_iterator it = std::find( elem.begin(), elem.end(), item );
    return it == elem.end() ? -1 :  it - elem.begin() ; 
}

int BStr::index_all( std::vector<unsigned>& indices, const std::vector<std::string>& elem, const char* item )
{
    for(unsigned i=0 ; i < elem.size() ; i++) if(elem[i].compare(item) == 0) indices.push_back(i) ; 
    return indices.size();
}







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


char* BStr::DAEIdToG4( const char* daeid, bool trimPtr)
{
    /**
        Convert daeid such as  "__dd__Geometry__PoolDetails__lvLegInIWSTub0xc400e40" 
        to G4 name                  /dd/Geometry/PoolDetails/lvLegInIWSTub
    **/

    std::string id = daeid ; 
    if(trimPtr) id = BStr::trimPointerSuffixPrefix(id.c_str(), NULL);

    std::string rep(id);
 
    boost::replace_all(rep, "__", "/"); 
    boost::replace_all(rep, "--", "#"); 

/*
    LOG(info) 
              << " daeid " << std::setw(40) << daeid 
              << " trimPtr " << std::setw(40) << trimPtr
              << " rep  " << std::setw(40) << rep
              ;
*/

    return strdup(rep.c_str());
}


char* BStr::trimPointerSuffixPrefix(const char* origname, const char* prefix)
{
    //  __dd__Materials__ADTableStainlessSteel0xc177178    0x is 9 chars from the end
    const char* ox = "0x" ;
    char* name = strdup(origname);       // make a copy to modify 

    if( strlen(name) > 9 )
    {
        char* c = name + strlen(name) - 9 ;    
        if(strncmp(c, ox, strlen(ox)) == 0) *c = '\0';   // insert NULL to snip off the 0x tail
    }

    if(prefix) name += strlen(prefix) ;
    return name ;   
}


std::string BStr::firstChars( const char* s, unsigned n)
{
     std::string fc( s, std::min<size_t>( n, std::strlen( s ) ));
     return fc ;
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


const char* BStr::utoa( unsigned u, int width, bool zeropad)
{
    std::stringstream ss ; 

    if(width > -1 )
    ss << std::setw(width) ; 

    if(zeropad)
    ss << std::setfill('0') ; 

    ss << u ; 

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


template<typename T>
const char* BStr::xtoa( T x )
{
    std::stringstream ss ; 
    ss << x ; 
    std::string s = ss.str() ;
    return strdup(s.c_str());
}



template<typename T>
T BStr::LexicalCast(const char* str) 
{
    return boost::lexical_cast<T>(str) ;
}

template<typename T>
T BStr::LexicalCast(const char* str, T fallback, bool& badcast ) 
{
    T value(fallback) ; 
    try{ 
        value = boost::lexical_cast<T>(str) ;
    }   
    catch (const boost::bad_lexical_cast& e ) { 
        badcast = true ; 
        LOG(warning)  << "Caught bad lexical cast with error " << e.what() ;
    }   
    catch( ... ){
        LOG(warning) << "Unknown exception caught!" ;
    }   
    return value ; 
}




int BStr::atoi( const char* str, int fallback )
{
    int i(fallback) ;   
    if(!str) return i ; 
  
    bool badlex = false ; 
 
    try{ 
        i = boost::lexical_cast<int>(str) ;
    }   
    catch (const boost::bad_lexical_cast& e ) { 
        LOG(warning)  << "bad_lexical_cast [" << e.what() << "] " 
                      << " with [" << str << "]" 
                       ;
        badlex = true ; 
    }   
    catch( ... ){
        LOG(warning) << "Unknown exception caught!" ;
    }   


    if(badlex)
    {
        LOG(error) << "BStr::atoi badlex "
                   << " str " << str  
                   << " fallback " << fallback
                   ;  
       // assert(0); 
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
        LOG(warning)  << "Caught bad lexical cast with error [" << e.what() << "] "
                      << " for str [" << str << "]" 
                      ;
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


/*
void BStr::split( std::vector<std::string>& elem, const char* line, char delim )
{
    if(line == NULL) return ; 
    std::istringstream f(line);
    std::string s;
    while (getline(f, s, delim)) elem.push_back(s);
}
*/




/**
BStr::Contains(s, q, delim)
-----------------------------

Returns true when any of the strings obtained by splitting 
q with delim are present in s. For example::

   assert( BStr::Contains("/some/path/to/VolCathodeEsque", "Cathode,cathode", ',' ) == true ) ; 
   assert( BStr::Contains("/some/path/to/VolcathodeEsque", "Cathode,cathode", ',' ) == true ) ; 

For single string contains use SStr::Contains

**/

bool BStr::Contains( const char* s_ , const char* q_, char delim )
{
    std::string s(s_); 
    std::vector<std::string> qv ; 
    BStr::split(qv, q_, delim) ;  

    for(unsigned i=0 ; i < qv.size() ; i++)
    {
        if(s.find(qv[i]) != std::string::npos) return true ;
    }
    return false ; 
}








template<typename T> 
unsigned BStr::Split(std::vector<T>& elem, const char* line, char delim )
{
    if(line == NULL) return 0 ; 
    std::istringstream f(line);
    std::string s;
    bool badcast(false); 

    unsigned count(0); 
    while (getline(f, s, delim))
    {
        T value = BStr::LexicalCast<T>(s.c_str(), -1, badcast );
        assert( !badcast );
        elem.push_back(value);
        count++ ; 
    }
    return count ; 
}



void BStr::split( std::vector<std::string>& elem, const char* line, char delim )
{
    if(line == NULL) return ; 

    std::stringstream ss;
    ss.str(line)  ;

    std::string s;
    while (std::getline(ss, s, delim)) elem.push_back(s);
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



void BStr::uslice_append( std::vector<unsigned>& elem, const char* sli, char delim )
{
    std::vector<unsigned> slice ; 
    BStr::usplit( slice, sli, delim );
    assert( slice.size() == 2 ); 
    unsigned u0 = slice[0] ; 
    unsigned u1 = slice[1] ; 
    for(unsigned u=u0 ; u < u1 ; u++)  elem.push_back(u); 
}

void BStr::usplit( std::vector<unsigned>& elem, const char* line, char delim )
{
    if(line == NULL) return ; 
    std::istringstream f(line);
    std::string s;
    while (getline(f, s, delim))
    {
        if(HasChar(s.c_str(), ':')) 
        {
             BStr::uslice_append(elem, s.c_str(), ':');
        }
        else 
        { 
            int i = BStr::atoi(s.c_str(), -1);
            unsigned u = i ; 
            elem.push_back(u);
        }
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


bool BStr::existsEnv( const char* envvar )
{
    char* line = getenv(envvar) ;
    return line != NULL ; 
}


void BStr::fsplitEnv( std::vector<float>& elem, const char* envvar, const char* fallback, char delim )
{
    char* line = getenv(envvar) ;

    if(line)
    {
        BStr::fsplit(elem, line, delim);
    }
    else
    {
        BStr::fsplit(elem, fallback, delim);
    }

    std::cout << "BStr::fsplitEnv"
              << " envvar " << envvar 
              << " line " << ( line ? line : " NULL " )
              << " fallback " << ( fallback ? fallback : " NULL " )
              << " elem.size " << elem.size()
              << std::endl 
              ;

    
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


std::string BStr::ujoin( std::vector<unsigned>& elem, char delim)
{
    char delimiter[2] ;
    delimiter[0] = delim ;
    delimiter[1] = '\0' ;

    std::stringstream ss ;    
    std::copy( elem.begin(), elem.end(), std::ostream_iterator<unsigned>(ss,delimiter));
    std::string s_ = ss.str();
    std::string s = s_.substr(0, s_.size()-1) ;  // remove trailing delimiter
    return s ;
}




std::string BStr::join( const char* a, const char* b, const char* c, const char* d, char delim)
{
    std::stringstream ss ; 
    if(a) ss << a ; 
    ss << delim ; 
    if(b) ss << b ; 
    ss << delim ; 
    if(c) ss << c ; 
    ss << delim ; 
    if(d) ss << d ; 
    return ss.str();
}


int BStr::ekv_split( std::vector<std::pair<std::string, std::string> > & ekv, const char* line_, char edelim, const char* kvdelim)
{
    int err = 0 ; 
    bool warn = true ; 
    const char* line = strdup(line_);
    typedef std::pair<std::string,std::string> KV ;  
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
            if(warn)
                LOG(warning) << "ignoring malformed kv [" << s.c_str() << "]" ; 
            err++ ; 
            //assert(0); 
        }
    }
    return err ; 
}


std::vector<std::pair<std::string, std::string> >  BStr::ekv_split( const char* line_, char edelim, const char* kvdelim )
{
    std::vector<std::pair<std::string, std::string> > ekv ; 
    ekv_split(ekv, line_, edelim, kvdelim);
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
const char* BStr::concat( const char* head, T body_, const char* tail  )
{
    std::string body = boost::lexical_cast<std::string>(body_) ;
    std::stringstream ss ; 

    if(head) ss << head ; 
    ss << body ; 
    if(tail) ss << tail ; 

    std::string s = ss.str();
    return strdup(s.c_str());
}





void BStr::ReplaceAll(std::string& subject, const char* search, const char* replace) 
{
    //https://stackoverflow.com/questions/3418231/replace-part-of-a-string-with-another-string
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) 
    {
        subject.replace(pos, strlen(search), replace);
        pos += strlen(replace) ;
    }
}






template BRAP_API const char* BStr::xtoa(float);
template BRAP_API const char* BStr::xtoa(double);




template BRAP_API bool               BStr::LexicalCast(const char* );
template BRAP_API std::string        BStr::LexicalCast(const char* );
template BRAP_API unsigned long long BStr::LexicalCast(const char* );
template BRAP_API unsigned char      BStr::LexicalCast(const char* );
template BRAP_API short              BStr::LexicalCast(const char* );
template BRAP_API unsigned           BStr::LexicalCast(const char* );
template BRAP_API int                BStr::LexicalCast(const char* );
template BRAP_API float              BStr::LexicalCast(const char* );
template BRAP_API double             BStr::LexicalCast(const char* ); 



template BRAP_API unsigned long long BStr::LexicalCast(const char*, unsigned long long, bool& );
template BRAP_API unsigned char BStr::LexicalCast(const char*, unsigned char, bool& );
template BRAP_API short    BStr::LexicalCast(const char*, short, bool& );
template BRAP_API unsigned BStr::LexicalCast(const char*, unsigned, bool& );
template BRAP_API int      BStr::LexicalCast(const char*, int     , bool& );
template BRAP_API float    BStr::LexicalCast(const char*, float   , bool& );
template BRAP_API double   BStr::LexicalCast(const char*, double  , bool& );


template BRAP_API unsigned BStr::Split(std::vector<unsigned long long>& , const char*, char );
template BRAP_API unsigned BStr::Split(std::vector<char>& ,               const char*, char );
template BRAP_API unsigned BStr::Split(std::vector<unsigned char>& ,      const char*, char );
template BRAP_API unsigned BStr::Split(std::vector<short>& ,              const char*, char );
template BRAP_API unsigned BStr::Split(std::vector<unsigned>& ,           const char*, char );
template BRAP_API unsigned BStr::Split(std::vector<int>& ,                const char*, char );
template BRAP_API unsigned BStr::Split(std::vector<float>& ,              const char*, char );
template BRAP_API unsigned BStr::Split(std::vector<double>& ,             const char*, char );


template BRAP_API const char* BStr::concat(const char*, int        , const char* );
template BRAP_API const char* BStr::concat(const char*, unsigned   , const char* );
template BRAP_API const char* BStr::concat(const char*, const char*, const char* );

