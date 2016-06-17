// started from http://www.boost.org/doc/libs/1_58_0/libs/regex/doc/html/boost_regex/partial_matches.html

#include <cstring>
#include <sstream>
#include <fstream>
#include <iomanip>

#include <boost/lexical_cast.hpp>


// cannot BLog.hh this as bregex- is below NPY
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


#include "BRegex.hh"
#include "BFile.hh"
#include "BStr.hh"
#include "BHex.hh"
#include "bffs.hh"


#include "BRAP_FLAGS.hh"


void BRegex::dump( pairs_t& pairs, const char* msg)
{
    std::cout << msg << " : " << pairs.size() << std::endl ; 
    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        pair_t pair = pairs[i];
        std::cout 
                 << std::setw(30) << pair.first  
                 << " : " 
                 << std::setw(10) << pair.second  
                 << std::endl ; 
    } 
}


//template <typename T>
void BRegex::udump( std::vector<std::pair<unsigned int, std::string> >& pairs, const char* msg)
{
    std::cout << msg << " : " << pairs.size() << std::endl ; 
    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        std::pair<unsigned int, std::string> pair = pairs[i];

        unsigned int v = pair.first ;
        std::string  k = pair.second ; 

        std::cout 
                 << std::setw(30) << k  
                 << " : " 
                 << std::setw(1) << std::hex << bffs(v) 
                 << " : " 
                 << std::setw(10) << std::dec << v
                 << " : " 
                 << std::setw(10) << std::hex << v 
                 << std::endl ; 
    } 
}





template <typename T>
bool parse( T& result, std::string expr )
{
    
    boost::regex eint("\\d*"); 
    boost::regex eshift("(0x\\d*)\\s*<<\\s*(\\d*)");  // eg "0x1 << 31" 
 
    if(boost::regex_match(expr, eint))
    {
        result = boost::lexical_cast<T>(expr);
        return true ; 
    }
    else if(boost::regex_match(expr, eshift))
    {
         boost::sregex_iterator mt(expr.begin(), expr.end(), eshift);
         boost::sregex_iterator me;

         while(mt != me)
         {
             std::string base = (*mt)[1].str() ;
             int ibase = BHex<T>::hex_lexical_cast(base.c_str()) ;
             int ishift = boost::lexical_cast<T>((*mt)[2]) ;

             /*
             std::cout << "matched " 
                       << "  base[" << base << "]" 
                       << " ibase[" << ibase << "]" 
                       << " ishift[" << ishift << "]" 
                       <<  std::endl ; 
             */

             result = ibase << ishift ;
             return true ;
             // taking first match only
             //mt++ ;
         }
    } 
    return false ; 
}




std::string BRegex::regex_matched_element(const char* line)
{
    const char* ptn = "__([^_\\s]+)\\s*$" ;
    boost::regex re(ptn);
    boost::cmatch matches;

    std::string result ; 
    if (boost::regex_search(line, matches, re))
    {
        result = matches[1] ;
    }
    else
    {
        LOG(warning)<<"regex_matched_element failed to match " ;
    }

    LOG(info)<<"regex_matched_element [" << line << "] [" << result << "]" ; 

    return result ;
}

/*
std::string BRegex::regex_matched_element_0(const char* line)
{
    const char* ptn = "__([^_]*)$" ;
    boost::regex e(ptn);
    std::string str(line);
    boost::sregex_iterator mt(str.begin(), str.end(), e);
    boost::sregex_iterator me;
    unsigned int i=0 ;
    while(mt != me)
    {
       LOG(info) << " : " << (*mt)[1] ;
       mt++; 
    } 
    std::string ret ;
    //if( results.size() > 0 ) ret = results[0] ;
    return ret ;
}
*/



std::string BRegex::regex_extract_quoted(const char* line)
{
    std::string str = line ;


    //char s = '\\' ; 
    //char q = '\"' ; 
    // "(.*?)"

    const char* ptn = "\"(.*?)\""  ;

    LOG(info) << "regexsearch::regex_extract_quoted "
              << " ptn [" << ptn << "] " 
              << " str [" << str << "] "
              ;

    std::stringstream ss ; 

    boost::regex e(ptn);

    if(boost::regex_match(line,e))
    {
        boost::sregex_iterator mt(str.begin(), str.end(), e);
        boost::sregex_iterator me;

        if(mt != me)
        {
             ss << (*mt)[1].str()  ;
        }
    }  
    else
    {
        LOG(info) << "regexsearch::regex_extract_quoted failed to match " ; 
    }
    return ss.str();
}





std::string BRegex::os_path_expandvars(const char* s, bool debug)
{

    printf("os_path_expandvars IS DEPRECATED : MOVE TO BFile::FormPath approach \n");

    assert(0);

    // const char* ptn = "\\$(\\w+)(/.+)?" ; // eg $ENV_HOME/graphics/ggeoview/cu/photon.h ->  ENV_HOME  graphics/ggeoview/cu/photon.h
    const char* ptn = "\\$(\\w+)" ; // eg $ENV_HOME/graphics/ggeoview/cu/photon.h ->  ENV_HOME  graphics/ggeoview/cu/photon.h


    std::string str = s ;

    boost::regex e(ptn);

    while(boost::regex_search(str,e,boost::match_default))
    {
        boost::sregex_iterator mt(str.begin(), str.end(), e);  // hmm : hangs for non-defined envvar
        boost::sregex_iterator me;

        if(mt != me)
        {
             std::string evar  = (*mt)[1].str();
             const char* key = evar.c_str();
             char* eval = getenv(key);

             printf("  getenv(\"%s\") -> \"%s\" \n", key, eval ? eval : "NULL" );


             if( eval )
             {
                 std::string seval = eval;

                 if(debug)  
                 printf("---> os_path_expandvars  evar %s eval %s  \n", evar.c_str(), seval.c_str() );

                 std::string skey = "\\$";
                 skey= skey+evar;

                 if(debug)  
                 printf("----> %s replace in %s \n", skey.c_str(), str.c_str());

                 boost::regex rep(skey);
                 str = boost::regex_replace(str, rep, seval);

                 if(debug)  
                 printf("-----> %s \n", str.c_str());
             } 
        }
    }  


    if(debug)  
    printf("---> os_path_expandvars(\"%s\") = \"%s\" \n", s, str.c_str() ) ;


    return str; 
}


void BRegex::enum_read(std::map<std::string, unsigned int>& emap, const char* path)
{
    const char* ptn = "^\\s*(\\w+)\\s*=\\s*(.*?),*\\s*?$" ;  
    boost::regex e(ptn);

    std::string epath = BFile::FormPath(path);
    std::ifstream is(epath.c_str(), std::ifstream::binary); 

    pairs_t   pairs ; 
    regexsearch( pairs, is , e );
    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        unsigned int uv ; 
        if(parse<unsigned int>(uv, pairs[i].second))
        {
            emap[pairs[i].first] = uv ; 
        }
    } 
}


void BRegex::enum_regexsearch( upairs_t& upairs, const char* path )
{
/*
Extracts names and values from files containing enum definitions looking like:

enum
{
    NO_HIT                 = 0x1 << 0,
    BULK_ABSORB            = 0x1 << 1,
    SURFACE_DETECT         = 0x1 << 2,
    SURFACE_ABSORB         = 0x1 << 3,
...


   TODO: get thus to handle comments 

*/

    const char* ptn = "^\\s*(\\w+)\\s*=\\s*(.*?),*\\s*?$" ;  
    boost::regex e(ptn);

    std::string epath = BFile::FormPath(path);
    std::ifstream is(epath.c_str(), std::ifstream::binary); 

    pairs_t   pairs ; 
    regexsearch( pairs, is , e );

    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        unsigned int uv ; 
        if(parse<unsigned int>(uv, pairs[i].second)) upairs.push_back(upair_t(uv, pairs[i].first));   
    } 
}



void BRegex::regexsearch( pairs_t& pairs, std::istream& is, boost::regex& e )
{
   char buf[4096];
   const char* next_pos = buf + sizeof(buf);  // end of buf

   bool more = true;
   //unsigned int count = 0;

   while(more)
   {
      unsigned long leftover = (buf + sizeof(buf)) - next_pos;
      std::streamsize size = next_pos - buf;        // 

      std::memmove(buf, next_pos, leftover); // shunt leftovers from prior partial matches to buf head
      is.read(buf + leftover, size);         // fill from stream
      std::streamsize read = is.gcount();
      more = read == size;                   // stream succeeds to fill buffer, so probably more available
      next_pos = buf + sizeof(buf);

      boost::cregex_iterator a(
         buf,
         buf + read + leftover,
         e,
         boost::match_default | boost::match_partial);

      boost::cregex_iterator b;

      while(a != b)
      {
         if((*a)[0].matched == false)
         {
            // Partial match, save position and break 
            // this is necessary for stream reading where the next chars may complete the match, 
            // so defer to the next round of reading
            next_pos = (*a)[0].first;
            break;
         }
         else
         {
            //std::cout << " 0[" << (*a)[0] << "]"  << " 1[" << (*a)[1] << "]" << " 2[" << (*a)[2] << "]" <<  std::endl ; 
            pairs.push_back( pair_t((*a)[1], (*a)[2]));
         }

         ++a;
      }
   }
}


