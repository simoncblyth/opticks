// started from http://www.boost.org/doc/libs/1_58_0/libs/regex/doc/html/boost_regex/partial_matches.html

#include "regexsearch.hh"
#include <sstream>
#include <fstream>
#include <iomanip>

#include <boost/lexical_cast.hpp>

void dump( pairs_t& pairs, const char* msg)
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

void dump( ipairs_t& pairs, const char* msg)
{
    std::cout << msg << " : " << pairs.size() << std::endl ; 
    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        ipair_t pair = pairs[i];

        //std::string k = pair.first ; 
        //int v = pair.second ;
        std::string k = pair.second ; 
        int v = pair.first ;

        std::cout 
                 << std::setw(30) << k  
                 << " : " 
                 << std::setw(10) << std::dec << v
                 << " : " 
                 << std::setw(10) << std::hex << v 
                 << std::endl ; 
    } 
}


bool iparse( int& result, std::string expr )
{
    
    boost::regex eint("\\d*"); 
    boost::regex eshift("(0x\\d*)\\s*<<\\s*(\\d*)");  // eg "0x1 << 31" 
 
    if(boost::regex_match(expr, eint))
    {
        result = boost::lexical_cast<int>(expr);
        return true ; 
    }
    else if(boost::regex_match(expr, eshift))
    {
         boost::sregex_iterator mt(expr.begin(), expr.end(), eshift);
         boost::sregex_iterator me;

         while(mt != me)
         {
             std::string base = (*mt)[1].str() ;
             int ibase = hex_lexical_cast<int>(base.c_str()) ;
             int ishift = boost::lexical_cast<int>((*mt)[2]) ;

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
             mt++ ;
         }
    } 
    return false ; 
}




std::string os_path_expandvars(const char* s)
{
    const char* ptn = "\\$(\\w*)/(\\S*)" ; // eg $ENV_HOME/graphics/ggeoview/cu/photon.h ->  ENV_HOME  graphics/ggeoview/cu/photon.h

    std::string str = s ;

    boost::regex e(ptn);

    std::stringstream ss ; 

    if(boost::regex_match(s,e))
    {
        boost::sregex_iterator mt(str.begin(), str.end(), e);
        boost::sregex_iterator me;

        if(mt != me)
        {
             const char* evar  = (*mt)[1].str().c_str() ;
             const char* etail = (*mt)[2].str().c_str() ;
             char* eval = getenv(evar);

             //printf("evar %s etail %s eval %s  \n", evar, etail, eval );

             if( eval )
             {
                 ss << eval << "/" << etail ;
             } 
        }
    }  

    std::string expanded = ss.str();
    return expanded.empty() ? s : expanded ; 
}






void enum_regexsearch( ipairs_t& ipairs, const char* path )
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

*/

    const char* ptn = "^\\s*(\\w+)\\s*=\\s*(.*?),*\\s*?$" ;  
    boost::regex e(ptn);

    std::string epath = os_path_expandvars(path);
    std::ifstream is(epath, std::ifstream::binary); 

    pairs_t   pairs ; 
    regexsearch( pairs, is , e );

    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        int iv ; 
        if(iparse(iv, pairs[i].second)) ipairs.push_back(ipair_t(iv, pairs[i].first));   
    } 
}



void regexsearch( pairs_t& pairs, std::istream& is, boost::regex& e )
{
   char buf[4096];
   const char* next_pos = buf + sizeof(buf);  // end of buf

   bool more = true;
   unsigned int count = 0;

   while(more)
   {
      unsigned leftover = (buf + sizeof(buf)) - next_pos;
      unsigned size = next_pos - buf;        // 

      std::memmove(buf, next_pos, leftover); // shunt leftovers from prior partial matches to buf head
      is.read(buf + leftover, size);         // fill from stream
      unsigned read = is.gcount();
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


