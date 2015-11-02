#include "Lookup.hpp"
#include "jsonutil.hpp"
#include <cstring>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

namespace fs = boost::filesystem;



void Lookup::loadA(const char* adir, const char* aname, const char* aprefix)
{
    typedef std::map<std::string, unsigned int> SU_t ; 

    SU_t raw ; 
    loadMap<std::string, unsigned int>(raw, adir, aname );

    for(SU_t::iterator it=raw.begin() ; it != raw.end() ; it++)
    {
        const char* name = it->first.c_str() ; 
        if(strncmp(name, aprefix, strlen(aprefix)) == 0)
        {
            std::string shortname = name + strlen(aprefix) ;
            m_A[shortname] = it->second ;  
        }
    }
}

void Lookup::loadB(const char* bdir, const char* bname, const char* bprefix)
{
    loadMap<std::string, unsigned int>(m_B, bdir, bname);
}


void Lookup::crossReference()
{
    m_A2B = create(m_A, m_B);
    m_B2A = create(m_B, m_A);
}


std::string Lookup::acode2name(unsigned int acode)
{
    return find(m_A, acode);
}
std::string Lookup::bcode2name(unsigned int bcode)
{
    return find(m_B, bcode);
}


void Lookup::dump(const char* msg)
{
    printf("Lookup::dump %s \n", msg );
    printf("A   %lu entries \n", m_A.size());  
    printf("B   %lu entries \n", m_B.size());  
    printf("A2B %lu entries in lookup  \n", m_A2B.size() );  

    for(Lookup_t::iterator it=m_A2B.begin() ; it != m_A2B.end() ; it++)
    {
        unsigned int acode = it->first ;  
        unsigned int bcode = it->second ;
        std::string aname = acode2name(acode);
        std::string bname = bcode2name(bcode);
        printf("  A %4u : %25s  B %4u : %25s \n", acode, aname.c_str(), bcode, bname.c_str() );
    }
}


std::map<unsigned int, unsigned int> Lookup::create(Map_t& a, Map_t&b)
{
    Lookup_t a2b ;
    for(Map_t::iterator ia=a.begin() ; ia != a.end() ; ia++)
    {
        std::string aname = ia->first ;
        unsigned int acode = ia->second ; 
        if(b.find(aname) != b.end())
        {
            unsigned int bcode = b[aname];
            a2b[acode] = bcode ; 
        }
    }
    return a2b ; 
}

void Lookup::dumpMap(const char* msg, Map_t& map)
{
    printf("Lookup::dumpMap %s \n", msg);
    for(Map_t::iterator it=map.begin() ; it != map.end() ; it++)
    {
        std::string name = it->first ;
        unsigned int code = it->second ; 
        printf("   %25s : %u \n", name.c_str(), code);
    }
}

std::string Lookup::find(Map_t& m, unsigned int code)
{
    std::string name ; 
    for(Map_t::iterator im=m.begin() ; im != m.end() ; im++)
    {
        unsigned int mcode = im->second ; 
        if(mcode == code)
        {
           name = im->first ;
           break;
        }
    }
    return name ; 
}


int Lookup::a2b(unsigned int a)
{
    return lookup(m_A2B, a );
}
int Lookup::b2a(unsigned int b)
{
    return lookup(m_B2A, b );
}
int Lookup::lookup(Lookup_t& lkup, unsigned int x)
{
    return lkup.find(x) == lkup.end() ? -1 : lkup[x] ;
}



