#include "Lookup.hpp"

#include "string.h"

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/lexical_cast.hpp>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


namespace fs = boost::filesystem;
namespace pt = boost::property_tree;

std::map<std::string, unsigned int> parseTree(boost::property_tree::ptree& tree, const char* prefix)
{
    if(!prefix) prefix="";
    Lookup::Map_t name2code ; 

    BOOST_FOREACH( boost::property_tree::ptree::value_type const& ak, tree.get_child("") )
    {
        const char* name = ak.first.c_str() ; 
        if(strncmp(name, prefix, strlen(prefix)) == 0)
        {
            unsigned int code = boost::lexical_cast<unsigned int>(ak.second.data().c_str());
            std::string shortname = name + strlen(prefix) ;
            name2code[shortname] = code ;  
            //std::cout << shortname << " : " << code << std::endl ;
        }
    }
    return name2code ; 
}


Lookup::Lookup()
{
}
const char* Lookup::ANAME = "ChromaMaterialMap.json";
const char* Lookup::BNAME = "GSubstanceLibMetadataMaterialMap.json";

void Lookup::loada(const char* adir, const char* aname, const char* aprefix)
{
    char apath[256];
    snprintf(apath, 256, "%s/%s", adir, aname);

    boost::property_tree::ptree atree ; 
    pt::read_json(apath, atree );

    m_a = parseTree(atree,aprefix);
    m_apath = apath;
}

void Lookup::loadb(const char* bdir, const char* bname, const char* bprefix)
{
    char bpath[256];
    snprintf(bpath, 256, "%s/%s", bdir, bname);

    boost::property_tree::ptree btree ; 
    pt::read_json(bpath, btree );

    m_b = parseTree(btree,bprefix);
    m_bpath = bpath;
}

void Lookup::create()
{
    m_a2b = _create(m_a, m_b);
    m_b2a = _create(m_b, m_a);
}


std::string Lookup::acode2name(unsigned int acode)
{
    return find(m_a, acode);
}
std::string Lookup::bcode2name(unsigned int bcode)
{
    return find(m_b, bcode);
}


void Lookup::dump(const char* msg)
{
    printf("Lookup::dump %s \n", msg );
    printf("A   %lu entries from %s\n", m_a.size(), m_apath.c_str());  
    printf("B   %lu entries from %s\n", m_b.size(), m_bpath.c_str());  
    printf("A2B %lu entries in lookup  \n", m_a2b.size() );  

    for(Lookup_t::iterator it=m_a2b.begin() ; it != m_a2b.end() ; it++)
    {
        unsigned int acode = it->first ;  
        unsigned int bcode = it->second ;
        std::string aname = acode2name(acode);
        std::string bname = bcode2name(bcode);
        printf("  A %4u : %25s  B %4u : %25s \n", acode, aname.c_str(), bcode, bname.c_str() );
    }
}


std::map<unsigned int, unsigned int> Lookup::_create(Map_t& a, Map_t&b)
{
    Lookup_t a2b ;
    for(Map_t::iterator ia=a.begin() ; ia != a.end() ; ia++)
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
    return lookup(m_a2b, a );
}
int Lookup::b2a(unsigned int b)
{
    return lookup(m_b2a, b );
}
int Lookup::lookup(Lookup_t& lkup, unsigned int x)
{
    return lkup.find(x) == lkup.end() ? -1 : lkup[x] ;
}






