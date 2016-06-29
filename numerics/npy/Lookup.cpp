#include <cstring>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include "BMap.hh"

#include "Lookup.hpp"
#include "PLOG.hh"

namespace fs = boost::filesystem;

Lookup::Lookup()
{
}

std::map<std::string, unsigned int>& Lookup::getA()
{
    return m_A ; 
}
std::map<std::string, unsigned int>& Lookup::getB()
{
    return m_B ; 
}


void Lookup::mockup(const char* dir, const char* aname, const char* bname)
{
    mockA(dir, aname);
    mockB(dir, bname);
}
void Lookup::mockA(const char* adir, const char* aname)
{
    Map_t mp ; 
    mp["/dd/Materials/red"] = 10 ; 
    mp["/dd/Materials/green"] = 20 ; 
    mp["/dd/Materials/blue"] = 30 ; 
    BMap<std::string, unsigned int>::save(&mp, adir, aname );
}
void Lookup::mockB(const char* bdir, const char* bname)
{
    Map_t mp; 
    mp["red"] = 1 ; 
    mp["green"] = 2 ; 
    mp["blue"] = 3 ; 
    BMap<std::string, unsigned int>::save(&mp, bdir, bname );
}




void Lookup::loadA(const char* adir, const char* aname, const char* aprefix)
{
    //typedef std::map<std::string, unsigned int> SU_t ; 

    Map_t raw ; 
    BMap<std::string, unsigned int>::load(&raw, adir, aname );

    for(Map_t::iterator it=raw.begin() ; it != raw.end() ; it++)
    {
        const char* name = it->first.c_str() ; 
        if(strncmp(name, aprefix, strlen(aprefix)) == 0)
        {
            std::string shortname = name + strlen(aprefix) ;
            m_A[shortname] = it->second ;  
        }
    }
}

void Lookup::loadB(const char* bdir, const char* bname, const char* /*bprefix*/)
{
    BMap<std::string, unsigned int>::load(&m_B, bdir, bname);
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
    std::cout << msg
              << " A entries " <<  m_A.size() 
              << " B entries " <<  m_B.size() 
              << " A2B entries " <<  m_A2B.size() 
              << std::endl   
              ;


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



