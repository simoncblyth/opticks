#include <cstring>
#include <iostream>
#include <sstream>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include "BMap.hh"

#include "NLookup.hpp"
#include "PLOG.hh"

namespace fs = boost::filesystem;

NLookup::NLookup()
   :
   m_alabel(NULL),
   m_blabel(NULL),
   m_closed(false)
{
}

const std::map<std::string, unsigned int>& NLookup::getA()
{
    return m_A ; 
}
const std::map<std::string, unsigned int>& NLookup::getB()
{
    return m_B ; 
}


void NLookup::mockup(const char* dir, const char* aname, const char* bname)
{
    mockA(dir, aname);
    mockB(dir, bname);
}
void NLookup::mockA(const char* adir, const char* aname)
{
    MSU mp ; 
    mp["/dd/Materials/red"] = 10 ; 
    mp["/dd/Materials/green"] = 20 ; 
    mp["/dd/Materials/blue"] = 30 ; 
    BMap<std::string, unsigned int>::save(&mp, adir, aname );
}
void NLookup::mockB(const char* bdir, const char* bname)
{
    MSU mp; 
    mp["red"] = 1 ; 
    mp["green"] = 2 ; 
    mp["blue"] = 3 ; 
    BMap<std::string, unsigned int>::save(&mp, bdir, bname );
}


void NLookup::setA( const char* json )
{
    BMap<std::string, unsigned>::LoadJSONString(&m_A, json, 0 );
    BMap<std::string, unsigned>::dump(&m_A, "NLookup::setA" );
}


void NLookup::setA( const std::map<std::string, unsigned int>& A, const char* aprefix, const char* alabel )
{
    assert(!m_closed);

    m_alabel = strdup(alabel);
    LOG(debug) << "NLookup::setA " << alabel  ; 

    for(MSU::const_iterator it=A.begin() ; it != A.end() ; it++)
    {
        const char* name = it->first.c_str() ; 

        if(aprefix)
        { 
            if(strncmp(name, aprefix, strlen(aprefix)) == 0)
            {
                std::string shortname = name + strlen(aprefix) ;
                m_A[shortname] = it->second ;  
            }
        }
        else
        {
            std::string shortname = name  ;
            m_A[shortname] = it->second ;  
        }
    }
}

void NLookup::setB( const std::map<std::string, unsigned int>& B, const char* bprefix, const char* blabel)
{
    assert(!m_closed);

    m_blabel = strdup(blabel);
    LOG(debug) << "NLookup::setB " << blabel  ; 
    for(MSU::const_iterator it=B.begin() ; it != B.end() ; it++)
    {
        const char* name = it->first.c_str() ; 
        
        if(bprefix)
        {
            if(strncmp(name, bprefix, strlen(bprefix)) == 0)
            {
                std::string shortname = name + strlen(bprefix) ;
                m_B[shortname] = it->second ;  
            }
        }
        else
        {
            std::string shortname = name  ;
            m_B[shortname] = it->second ;  
        }
    }
}

std::string NLookup::brief()
{
    std::stringstream ss ; 

    ss 
        << " A " << ( m_alabel ? m_alabel : "NULL" ) << " " <<  m_A.size()
        << " B " << ( m_blabel ? m_blabel : "NULL" ) << " " <<  m_B.size() 
        ;
 
    return ss.str();
}


void NLookup::loadA(const char* adir, const char* aname, const char* aprefix)
{
    MSU A ; 
    BMap<std::string, unsigned int>::load(&A, adir, aname );
    setA(A, aprefix, "NLookup::loadA");
}

void NLookup::loadB(const char* bdir, const char* bname, const char* /*bprefix*/)
{
    MSU B ; 
    BMap<std::string, unsigned int>::load(&B, bdir, bname);
    setB(B, "", "NLookup::loadB");
}


bool NLookup::isClosed() const
{
    return m_closed ; 
}

void NLookup::close(const char* msg)
{
    if(isClosed())
    {
        LOG(info) << "NLookup::close " << msg 
                  << " CLOSED ALREADY : SKIPPING "
                  ;

    }

    LOG(info) << msg << brief() ;

    assert(m_alabel && m_blabel) ; // have to setA and setB before close

    crossReference();

    //dump(msg);
}

void NLookup::crossReference()
{
    assert(!m_closed);
    m_closed = true ; 
    m_A2B = create(m_A, m_B);
    m_B2A = create(m_B, m_A);
}


std::string NLookup::acode2name(unsigned int acode)
{
    return find(m_A, acode);
}
std::string NLookup::bcode2name(unsigned int bcode)
{
    return find(m_B, bcode);
}


void NLookup::dump(const char* msg)
{
    std::cout << msg
              << " A entries " <<  m_A.size() 
              << " B entries " <<  m_B.size() 
              << " A2B entries " <<  m_A2B.size() 
              << std::endl   
              ;


    for(NLookup_t::iterator it=m_A2B.begin() ; it != m_A2B.end() ; it++)
    {
        unsigned int acode = it->first ;  
        unsigned int bcode = it->second ;
        std::string aname = acode2name(acode);
        std::string bname = bcode2name(bcode);
        printf("  A %4u : %25s  B %4u : %25s \n", acode, aname.c_str(), bcode, bname.c_str() );
    }
}


std::map<unsigned int, unsigned int> NLookup::create(MSU& a, MSU&b)
{
    assert(m_closed);
    // cross referencing codes which correspond to the same names
    NLookup_t a2b_ ;
    for(MSU::iterator ia=a.begin() ; ia != a.end() ; ia++)
    {
        std::string aname = ia->first ;
        unsigned int acode = ia->second ; 
        if(b.find(aname) != b.end())
        {
            unsigned int bcode = b[aname];
            a2b_[acode] = bcode ; 
        }
    }
    return a2b_ ; 
}

void NLookup::dumpMap(const char* msg, MSU& map)
{
    printf("NLookup::dumpMap %s \n", msg);
    for(MSU::iterator it=map.begin() ; it != map.end() ; it++)
    {
        std::string name = it->first ;
        unsigned int code = it->second ; 
        printf("   %25s : %u \n", name.c_str(), code);
    }
}

std::string NLookup::find(MSU& m, unsigned int code)
{
    std::string name ; 
    for(MSU::iterator im=m.begin() ; im != m.end() ; im++)
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


int NLookup::a2b(unsigned int a)
{
    return lookup(m_A2B, a );
}
int NLookup::b2a(unsigned int b)
{
    return lookup(m_B2A, b );
}
int NLookup::lookup(NLookup_t& lkup, unsigned int x)
{
    assert(m_closed);
    return lkup.find(x) == lkup.end() ? -1 : lkup[x] ;
}



