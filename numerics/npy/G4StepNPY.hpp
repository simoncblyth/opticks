#pragma once

#include "assert.h"
#include "uif.h"

#include <map>
#include <string>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


namespace fs = boost::filesystem;
namespace pt = boost::property_tree;


class NPY ;

// resist temptation to use inheritance here, 
// it causes much grief for little benefit 
// instead using "friend class" status to 
// give G4StepNPY access to innards of NPY
//
 
class G4StepNPY {

   typedef std::map<unsigned int, unsigned int> Lookup_t ;
   typedef std::map<std::string, unsigned int> Map_t ;

   public:  
       G4StepNPY(NPY* npy);

   public:  
       void dump(const char* msg);
       Map_t parseTree(boost::property_tree::ptree& tree, const char* prefix) ;
       void dumpMap(const char* msg, Map_t& map);

       void createLookup(const char* apath, const char* aprefix, const char* bpath, const char* bprefix);
       std::map<unsigned int, unsigned int> _createLookup(const char* apath, const char* aprefix, const char* bpath, const char* bprefix);
       std::map<unsigned int, unsigned int> _createLookup(Map_t& a, Map_t&b);
       std::string find(Map_t& m, unsigned int code);

       int a2b(unsigned int a);
       int b2a(unsigned int b);
       int lookup(Lookup_t& lkup, unsigned int x); 

   private:
        NPY* m_npy ; 
        Lookup_t m_a2b ; 
        Lookup_t m_b2a ; 
 
};





G4StepNPY::G4StepNPY(NPY* npy) :  m_npy(npy)
{
}

//
// hmm CerenkovStep and ScintillationStep have same shapes but different meanings see
//     /usr/local/env/chroma_env/src/chroma/chroma/cuda/cerenkov.h
//     /usr/local/env/chroma_env/src/chroma/chroma/cuda/scintillation.h
//
//  but whats needed for visualization should be in the same locations ?
//

void G4StepNPY::dump(const char* msg)
{
    if(!m_npy) return ;

    printf("%s\n", msg);

    unsigned int ni = m_npy->m_len0 ;
    unsigned int nj = m_npy->m_len1 ;
    unsigned int nk = m_npy->m_len2 ;
    std::vector<float>& data = m_npy->m_data ; 

    printf(" ni %u nj %u nk %u nj*nk %u \n", ni, nj, nk, nj*nk ); 

    uif_t uif ; 

    unsigned int check = 0 ;
    for(unsigned int i=0 ; i<ni ; i++ ){
    for(unsigned int j=0 ; j<nj ; j++ )
    {
       bool out = i == 0 || i == ni-1 ; 
       if(out) printf(" (%5u,%5u) ", i,j );
       for(unsigned int k=0 ; k<nk ; k++ )
       {
           unsigned int index = i*nj*nk + j*nk + k ;
           if(out)
           {
               uif.f = data[index] ;
               if( j == 0 || (j == 3 && k == 0)) printf(" %15d ",   uif.i );
               else         printf(" %15.3f ", uif.f );
           }
           assert(index == check);
           check += 1 ; 
       }
       if(out)
       {
           if( j == 0 ) printf(" sid/parentId/materialIndex/numPhotons ");
           if( j == 1 ) printf(" position/time ");
           if( j == 2 ) printf(" deltaPosition/stepLength ");
           if( j == 3 ) printf(" code ");

           printf("\n");
       }
    }
    }
}


void G4StepNPY::createLookup(const char* apath, const char* aprefix, const char* bpath, const char* bprefix)
{
    m_a2b = _createLookup(apath, aprefix, bpath, bprefix);
    m_b2a = _createLookup(bpath, bprefix, apath, aprefix);
}

std::map<unsigned int,unsigned int> G4StepNPY::_createLookup(const char* apath, const char* aprefix, const char* bpath, const char* bprefix)
{
    boost::property_tree::ptree atree ; 
    pt::read_json(apath, atree );
    Map_t a = parseTree(atree,aprefix);


    printf("A   %lu entries from %s\n", a.size(), apath);  

    boost::property_tree::ptree btree ; 
    pt::read_json(bpath, btree );
    Map_t b = parseTree(btree,bprefix);
    printf("B   %lu entries from %s\n", b.size(), bpath);  

    Lookup_t a2b = _createLookup(a, b);
    printf("A2B %lu entries in lookup  \n", a2b.size() );  

    for(Lookup_t::iterator it=a2b.begin() ; it != a2b.end() ; it++)
    {
        unsigned int acode = it->first ;  
        unsigned int bcode = it->second ;

        std::string aname = find(a, acode);
        std::string bname = find(b, bcode);

        printf("  A %4u : %25s  B %4u : %25s \n", acode, aname.c_str(), bcode, bname.c_str() );
  
    }
    return a2b ;
}


std::map<unsigned int, unsigned int> G4StepNPY::_createLookup(Map_t& a, Map_t&b)
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


std::map<std::string, unsigned int> G4StepNPY::parseTree(boost::property_tree::ptree& tree, const char* prefix)
{
    if(!prefix) prefix="";
    Map_t name2code ; 
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

void G4StepNPY::dumpMap(const char* msg, Map_t& map)
{
    printf("G4StepNPY::dumpMap %s \n", msg);
    for(Map_t::iterator it=map.begin() ; it != map.end() ; it++)
    {
        std::string name = it->first ;
        unsigned int code = it->second ; 
        printf("   %25s : %u \n", name.c_str(), code);
    }
}

std::string G4StepNPY::find(Map_t& m, unsigned int code)
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


int G4StepNPY::a2b(unsigned int a)
{
    return lookup(m_a2b, a );
}
int G4StepNPY::b2a(unsigned int b)
{
    return lookup(m_b2a, b );
}
int G4StepNPY::lookup(Lookup_t& lkup, unsigned int x)
{
    return lkup.find(x) == lkup.end() ? -1 : lkup[x] ;
}




