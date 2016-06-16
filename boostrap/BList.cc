#include "BList.hh"
#include "BTree.hh"

#include <iostream>

#include "regexsearch.hh"
#include "fsutil.hh"

#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <boost/algorithm/string.hpp> 
#include <boost/lexical_cast.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

namespace fs = boost::filesystem;
namespace pt = boost::property_tree;



template <typename A, typename B>
BList<A,B>::BList(std::vector<std::pair<A,B> >* vec) 
   :
   m_vec(vec)
{
} 


template<typename A, typename B> 
void BList<A,B>::save(  const char* dir, const char* name)
{
     std::string path = fsutil::preparePath(dir, name, true);
     LOG(debug) << "BList::save to " << path ;

     if(!path.empty()) save(path.c_str() );
}

template<typename A, typename B> 
void BList<A,B>::save(const char* path)
{
    pt::ptree t;
    for(typename std::vector<std::pair<A,B> >::iterator it=m_vec->begin() ; it != m_vec->end() ; it++)
    {
        t.put( 
           boost::lexical_cast<std::string>(it->first), 
           boost::lexical_cast<std::string>(it->second)
             ) ;
    }
    BTree::saveTree(t, path);
}


template<typename A, typename B> 
void BList<A,B>::load(const char* dir, const char* name)
{
    LOG(trace) << "load"
              << " dir [" << dir << "]" 
              << " name [" << name << "]" 
              ;

    std::string path = fsutil::preparePath(dir, name, false);
    if(!path.empty())
    {
        std::string shortpath = fsutil::prefixShorten( path.c_str(), "$LOCAL_BASE/env/geant4/geometry/export/" ); // cosmetic shortening only
        LOG(debug) << "loadMap " << shortpath  ;
        load(path.c_str() );
    }
    else
    {
        LOG(fatal)<< "loadList : no such directory " << dir ;
    }
}

template<typename A, typename B> 
void BList<A,B>::load(const char* path)
{
    pt::ptree t;
    BTree::loadTree(t, path );

    BOOST_FOREACH( pt::ptree::value_type const& ab, t.get_child("") )
    {
         A a = boost::lexical_cast<A>(ab.first.data());
         B b = boost::lexical_cast<B>(ab.second.data());
         m_vec->push_back( std::pair<A,B>(a,b) );
    }
}


template<typename A, typename B> 
void BList<A,B>::dump(  const char* msg)
{
    LOG(info) << msg ; 
    for(typename std::vector<std::pair<A,B> >::iterator it=m_vec->begin() ; it != m_vec->end() ; it++)
    {
         std::cout << std::setw(25) << boost::lexical_cast<std::string>(it->first)
                   << std::setw(50) << boost::lexical_cast<std::string>(it->second)
                   << std::endl ; 
                   ;
    }
}




template class BList<std::string, std::string>;






