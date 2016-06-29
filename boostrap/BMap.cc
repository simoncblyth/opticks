#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <boost/algorithm/string.hpp> 
#include <boost/lexical_cast.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include "BJSONParser.hh"

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

#include "PLOG.hh"
// trace/debug/info/warning/error/fatal

namespace fs = boost::filesystem;
namespace pt = boost::property_tree;


#include "BMap.hh"
#include "BTree.hh"
#include "BFile.hh"

#ifdef _MSC_VER
// foreach shadowing
#pragma warning( disable : 4456)
#endif


template <typename A, typename B>
void BMap<A,B>::save( std::map<A,B>* mp, const char* dir, const char* name) 
{
    BMap<A,B> bmp(mp);
    bmp.save(dir, name);  
}

template <typename A, typename B>
void BMap<A,B>::save( std::map<A,B>* mp, const char* path) 
{
    BMap<A,B> bmp(mp);
    bmp.save(path);  
}

template <typename A, typename B>
int  BMap<A,B>::load( std::map<A,B>* mp, const char* dir, const char* name, unsigned int depth) 
{
    BMap<A,B> bmp(mp);
    return bmp.load(dir, name, depth);  
}

template <typename A, typename B>
int  BMap<A,B>::load( std::map<A,B>* mp, const char* path, unsigned int depth )
{
    BMap<A,B> bmp(mp);
    return bmp.load(path, depth);  
}

template <typename A, typename B>
void BMap<A,B>::dump( std::map<A,B>* mp, const char* msg) 
{
    BMap<A,B> bmp(mp);
    bmp.dump(msg);  
}


template <typename A, typename B>
BMap<A,B>::BMap(std::map<A,B>* mp) 
   :
   m_map(mp)
{
} 


template <typename A, typename B>
void BMap<A,B>::save(const char* dir, const char* name)
{
     std::string path = BFile::preparePath(dir, name, true);
     LOG(debug) << "saveMap to " << path ;
     if(!path.empty()) save( path.c_str() );
}


template<typename A, typename B> 
void BMap<A,B>::save( const char* path)
{
    pt::ptree t;
    for(typename std::map<A,B>::iterator it=m_map->begin() ; it != m_map->end() ; it++)
    {
        t.put( 
           boost::lexical_cast<std::string>(it->first), 
           boost::lexical_cast<std::string>(it->second)
             ) ;
    }
    BTree::saveTree(t, path);
}





template<typename A, typename B> 
int BMap<A,B>::load( const char* dir, const char* name, unsigned int depth)
{
    int rc(0) ; 
    std::string path = BFile::preparePath(dir, name, false);
    if(!path.empty())
    {
        std::string shortpath = BFile::prefixShorten( path.c_str(), "$LOCAL_BASE/env/geant4/geometry/export/" ); // cosmetic shortening only
        LOG(debug) << "loadMap " << shortpath  ;
        rc = load( path.c_str(), depth );
    }
    else
    {
        LOG(fatal)<< "loadMap : no such directory " << dir ;
    }
    return rc ;
}



template<typename A, typename B> 
int BMap<A,B>::load(const char* path, unsigned int depth)
{
    pt::ptree t;
    int rc = BTree::loadTree(t, path );

    if(depth == 0)
    {
        BOOST_FOREACH( pt::ptree::value_type const& ab, t.get_child("") )
        {
             A a = boost::lexical_cast<A>(ab.first.data());
             B b = boost::lexical_cast<B>(ab.second.data());
             (*m_map)[a] = b ; 
        }
    } 
    else if(depth == 1)
    {

        BOOST_FOREACH( pt::ptree::value_type const& skv, t.get_child("") )
        {
            A s = boost::lexical_cast<A>(skv.first.data());

            BOOST_FOREACH( pt::ptree::value_type const& kv, skv.second.get_child("") ) 
            {   
                std::string k = kv.first.data();
                std::stringstream ss ; 
                ss << s << "." << k ; 

                A key = boost::lexical_cast<A>(ss.str()) ;  
               // defer the problem if A is not std::string to runtime
                B val = boost::lexical_cast<B>(kv.second.data()) ;  

                LOG(debug) << "loadMap(1) " 
                          << std::setw(15) << s  
                          << std::setw(15) << k  
                          << std::setw(30) << key 
                          << std::setw(50) << val
                          ;  

                (*m_map)[key] = val ; 
            }   
        }
    }
    else
        assert(0) ;

    return rc ; 
}



template<typename A, typename B> 
void BMap<A,B>::dump( const char* msg)
{
    LOG(info) << msg ; 
    for(typename std::map<A,B>::iterator it=m_map->begin() ; it != m_map->end() ; it++)
    {
         std::cout << std::setw(25) << boost::lexical_cast<std::string>(it->first)
                   << std::setw(50) << boost::lexical_cast<std::string>(it->second)
                   << std::endl ; 
                   ;
    }
}



template class BMap<std::string, std::string>;
template class BMap<std::string, unsigned int>;
template class BMap<unsigned int, std::string>;




