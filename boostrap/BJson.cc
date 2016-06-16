#include "BJson.hh"
#include "BTree.hh"

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




template<typename A, typename B> 
void BJson::saveList( typename std::vector<std::pair<A,B> > & vp, const char* dir, const char* name)
{
     std::string path = fsutil::preparePath(dir, name, true);
     LOG(debug) << "saveList to " << path ;

     if(!path.empty()) saveList( vp, path.c_str() );
}

template<typename A, typename B> 
void BJson::saveList( typename std::vector<std::pair<A,B> > & vp, const char* path)
{
    pt::ptree t;
    for(typename std::vector<std::pair<A,B> >::iterator it=vp.begin() ; it != vp.end() ; it++)
    {
        t.put( 
           boost::lexical_cast<std::string>(it->first), 
           boost::lexical_cast<std::string>(it->second)
             ) ;
    }
    BTree::saveTree(t, path);
}


template<typename A, typename B> 
void BJson::loadList( typename std::vector<std::pair<A,B> >& vp, const char* dir, const char* name)
{
    LOG(trace) << "loadList"
              << " dir [" << dir << "]" 
              << " name [" << name << "]" 
              ;

    std::string path = fsutil::preparePath(dir, name, false);
    if(!path.empty())
    {
        std::string shortpath = fsutil::prefixShorten( path.c_str(), "$LOCAL_BASE/env/geant4/geometry/export/" ); // cosmetic shortening only
        LOG(debug) << "loadMap " << shortpath  ;
        loadList( vp, path.c_str() );
    }
    else
    {
        LOG(fatal)<< "loadList : no such directory " << dir ;
    }
}

template<typename A, typename B> 
void BJson::loadList( typename std::vector<std::pair<A,B> > & vp, const char* path)
{
    pt::ptree t;
    BTree::loadTree(t, path );

    BOOST_FOREACH( pt::ptree::value_type const& ab, t.get_child("") )
    {
         A a = boost::lexical_cast<A>(ab.first.data());
         B b = boost::lexical_cast<B>(ab.second.data());
         vp.push_back( std::pair<A,B>(a,b) );
    }
}


template<typename A, typename B> 
void BJson::dumpList( typename std::vector<std::pair<A,B> > & vp, const char* msg)
{
    LOG(info) << msg ; 
    for(typename std::vector<std::pair<A,B> >::iterator it=vp.begin() ; it != vp.end() ; it++)
    {
         std::cout << std::setw(25) << boost::lexical_cast<std::string>(it->first)
                   << std::setw(50) << boost::lexical_cast<std::string>(it->second)
                   << std::endl ; 
                   ;
    }
}



















template<typename A, typename B> 
void BJson::saveMap( typename std::map<A,B> & mp, const char* dir, const char* name)
{
     std::string path = fsutil::preparePath(dir, name, true);
     LOG(debug) << "saveMap to " << path ;

     if(!path.empty()) saveMap( mp, path.c_str() );
}

template<typename A, typename B> 
void BJson::saveMap( typename std::map<A,B> & mp, const char* path)
{
    pt::ptree t;
    for(typename std::map<A,B>::iterator it=mp.begin() ; it != mp.end() ; it++)
    {
        t.put( 
           boost::lexical_cast<std::string>(it->first), 
           boost::lexical_cast<std::string>(it->second)
             ) ;
    }
    BTree::saveTree(t, path);
}

template<typename A, typename B> 
int BJson::loadMap( typename std::map<A,B> & mp, const char* dir, const char* name, unsigned int depth)
{
    int rc(0) ; 
    std::string path = fsutil::preparePath(dir, name, false);
    if(!path.empty())
    {
        std::string shortpath = fsutil::prefixShorten( path.c_str(), "$LOCAL_BASE/env/geant4/geometry/export/" ); // cosmetic shortening only
        LOG(debug) << "loadMap " << shortpath  ;
        rc = loadMap( mp, path.c_str(), depth );
    }
    else
    {
        LOG(fatal)<< "loadMap : no such directory " << dir ;
    }
    return rc ;
}



template<typename A, typename B> 
int BJson::loadMap( typename std::map<A,B> & mp, const char* path, unsigned int depth)
{
    pt::ptree t;
    int rc = BTree::loadTree(t, path );

    if(depth == 0)
    {
        BOOST_FOREACH( pt::ptree::value_type const& ab, t.get_child("") )
        {
             A a = boost::lexical_cast<A>(ab.first.data());
             B b = boost::lexical_cast<B>(ab.second.data());
             mp[a] = b ; 
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

                mp[key] = val ; 
            }   
        }
    }
    else
        assert(0) ;

    return rc ; 
}


template<typename A, typename B> 
void BJson::dumpMap( typename std::map<A,B> & mp, const char* msg)
{
    LOG(info) << msg ; 
    for(typename std::map<A,B>::iterator it=mp.begin() ; it != mp.end() ; it++)
    {
         std::cout << std::setw(25) << boost::lexical_cast<std::string>(it->first)
                   << std::setw(50) << boost::lexical_cast<std::string>(it->second)
                   << std::endl ; 
                   ;
    }
}






//  below not being exported/imported on windows ... so try a template class approach 
//  http://stackoverflow.com/questions/666628/importing-explicitly-instantiated-template-class-from-dll
//
// explicit instantiation of template functions, 
// allowing declaration and definition to reside in separate header and implementation files

///////////////  US /////////////////

template void BJson::saveMap<unsigned int, std::string>(std::map<unsigned int, std::string>& mp, const char* dir, const char* name) ;
template void BJson::saveMap<unsigned int, std::string>(std::map<unsigned int, std::string>& mp, const char* path) ;

template int BJson::loadMap<unsigned int, std::string>(std::map<unsigned int, std::string>& mp, const char* dir, const char* name, unsigned int depth) ;
template int BJson::loadMap<unsigned int, std::string>(std::map<unsigned int, std::string>& mp, const char* path, unsigned int depth ) ;

template void BJson::dumpMap<unsigned int, std::string>(std::map<unsigned int, std::string>& mp, const char* msg) ;



///////////////  SU /////////////////

template void BJson::saveMap<std::string, unsigned int>(std::map<std::string, unsigned int>& mp, const char* dir, const char* name ) ;
template void BJson::saveMap<std::string, unsigned int>(std::map<std::string, unsigned int>& mp, const char* path) ;

template int BJson::loadMap<std::string, unsigned int>(std::map<std::string, unsigned int>& mp, const char* dir, const char* name, unsigned int depth) ;
template int BJson::loadMap<std::string, unsigned int>(std::map<std::string, unsigned int>& mp, const char* path, unsigned int depth ) ;

template void BJson::dumpMap<std::string, unsigned int>(std::map<std::string, unsigned int>& mp, const char* msg) ;


template void BJson::saveList<std::string, unsigned int>(std::vector<std::pair<std::string, unsigned int> >& vp, const char* dir, const char* name) ;
template void BJson::saveList<std::string, unsigned int>(std::vector<std::pair<std::string, unsigned int> >& vp, const char* path) ;

template void BJson::loadList<std::string, unsigned int>(std::vector<std::pair<std::string, unsigned int> >& vp, const char* dir, const char* name) ;
template void BJson::loadList<std::string, unsigned int>(std::vector<std::pair<std::string, unsigned int> >& vp, const char* path) ;

template void BJson::dumpList<std::string, unsigned int>(std::vector<std::pair<std::string, unsigned int> >& vp, const char* msg) ;

///////////////  SD /////////////////

template void BJson::saveList<std::string, double>(std::vector<std::pair<std::string, double> >& vp, const char* dir, const char* name) ;
template void BJson::saveList<std::string, double>(std::vector<std::pair<std::string, double> >& vp, const char* path) ;

template void BJson::loadList<std::string, double>(std::vector<std::pair<std::string, double> >& vp, const char* dir, const char* name) ;
template void BJson::loadList<std::string, double>(std::vector<std::pair<std::string, double> >& vp, const char* path) ;



///////////////  SS /////////////////

template void BJson::saveList<std::string, std::string>(std::vector<std::pair<std::string, std::string> >& vp, const char* dir, const char* name) ;
template void BJson::saveList<std::string, std::string>(std::vector<std::pair<std::string, std::string> >& vp, const char* path) ;

template void BJson::loadList<std::string, std::string>(std::vector<std::pair<std::string, std::string> >& vp, const char* dir, const char* name) ;
template void BJson::loadList<std::string, std::string>(std::vector<std::pair<std::string, std::string> >& vp, const char* path) ;




///////////////  SS /////////////////


template void BJson::saveMap<std::string, std::string>(std::map<std::string, std::string>& mp, const char* dir, const char* name ) ;
template void BJson::saveMap<std::string, std::string>(std::map<std::string, std::string>& mp, const char* path) ;

template int BJson::loadMap<std::string, std::string>(std::map<std::string, std::string>& mp, const char* dir, const char* name, unsigned int depth) ;
template int BJson::loadMap<std::string, std::string>(std::map<std::string, std::string>& mp, const char* path, unsigned int depth) ;



template void BJson::dumpMap<std::string, std::string>(std::map<std::string, std::string>& mp, const char* msg) ;


