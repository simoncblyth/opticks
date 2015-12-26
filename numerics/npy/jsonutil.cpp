#include "jsonutil.hpp"

#include "regexsearch.hh"

#include <string>
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



bool existsPath(const char* dir_, const char* name )
{
    std::string dir = os_path_expandvars(dir_) ; 
    fs::path fdir(dir);
    if(fs::exists(fdir) && fs::is_directory(fdir))
    {
        fs::path fpath(dir);
        fpath /= name ;
        return fs::exists(fpath ) && fs::is_regular_file(fpath) ; 
    }
  
    return false ; 
}



std::string preparePath(const char* dir_, const char* name, bool create )
{
    std::string dir = os_path_expandvars(dir_) ; 
    fs::path fdir(dir.c_str());
    if(!fs::exists(fdir) && create)
    {
        if (fs::create_directories(fdir))
        {
            LOG(info)<< "preparePath : created directory " << dir ;
        }
    }
    if(fs::exists(fdir) && fs::is_directory(fdir))
    {
        fs::path fpath(dir);
        fpath /= name ;
        return fpath.string();
    }
    else
    {
        LOG(warning)<< "preparePath : FAILED " 
                    << " dir " << dir 
                    << " dir_ " << dir_ 
                    << " name " << name ;
    }
    std::string empty ; 
    return empty ; 
}


template<typename A, typename B> 
void saveMap( typename std::map<A,B> & mp, const char* dir, const char* name)
{
     std::string path = preparePath(dir, name, true);
     LOG(debug) << "saveMap to " << path ;

     if(!path.empty()) saveMap( mp, path.c_str() );
}



template<typename A, typename B> 
void saveList( typename std::vector<std::pair<A,B> > & vp, const char* dir, const char* name)
{
     std::string path = preparePath(dir, name, true);
     LOG(debug) << "saveList to " << path ;

     if(!path.empty()) saveList( vp, path.c_str() );
}


void saveTree(const pt::ptree& t , const char* path);


template<typename A, typename B> 
void saveList( typename std::vector<std::pair<A,B> > & vp, const char* path)
{
    pt::ptree t;
    for(typename std::vector<std::pair<A,B> >::iterator it=vp.begin() ; it != vp.end() ; it++)
    {
        t.put( 
           boost::lexical_cast<std::string>(it->first), 
           boost::lexical_cast<std::string>(it->second)
             ) ;
    }
    saveTree(t, path);
}


template<typename A, typename B> 
void saveMap( typename std::map<A,B> & mp, const char* path)
{
    pt::ptree t;
    for(typename std::map<A,B>::iterator it=mp.begin() ; it != mp.end() ; it++)
    {
        t.put( 
           boost::lexical_cast<std::string>(it->first), 
           boost::lexical_cast<std::string>(it->second)
             ) ;
    }
    saveTree(t, path);
}


void saveTree(const pt::ptree& t , const char* path)
{
    fs::path fpath(path);
    std::string ext = fpath.extension().string();
    if(ext.compare(".json")==0)
        pt::write_json(path, t );
    else if(ext.compare(".ini")==0)
        pt::write_ini(path, t );
    else
        LOG(warning) << "saveTree cannot write to path with extension " << ext ; 
}



void loadTree(pt::ptree& t , const char* path)
{
    fs::path fpath(path);
    LOG(debug) << "jsonutil.loadTree: "
              << " load path: " << path;
    if ( not (fs::exists(fpath ) && fs::is_regular_file(fpath)) ) {
        LOG(warning) << "jsonutil.loadTree: "
                     << "can't find file " << path;
        return;
    }
    std::string ext = fpath.extension().string();
    if(ext.compare(".json")==0)
        pt::read_json(path, t );
    else if(ext.compare(".ini")==0)
        pt::read_ini(path, t );
    else
        LOG(warning) << "readTree cannot read path with extension " << ext ; 
}







std::string prefixShorten( const char* path, const char* prefix_)
{
    std::string prefix = os_path_expandvars(prefix_);  
    if(strncmp(path, prefix.c_str(), strlen(prefix.c_str()))==0)
        return path + strlen(prefix.c_str()) ;
    else
        return path  ;
}


template<typename A, typename B> 
void loadMap( typename std::map<A,B> & mp, const char* dir, const char* name)
{
    std::string path = preparePath(dir, name, false);
    if(!path.empty())
    {
        std::string shortpath = prefixShorten( path.c_str(), "$LOCAL_BASE/env/geant4/geometry/export/" ); // cosmetic shortening only
        LOG(debug) << "loadMap " << shortpath  ;
        loadMap( mp, path.c_str() );
    }
    else
    {
        LOG(fatal)<< "loadMap : no such directory " << dir ;
    }
}



template<typename A, typename B> 
void loadList( typename std::vector<std::pair<A,B> >& vp, const char* dir, const char* name)
{
    std::string path = preparePath(dir, name, false);
    if(!path.empty())
    {
        std::string shortpath = prefixShorten( path.c_str(), "$LOCAL_BASE/env/geant4/geometry/export/" ); // cosmetic shortening only
        LOG(debug) << "loadMap " << shortpath  ;
        loadList( vp, path.c_str() );
    }
    else
    {
        LOG(fatal)<< "loadList : no such directory " << dir ;
    }
}






template<typename A, typename B> 
void loadList( typename std::vector<std::pair<A,B> > & vp, const char* path)
{
    pt::ptree t;
    loadTree(t, path );

    BOOST_FOREACH( pt::ptree::value_type const& ab, t.get_child("") )
    {
         A a = boost::lexical_cast<A>(ab.first.data());
         B b = boost::lexical_cast<B>(ab.second.data());
         vp.push_back( std::pair<A,B>(a,b) );
    }
}

template<typename A, typename B> 
void loadMap( typename std::map<A,B> & mp, const char* path)
{
    pt::ptree t;
    loadTree(t, path );

    BOOST_FOREACH( pt::ptree::value_type const& ab, t.get_child("") )
    {
         A a = boost::lexical_cast<A>(ab.first.data());
         B b = boost::lexical_cast<B>(ab.second.data());
         mp[a] = b ;         
    }
}





template<typename A, typename B> 
void dumpMap( typename std::map<A,B> & mp, const char* msg)
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



template<typename A, typename B> 
void dumpList( typename std::vector<std::pair<A,B> > & vp, const char* msg)
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






// explicit instantiation of template functions, 
// allowing declaration and definition to reside in separate header and implementation files

///////////////  US /////////////////

template void saveMap<unsigned int, std::string>(std::map<unsigned int, std::string>& mp, const char* dir, const char* name) ;
template void saveMap<unsigned int, std::string>(std::map<unsigned int, std::string>& mp, const char* path) ;

template void loadMap<unsigned int, std::string>(std::map<unsigned int, std::string>& mp, const char* dir, const char* name) ;
template void loadMap<unsigned int, std::string>(std::map<unsigned int, std::string>& mp, const char* path) ;

template void dumpMap<unsigned int, std::string>(std::map<unsigned int, std::string>& mp, const char* msg) ;



///////////////  SU /////////////////

template void saveMap<std::string, unsigned int>(std::map<std::string, unsigned int>& mp, const char* dir, const char* name ) ;
template void saveMap<std::string, unsigned int>(std::map<std::string, unsigned int>& mp, const char* path) ;

template void loadMap<std::string, unsigned int>(std::map<std::string, unsigned int>& mp, const char* dir, const char* name ) ;
template void loadMap<std::string, unsigned int>(std::map<std::string, unsigned int>& mp, const char* path) ;

template void dumpMap<std::string, unsigned int>(std::map<std::string, unsigned int>& mp, const char* msg) ;


template void saveList<std::string, unsigned int>(std::vector<std::pair<std::string, unsigned int> >& vp, const char* dir, const char* name) ;
template void saveList<std::string, unsigned int>(std::vector<std::pair<std::string, unsigned int> >& vp, const char* path) ;

template void loadList<std::string, unsigned int>(std::vector<std::pair<std::string, unsigned int> >& vp, const char* dir, const char* name) ;
template void loadList<std::string, unsigned int>(std::vector<std::pair<std::string, unsigned int> >& vp, const char* path) ;

template void dumpList<std::string, unsigned int>(std::vector<std::pair<std::string, unsigned int> >& vp, const char* msg) ;

///////////////  SD /////////////////

template void saveList<std::string, double>(std::vector<std::pair<std::string, double> >& vp, const char* dir, const char* name) ;
template void saveList<std::string, double>(std::vector<std::pair<std::string, double> >& vp, const char* path) ;

template void loadList<std::string, double>(std::vector<std::pair<std::string, double> >& vp, const char* dir, const char* name) ;
template void loadList<std::string, double>(std::vector<std::pair<std::string, double> >& vp, const char* path) ;



///////////////  SS /////////////////

template void saveList<std::string, std::string>(std::vector<std::pair<std::string, std::string> >& vp, const char* dir, const char* name) ;
template void saveList<std::string, std::string>(std::vector<std::pair<std::string, std::string> >& vp, const char* path) ;

template void loadList<std::string, std::string>(std::vector<std::pair<std::string, std::string> >& vp, const char* dir, const char* name) ;
template void loadList<std::string, std::string>(std::vector<std::pair<std::string, std::string> >& vp, const char* path) ;




///////////////  SS /////////////////


template void saveMap<std::string, std::string>(std::map<std::string, std::string>& mp, const char* dir, const char* name ) ;
template void saveMap<std::string, std::string>(std::map<std::string, std::string>& mp, const char* path) ;

template void loadMap<std::string, std::string>(std::map<std::string, std::string>& mp, const char* dir, const char* name ) ;
template void loadMap<std::string, std::string>(std::map<std::string, std::string>& mp, const char* path) ;

template void dumpMap<std::string, std::string>(std::map<std::string, std::string>& mp, const char* msg) ;


