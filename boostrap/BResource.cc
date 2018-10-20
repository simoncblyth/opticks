#include <iostream>
#include <iomanip>

#include "BResource.hh"
#include "BFile.hh"
#include "PLOG.hh"

const BResource* BResource::INSTANCE = NULL ; 

const BResource* BResource::GetInstance()
{
    return INSTANCE ; 
}

const char* BResource::Get(const char* label)
{
    const BResource* br = GetInstance(); 
    const char* ret = NULL ; 
    if( !br ) return ret ;  

    if( ret == NULL ) ret = br->getPath(label); 
    if( ret == NULL ) ret = br->getDir(label); 
    if( ret == NULL ) ret = br->getName(label); 

    LOG(debug)
         << " label " << label 
         << " ret " << ret 
         ;
 

    return ret ;
}


BResource::BResource()
{
    INSTANCE=this ; 
}

BResource::~BResource()
{
}

const char* BResource::getPath(const char* label) const { return get(label, m_paths); }
const char* BResource::getDir(const char* label) const { return get(label, m_dirs); }
const char* BResource::getName(const char* label) const { return get(label, m_names); }


const char* BResource::get(const char* label, const std::vector<std::pair<std::string, std::string>>& vss) const 
{
    typedef std::pair<std::string, std::string> SS ; 
    typedef std::vector<SS> VSS ; 

    const char* path = NULL ; 
 
    for(VSS::const_iterator it=vss.begin() ; it != vss.end() ; it++)
    {
        const SS& ss = *it ;
        if(ss.first.compare(label) == 0) 
        {
            path = ss.second.c_str() ; 
        }
    }
    return path ; 
}

void BResource::addName( const char* label, const char* name)
{
    typedef std::pair<std::string, std::string> SS ; 
    m_names.push_back( SS(label, name ? name : "") );
}
void BResource::addPath( const char* label, const char* path)
{
    typedef std::pair<std::string, std::string> SS ; 
    m_paths.push_back( SS(label, path ? path : "") );
}
void BResource::addDir( const char* label, const char* dir)
{
    typedef std::pair<std::string, std::string> SS ; 
    m_dirs.push_back( SS(label, dir ? dir : "" ) );
}

void BResource::dumpNames(const char* msg) const 
{
    LOG(info) << msg ; 

    typedef std::pair<std::string, std::string> SS ; 
    typedef std::vector<SS> VSS ; 

    for(VSS::const_iterator it=m_names.begin() ; it != m_names.end() ; it++)
    {
        const char* label = it->first.c_str() ; 
        const char* name = it->second.empty() ? NULL : it->second.c_str() ; 
        std::cerr
             << std::setw(30) << label
             << " : " 
             << std::setw(2) << "-" 
             << " : " 
             << std::setw(50) << ( name ? name : "-" )
             << std::endl 
             ;
    }
}

void BResource::dumpPaths(const char* msg) const 
{
    LOG(info) << msg ; 

    typedef std::pair<std::string, std::string> SS ; 
    typedef std::vector<SS> VSS ; 

    for(VSS::const_iterator it=m_paths.begin() ; it != m_paths.end() ; it++)
    {
        const char* name = it->first.c_str() ; 
        const char* path = it->second.empty() ? NULL : it->second.c_str() ; 

        bool exists = path ? BFile::ExistsFile(path ) : false ; 

        const char* path2 = getPath(name) ; 

        std::cerr
             << std::setw(30) << name
             << " : " 
             << std::setw(2) << ( exists ? "Y" : "N" ) 
             << " : " 
             << std::setw(50) << ( path ? path : "-" )
             << std::endl 
             ;


        bool match = path2 == path ;         
        if(!match) LOG(fatal) 
                    << " path [" << path << "] " 
                    << " path2 [" << path2 << "] "
                    ;

        assert( match );


    } 
}


void BResource::dumpDirs(const char* msg) const 
{
    LOG(info) << msg ; 

    typedef std::pair<std::string, std::string> SS ; 
    typedef std::vector<SS> VSS ; 

    for(VSS::const_iterator it=m_dirs.begin() ; it != m_dirs.end() ; it++)
    {
        const char* name = it->first.c_str() ; 
        const char* dir = it->second.empty() ? NULL : it->second.c_str() ; 
        bool exists = dir ? BFile::ExistsDir(dir ) : false ; 

        std::cerr
             << std::setw(30) << name
             << " : " 
             << std::setw(2) << ( exists ? "Y" : "N" ) 
             << " : "  
             << std::setw(50) << ( dir ? dir : "-") 
             << std::endl 
             ;
    } 
}


