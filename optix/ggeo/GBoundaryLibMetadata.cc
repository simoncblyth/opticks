#include "GBoundaryLibMetadata.hh"
#include "GBoundaryLib.hh"
#include "GBoundary.hh"
#include "GPropertyMap.hh"

#include "stdio.h"
#include "string.h"

#include <map>
#include <string>
#include <iostream>
#include <iomanip>


#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>

#include "stringutil.hpp"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



namespace fs = boost::filesystem;
namespace pt = boost::property_tree;

const char* GBoundaryLibMetadata::filename = "GBoundaryLibMetadata.json" ; 
const char* GBoundaryLibMetadata::mapname  = "GBoundaryLibMetadataMaterialMap.json" ; 

GBoundaryLibMetadata::GBoundaryLibMetadata() 
{
}

void GBoundaryLibMetadata::add(const char* kfmt, unsigned int isub, const char* cat, const char* tag, const char* val)
{
    char key[128];
    snprintf(key, 128, kfmt, isub, cat, tag );

    LOG(info) << "GBoundaryLibMetadata::add"
              << " key " << key 
              << " val " << val 
              ; 
    m_tree.add(key, val);
}


void GBoundaryLibMetadata::addDigest(const char* kfmt, unsigned int isub, const char* cat, const char* dig )
{
    add(kfmt, isub, cat, "digest",  dig );
}



void GBoundaryLibMetadata::add(const char* kfmt, unsigned int isub, const char* cat, GPropertyMap<float>* pmap )
{
    if(!pmap) return ;

    const char* name = pmap->getName() ;
    const char* type = pmap->getType() ;
    std::string keys = pmap->getKeysString() ;

    // "lib.boundary.%d.%s.%s" ;
    add(kfmt, isub, cat, "name",    name); 
    add(kfmt, isub, cat, "type",    type);
    add(kfmt, isub, cat, "keys",    keys.c_str());

    if(strcmp(cat, "imat") == 0 || strcmp(cat, "omat") == 0)
    {
        const char* shortname = pmap->getShortName() ; 
        add(kfmt, isub, cat, "shortname", shortname); 
        //free(shortname);
    }
}
void GBoundaryLibMetadata::addMaterial(unsigned int isub, const char* cat, const char* shortname, const char* digest )
{
    bool imat = strcmp(cat, "imat") == 0 ;
    bool omat = strcmp(cat, "omat") == 0 ;
    assert(imat || omat) ;  

    unsigned int code = isub*10 ;
    if(omat) code += 1 ; 

    char key[128];
    snprintf(key, 128, "lib.material.%s.%s.%u", shortname, cat, code );
    m_tree.add(key, digest);

    snprintf(key, 128, "lib.material.%s.%s.%u", shortname, "mat", code );
    m_tree.add(key, digest);
}



std::string GBoundaryLibMetadata::get(const char* kfmt, const char* idx)
{
    char key[128];
    snprintf(key, 128, kfmt, idx);
    return m_tree.get<std::string>(key);
}
std::string GBoundaryLibMetadata::get(const char* kfmt, unsigned int idx)
{
    char key[128];
    snprintf(key, 128, kfmt, idx);
    return m_tree.get<std::string>(key);
}


std::string GBoundaryLibMetadata::getBoundaryQtyByIndex(unsigned int isub, unsigned int icat, const char* tag)
{
    unsigned int numQuad = GBoundaryLib::getNumQuad();
    assert(icat < numQuad); 
    const char* cat = GBoundary::getConstituentNameByIndex(icat);
    return getBoundaryQty(isub, cat, tag);
}

std::string GBoundaryLibMetadata::getBoundaryQty(unsigned int isub, const char* cat, const char* tag)
{
    char key[128];
    snprintf(key, 128, "lib.boundary.%u.%s.%s", isub, cat, tag);
    return m_tree.get<std::string>(key);
}


unsigned int GBoundaryLibMetadata::getNumBoundary()
{
    unsigned int count(0);
    BOOST_FOREACH( boost::property_tree::ptree::value_type const& ak, m_tree.get_child("lib.boundary") ) 
    {
        unsigned int index = boost::lexical_cast<unsigned int>(ak.first.c_str());
        assert(index == count);
        count++;
    }
    return count ;
}

unsigned int GBoundaryLibMetadata::getBoundaryCode(unsigned int isub)
{
    return isub + 1 ;   
}
std::string GBoundaryLibMetadata::getBoundaryName(unsigned int isub)
{
    std::string imat = getBoundaryQty(isub, "imat", "shortname") ;
    std::string omat = getBoundaryQty(isub, "omat", "shortname") ;
    std::string isur = getBoundaryQty(isub, "isur", "name") ;
    std::string osur = getBoundaryQty(isub, "osur", "name") ;

    if(isur == "?" ) isur = "" ;
    if(osur == "?" ) osur = "" ;

    std::string isur_s = patternPickField(isur, "__", -1);
    std::string osur_s = patternPickField(osur, "__", -1);

    std::vector<std::string> vals ; 
    vals.push_back(imat);
    vals.push_back(omat);
    vals.push_back(isur_s);
    vals.push_back(osur_s);

    return boost::algorithm::join(vals, ".");
}


std::map<int, std::string> GBoundaryLibMetadata::getBoundaryNames()
{
    std::map<int, std::string> nmap ; 
    unsigned int nsub = getNumBoundary();
    nmap[0] = "Miss" ;
    for(unsigned int isub=0 ; isub < nsub ; isub++)
    {
        std::string name = getBoundaryName(isub);
        unsigned int code = getBoundaryCode(isub);
        assert(code != 0);
        nmap[code] = name ;        
    }
    return nmap ; 
}


void GBoundaryLibMetadata::dumpNames()
{
    unsigned int nsub = getNumBoundary();
    for(unsigned int isub=0 ; isub < nsub ; isub++)
    {
        std::string name = getBoundaryName(isub);
        printf("%2d : %s \n", isub, name.c_str());
    }
}




void GBoundaryLibMetadata::createMaterialMap()
{
    typedef std::map<std::string, unsigned int> Map_t ;
    Map_t name2line ; 

    char key[128];
    BOOST_FOREACH( boost::property_tree::ptree::value_type const& ak, m_tree.get_child("lib.material") ) 
    {
        std::string matname = ak.first ;
        snprintf(key, 128, "lib.material.%s.mat", matname.c_str());
        printf("GBoundaryLibMetadata::createMaterialMap %s \n", key);

        std::string digest ;     
        BOOST_FOREACH( boost::property_tree::ptree::value_type const& bk, m_tree.get_child(key) ) // absolute key
        {
            unsigned int code = boost::lexical_cast<unsigned int>(bk.first.c_str());
            const char*  dig = bk.second.data().c_str();

            unsigned int isub   = code / 10 ;
            unsigned int offset = code % 10 ; 
            unsigned int line   = GBoundaryLib::getLine(isub, offset) ;   // into the wavelengthBuffer 

            bool first = digest.empty();
            if(first)
            {
                digest = dig ;
            }
            else
            {
                LOG(info)<< __func__
                         << " digest " << digest                
                         << " dig " << dig 
                         ;                

                assert(strcmp(digest.c_str(), dig) == 0);
            }

            // only record line for 1st occurence of the name
            if(name2line.find(matname) == name2line.end()) name2line[matname] = line ; 

            //printf("   code %4u isub %3u offset %u line %u dig %s matname %s \n", code, isub, offset, line, dig, matname.c_str() );
        }
    }

    printf("GBoundaryLibMetadata::createMaterialMap\n");
    for(Map_t::iterator it=name2line.begin() ; name2line.end() != it ; it++)
    {
        printf(" %25s : %u \n", it->first.c_str(), it->second ); 
        m_material_map.add( it->first, it->second );
    }

}





void GBoundaryLibMetadata::Summary(const char* msg)
{    
    printf("%s\n", msg );

    // TODO: recursive dumping for greater flexibility  

    BOOST_FOREACH( boost::property_tree::ptree::value_type const& ak, m_tree.get_child("") ) 
    {   
        std::cout << ak.first << std::endl ;  // lib

        BOOST_FOREACH( boost::property_tree::ptree::value_type const& bk, ak.second.get_child("") ) 
        {   
            std::cout << bk.first << std::endl ;   // boundary

            BOOST_FOREACH( boost::property_tree::ptree::value_type const& ck, bk.second.get_child("") ) 
            {
                std::cout << ck.first << std::endl ;   // 1,2,3,...

                BOOST_FOREACH( boost::property_tree::ptree::value_type const& dk, ck.second.get_child("") ) 
                {
                    std::cout << dk.first << std::endl ;   // imat,omat,...

                    BOOST_FOREACH( boost::property_tree::ptree::value_type const& ek, dk.second.get_child("") ) 
                    {
                        std::string ev = ek.second.data();    // key, value pairs
                        std::cout << std::setw(15) << ek.first << " : " << ev << std::endl ;


                    }  // ek
                }      // dk
            }          // ck
        }              // bk
    }                  // ak

    
   
}



void GBoundaryLibMetadata::save(const char* dir)
{
    fs::path cachedir(dir);
    if(!fs::exists(cachedir))
    {
        if (fs::create_directory(cachedir))
        {
            printf("GBoundaryLibMetadata::save created directory %s \n", dir );
        }
    }

    if(fs::exists(cachedir) && fs::is_directory(cachedir))
    {
        fs::path treepath(dir);
        treepath /= filename ; 
        pt::write_json(treepath.string().c_str(), m_tree);

        fs::path mappath(dir);
        mappath /= mapname ; 
        pt::write_json(mappath.string().c_str(), m_material_map);

    }
    else
    {
        printf("GBoundaryLibMetadata::save directory %s DOES NOT EXIST \n", dir);
    }
}

void GBoundaryLibMetadata::read(const char* path)
{
    pt::read_json(path, m_tree);
}

GBoundaryLibMetadata* GBoundaryLibMetadata::load(const char* dir)
{
    GBoundaryLibMetadata* meta(NULL);
    fs::path cachedir(dir);
    if(!fs::exists(cachedir))
    {
        printf("GBoundaryLibMetadata::load directory %s DOES NOT EXIST \n", dir);
    }
    else
    {
        fs::path path(dir);
        path /= filename ; 
        meta = new GBoundaryLibMetadata  ;
        meta->read(path.string().c_str());
    }
    return meta ; 
}

