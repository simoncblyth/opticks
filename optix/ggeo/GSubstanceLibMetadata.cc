#include "GSubstanceLibMetadata.hh"
#include "GPropertyMap.hh"

#include "stdio.h"
#include "string.h"
#include <string>
#include <iostream>
#include <iomanip>


#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace fs = boost::filesystem;
namespace pt = boost::property_tree;

const char* GSubstanceLibMetadata::filename = "GSubstanceLibMetadata.json" ; 

GSubstanceLibMetadata::GSubstanceLibMetadata()
{
}

void GSubstanceLibMetadata::add(const char* kfmt, unsigned int isub, const char* cat, const char* tag, const char* val)
{
    char key[128];
    snprintf(key, 128, kfmt, isub, cat, tag );
    m_tree.add(key, val);
}


void GSubstanceLibMetadata::add(const char* kfmt, unsigned int isub, const char* cat, GPropertyMap* pmap )
{
    if(!pmap) return ;

    const char* name = pmap->getName() ;
    const char* type = pmap->getType() ;
    char* digest = pmap->digest() ;
    std::string keys = pmap->getKeysString() ;

    // "lib.substance.%d.%s.%s" ;
    add(kfmt, isub, cat, "name",    name); 
    add(kfmt, isub, cat, "type",    type);
    add(kfmt, isub, cat, "keys",    keys.c_str());
    add(kfmt, isub, cat, "digest",  digest );

    free(digest);

    if(strcmp(cat, "imat") == 0 || strcmp(cat, "omat") == 0)
    {
        char* shortname = pmap->getShortName("__dd__Materials__") ; 
        add(kfmt, isub, cat, "shortname", shortname); 
        free(shortname);
    }
}
void GSubstanceLibMetadata::addMaterial(unsigned int isub, const char* cat, GPropertyMap* pmap )
{
    char key[128];
    char* shortname = pmap->getShortName("__dd__Materials__") ; 
    char* digest = pmap->digest() ; 

    
    bool imat = strcmp(cat, "imat") == 0 ;
    bool omat = strcmp(cat, "omat") == 0 ;
    assert(imat || omat) ;  

    unsigned int code = isub*10 ;
    if(omat) code += 1 ; 

    snprintf(key, 128, "lib.material.%s.%s.%u", shortname, cat, code );
    m_tree.add(key, digest);

    snprintf(key, 128, "lib.material.%s.%s.%u", shortname, "mat", code );
    m_tree.add(key, digest);

    free(digest);
    free(shortname);
}



std::string GSubstanceLibMetadata::get(const char* kfmt, const char* idx)
{
    char key[128];
    snprintf(key, 128, kfmt, idx);
    return m_tree.get<std::string>(key);
}
std::string GSubstanceLibMetadata::get(const char* kfmt, unsigned int idx)
{
    char key[128];
    snprintf(key, 128, kfmt, idx);
    return m_tree.get<std::string>(key);
}

void GSubstanceLibMetadata::createMaterialMap()
{
   /*
    BOOST_FOREACH( boost::property_tree::ptree::value_type const& ak, m_tree.get_child("lib.substance") ) 
    {
        unsigned int idx = boost::lexical_cast<unsigned int>(ak.first.c_str());
        std::string imat = get("lib.substance.%u.imat.shortname", idx);
        std::string omat = get("lib.substance.%u.omat.shortname", idx);
        printf(" %2u %25s %25s \n", idx, imat.c_str(), omat.c_str() );          
    }
   */


    char key[128];
    BOOST_FOREACH( boost::property_tree::ptree::value_type const& ak, m_tree.get_child("lib.material") ) 
    {
        const char* matname = ak.first.c_str();
        snprintf(key, 128, "lib.material.%s.mat", matname);
        printf(" %s \n", key);

        std::string digest ;     
        BOOST_FOREACH( boost::property_tree::ptree::value_type const& bk, m_tree.get_child(key) ) // absolute key
        {
            unsigned int code = boost::lexical_cast<unsigned int>(bk.first.c_str());
            const char*  dig = bk.second.data().c_str();

            unsigned int isub   = code / 10 ;
            unsigned int offset = code % 10 ; 
            unsigned int line   = isub*4 + offset ;   // into the wavelengthBuffer assuming the 4x4 layout

            bool first = digest.empty();
            if(first)          digest = dig ;
            else               assert(strcmp(digest.c_str(), dig) == 0);

            if(first)
            {
                addMapEntry(line, matname);
            } 

            printf(" code %4u isub %3u offset %u line %u dig %s matname %s \n", code, isub, offset, line, dig, matname );
        }
    }


}

void GSubstanceLibMetadata::addMapEntry(unsigned int line, const char* shortname)
{
    char key[128];
    snprintf(key, 128, "lib.material_map.%u", line );
    m_tree.add(key, shortname); 
}



void GSubstanceLibMetadata::Summary(const char* msg)
{    
    printf("%s\n", msg );

    // TODO: recursive dumping for greater flexibility  

    BOOST_FOREACH( boost::property_tree::ptree::value_type const& ak, m_tree.get_child("") ) 
    {   
        std::cout << ak.first << std::endl ;  // lib

        BOOST_FOREACH( boost::property_tree::ptree::value_type const& bk, ak.second.get_child("") ) 
        {   
            std::cout << bk.first << std::endl ;   // substance

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



void GSubstanceLibMetadata::save(const char* dir)
{
    fs::path cachedir(dir);
    if(!fs::exists(cachedir))
    {
        if (fs::create_directory(cachedir))
        {
            printf("GSubstanceLibMetadata::save created directory %s \n", dir );
        }
    }

    if(fs::exists(cachedir) && fs::is_directory(cachedir))
    {
        fs::path path(dir);
        path /= filename ; 

        pt::write_json(path.string().c_str(), m_tree);
    }
    else
    {
        printf("GSubstanceLibMetadata::save directory %s DOES NOT EXIST \n", dir);
    }
}

void GSubstanceLibMetadata::read(const char* path)
{
    pt::read_json(path, m_tree);
}

GSubstanceLibMetadata* GSubstanceLibMetadata::load(const char* dir)
{
    GSubstanceLibMetadata* meta(NULL);
    fs::path cachedir(dir);
    if(!fs::exists(cachedir))
    {
        printf("GSubstanceLibMetadata::load directory %s DOES NOT EXIST \n", dir);
    }
    else
    {
        fs::path path(dir);
        path /= filename ; 
        meta = new GSubstanceLibMetadata ;
        meta->read(path.string().c_str());
    }
    return meta ; 
}

