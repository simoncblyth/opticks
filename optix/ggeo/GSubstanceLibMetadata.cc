#include "GSubstanceLibMetadata.hh"
#include "GSubstanceLib.hh"
#include "GSubstance.hh"
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



namespace fs = boost::filesystem;
namespace pt = boost::property_tree;

const char* GSubstanceLibMetadata::filename = "GSubstanceLibMetadata.json" ; 
const char* GSubstanceLibMetadata::mapname  = "GSubstanceLibMetadataMaterialMap.json" ; 

GSubstanceLibMetadata::GSubstanceLibMetadata() 
{
}

void GSubstanceLibMetadata::add(const char* kfmt, unsigned int isub, const char* cat, const char* tag, const char* val)
{
    char key[128];
    snprintf(key, 128, kfmt, isub, cat, tag );
    m_tree.add(key, val);
}


void GSubstanceLibMetadata::addDigest(const char* kfmt, unsigned int isub, const char* cat, const char* dig )
{
    add(kfmt, isub, cat, "digest",  dig );
}



void GSubstanceLibMetadata::add(const char* kfmt, unsigned int isub, const char* cat, GPropertyMap<float>* pmap )
{
    if(!pmap) return ;

    const char* name = pmap->getName() ;
    const char* type = pmap->getType() ;
    std::string keys = pmap->getKeysString() ;

    // "lib.substance.%d.%s.%s" ;
    add(kfmt, isub, cat, "name",    name); 
    add(kfmt, isub, cat, "type",    type);
    add(kfmt, isub, cat, "keys",    keys.c_str());

    if(strcmp(cat, "imat") == 0 || strcmp(cat, "omat") == 0)
    {
        char* shortname = pmap->getShortName("__dd__Materials__") ; 
        add(kfmt, isub, cat, "shortname", shortname); 
        free(shortname);
    }
}
void GSubstanceLibMetadata::addMaterial(unsigned int isub, const char* cat, const char* shortname, const char* digest )
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


std::string GSubstanceLibMetadata::getSubstanceQtyByIndex(unsigned int isub, unsigned int icat, const char* tag)
{
    unsigned int numQuad = GSubstanceLib::getNumQuad();
    assert(icat < numQuad); 
    const char* cat = GSubstance::getConstituentNameByIndex(icat);
    return getSubstanceQty(isub, cat, tag);
}

std::string GSubstanceLibMetadata::getSubstanceQty(unsigned int isub, const char* cat, const char* tag)
{
    char key[128];
    snprintf(key, 128, "lib.substance.%u.%s.%s", isub, cat, tag);
    return m_tree.get<std::string>(key);
}


unsigned int GSubstanceLibMetadata::getNumSubstance()
{
    unsigned int count(0);
    BOOST_FOREACH( boost::property_tree::ptree::value_type const& ak, m_tree.get_child("lib.substance") ) 
    {
        unsigned int index = boost::lexical_cast<unsigned int>(ak.first.c_str());
        assert(index == count);
        count++;
    }
    return count ;
}


std::string GSubstanceLibMetadata::getSubstanceName(unsigned int isub)
{
    std::string imat = getSubstanceQty(isub, "imat", "shortname") ;
    std::string omat = getSubstanceQty(isub, "omat", "shortname") ;
    std::string isur = getSubstanceQty(isub, "isur", "name") ;
    std::string osur = getSubstanceQty(isub, "osur", "name") ;

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


std::map<int, std::string> GSubstanceLibMetadata::getBoundaryNames()
{
    std::map<int, std::string> nmap ; 
    unsigned int nsub = getNumSubstance();
    nmap[-1] = "unknown" ;
    for(unsigned int isub=0 ; isub < nsub ; isub++)
    {
        std::string name = getSubstanceName(isub);
        nmap[isub] = name ;        
        //printf("%2d : %s \n", isub, name.c_str());
    }
    return nmap ; 
}




void GSubstanceLibMetadata::dumpNames()
{
    unsigned int nsub = getNumSubstance();
    for(unsigned int isub=0 ; isub < nsub ; isub++)
    {
        std::string name = getSubstanceName(isub);
        printf("%2d : %s \n", isub, name.c_str());
    }
}




void GSubstanceLibMetadata::createMaterialMap()
{
    typedef std::map<std::string, unsigned int> Map_t ;
    Map_t name2line ; 

    char key[128];
    BOOST_FOREACH( boost::property_tree::ptree::value_type const& ak, m_tree.get_child("lib.material") ) 
    {
        std::string matname = ak.first ;
        snprintf(key, 128, "lib.material.%s.mat", matname.c_str());
        //printf("GSubstanceLibMetadata::createMaterialMap %s \n", key);

        std::string digest ;     
        BOOST_FOREACH( boost::property_tree::ptree::value_type const& bk, m_tree.get_child(key) ) // absolute key
        {
            unsigned int code = boost::lexical_cast<unsigned int>(bk.first.c_str());
            const char*  dig = bk.second.data().c_str();

            unsigned int isub   = code / 10 ;
            unsigned int offset = code % 10 ; 
            unsigned int line   = GSubstanceLib::getLine(isub, offset) ;   // into the wavelengthBuffer 

            bool first = digest.empty();
            if(first)          digest = dig ;
            else               assert(strcmp(digest.c_str(), dig) == 0);

            // only record line for 1st occurence of the name
            if(name2line.find(matname) == name2line.end()) name2line[matname] = line ; 

            //printf("   code %4u isub %3u offset %u line %u dig %s matname %s \n", code, isub, offset, line, dig, matname.c_str() );
        }
    }

    printf("GSubstanceLibMetadata::createMaterialMap\n");
    for(Map_t::iterator it=name2line.begin() ; name2line.end() != it ; it++)
    {
        printf(" %25s : %u \n", it->first.c_str(), it->second ); 
        m_material_map.add( it->first, it->second );
    }

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
        fs::path treepath(dir);
        treepath /= filename ; 
        pt::write_json(treepath.string().c_str(), m_tree);

        fs::path mappath(dir);
        mappath /= mapname ; 
        pt::write_json(mappath.string().c_str(), m_material_map);

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
        meta = new GSubstanceLibMetadata  ;
        meta->read(path.string().c_str());
    }
    return meta ; 
}

