#include "NSensorList.hpp"
#include "NSensor.hpp"

#include "assert.h"
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

#include "stdio.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>

namespace fs = boost::filesystem;



void NSensorList::load(const char* idpath_, const char* ext)
{
    if(!idpath_) return ;

    fs::path idpath(idpath_);
    std::string name = idpath.filename().string() ;
    fs::path pdir = idpath.parent_path();

    std::vector<std::string> elem ;
    boost::split(elem, name, boost::is_any_of("."));

    assert(elem.size() == 3 );
    elem.erase(elem.begin() + 1); // remove hex digest

    std::string daename = boost::algorithm::join(elem, ".");

    fs::path daepath(pdir);
    daepath /= daename ;

    elem[1] = ext ; 
    std::string idmname = boost::algorithm::join(elem, ".");

    fs::path idmpath(pdir);
    idmpath /= idmname ; 

    LOG(debug) << "NSensorList::load "
              << "\n idpath:   " << idpath.string() 
              << "\n pdir:     " << pdir.string() 
              << "\n filename: " << name 
              << "\n daepath:  " << daepath.string() 
              << "\n idmpath:  " << idmpath.string() 
              ;   

    read(idmpath.string().c_str());
}


void NSensorList::read(const char* path)
{
    std::ifstream in(path, std::ios::in);
    if(!in.is_open()) 
    {   
        LOG(fatal) << "NSensorList::read failed to open " << path ; 
        return ;
    }   

    typedef boost::tokenizer< boost::char_separator<char> > Tok_t;
    boost::char_separator<char> delim(" ");

    std::string line ; 
    std::vector<std::string> elem ; 

    unsigned int count(0);
    while(std::getline(in, line))
    {   
        if(line[0] == '#') continue ; 

        Tok_t tok(line, delim) ;
        elem.assign(tok.begin(), tok.end());
        NSensor* sensor = createSensor(elem);
        if(sensor) add(sensor);

        //if(count < 10) printf("[%lu] %s \n", elem.size(), line.c_str());
        count++;
    }
    in.close();
    LOG(debug) << "NSensorList::read " 
              << " path " << path 
              << " desc " << description() 
              ; 
}


std::string NSensorList::description()
{
    std::stringstream ss ; 
    ss << "NSensorList: " 
       << " NSensor count " << m_sensors.size()
       << " distinct identier count " << m_ids.size() 
      ;
    return ss.str();
}


NSensor* NSensorList::createSensor(std::vector<std::string>& elem )
{
    assert(elem.size() == 6 );

    unsigned int nodeIndex  = boost::lexical_cast<unsigned int>(elem[0]); 
    unsigned int id         = boost::lexical_cast<unsigned int>(elem[1]); 
    unsigned int id_hex     = parseHexString(elem[2]);
    assert(id == id_hex );

    NSensor* sensor(NULL);
    if( id > 0 )
    {
        unsigned int index = m_sensors.size() ;    // 0-based in tree node order
        sensor = new NSensor(index, id, elem[5].c_str(), nodeIndex);
    }
    return sensor ; 
}


void NSensorList::add(NSensor* sensor)
{
    assert(sensor);

    m_sensors.push_back(sensor);  

    m_ids.insert(sensor->getId());

    unsigned int nid = sensor->getNodeIndex(); 

    assert(m_nid2sen.count(nid) == 0 && "there should only ever be one NSensor for each node index");

    m_nid2sen[nid] = sensor ; 
}


unsigned int NSensorList::parseHexString(std::string& str)
{
    unsigned int x;   
    std::stringstream ss;
    ss << std::hex << str ;
    ss >> x;
    return x ; 
}


void NSensorList::dump(const char* msg)
{
    unsigned int nsen = getNumSensors();
    printf("%s : %u sensors \n", msg, nsen);
    for(unsigned int i=0 ; i < nsen ; i++)
    {
        NSensor* sensor = getSensor(i);
        assert(sensor);
        printf("%s\n", sensor->description().c_str()) ;

        NSensor* check = findSensorForNode(sensor->getNodeIndex());
        assert(check == sensor);
    }
}

NSensor* NSensorList::findSensorForNode(unsigned int nodeIndex)
{
    return m_nid2sen.count(nodeIndex) == 1 ? m_nid2sen[nodeIndex] : NULL ; 
}



