#include "Sensor.hpp"

#include "assert.h"
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


#include <vector>
#include <string>
#include <iostream>     
#include <fstream>     

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>

namespace fs = boost::filesystem;


void Sensor::load(const char* idpath_, const char* ext)
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

    LOG(info) << "Sensor::load "
              << "\n idpath:   " << idpath.string() 
              << "\n pdir:     " << pdir.string() 
              << "\n filename: " << name 
              << "\n daepath:  " << daepath.string() 
              << "\n idmpath:  " << idmpath.string() 
              ; 

    read(idmpath.string().c_str());
}



void Sensor::read(const char* path)
{
    std::ifstream in(path, std::ios::in);
    if(!in.is_open()) 
    {   
        LOG(fatal) << "Sensor::read failed to open " << path ; 
        return ;
    }   

    typedef boost::tokenizer< boost::char_separator<char> > Tok_t;
    boost::char_separator<char> delim(" ");

    std::vector<std::string> elem ; 
    std::string comment ;
    std::string line ; 

    unsigned int count(0);
    while(std::getline(in, line))
    {
        if(line[0] == '#')
        {
            comment = line ; 
            continue ; 
        }
        Tok_t tok(line, delim) ;
        elem.assign(tok.begin(), tok.end());

        if(count < 10) printf("[%lu] %s \n", elem.size(), line.c_str());

        count++;
    }  
    in.close();

    printf("comment %s\n", comment.c_str());
}


