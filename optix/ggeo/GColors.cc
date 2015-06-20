#include "GColors.hh"
#include "jsonutil.hpp"
#include "assert.h"

#include <iostream>
#include <iomanip>
#include <sstream>

using namespace std ; 

const char* GColors::NAME = "colors.json" ;

GColors* GColors::load(const char* dir)
{
    GColors* gc = new GColors ; 
    gc->loadMaps(dir);
    return gc ; 
}

void GColors::loadMaps(const char* dir)
{
    loadMap<std::string, std::string>( m_name2hex, dir, NAME );
}

void GColors::dump(const char* msg)
{
    dumpMap<std::string, std::string>( m_name2hex, msg );
}


const char* GColors::getHex( const char* name, const char* missing)
{
    return m_name2hex.count(name) == 1 ? m_name2hex[name].c_str() : missing ;
}

const char* GColors::getName( const char* hex_, const char* missing)
{
    typedef std::map<std::string, std::string> MSS ; 
    for(MSS::iterator it=m_name2hex.begin() ; it != m_name2hex.end() ; it++ ) 
        if(it->second == hex_) return it->first.c_str();
    return missing ; 
}

void GColors::test(const char* msg)
{
    unsigned char* buffer = make_uchar4_buffer();

    unsigned int count(0); 
    typedef std::map<std::string, std::string> MSS ; 
    for(MSS::iterator it=m_name2hex.begin() ; it != m_name2hex.end() ; it++ ) 
    {
        string name = it->first ;
        string hex_ = it->second ;

        assert(hex_[0] == '#');
        //assert(strcmp(getName(hex_.c_str()), name.c_str())==0);  name to hex is non-unique
        assert(strcmp(getHex(name.c_str()), hex_.c_str())==0);

        unsigned int color = parseHex(hex_.c_str()+1);
        unsigned int red   = (color & 0xFF0000) >> 16;
        unsigned int green = (color & 0x00FF00) >> 8 ;
        unsigned int blue  = (color & 0x0000FF)      ;

        unsigned int rgb   = getBufferEntry(buffer+count*4) ; 

        cout 
             << setw(3)  << count 
             << setw(20) << name  
             << setw(20) << hex_ 
             << setw(20) << dec << color 
             << setw(20) << hex << color
             << setw(4)  << hex << red 
             << setw(4)  << hex << green
             << setw(4)  << hex << blue 
             << setw(20) << hex << rgb
             << endl ; 

        assert(rgb == color);
        count++ ; 
    } 

    delete buffer ; 

}

unsigned int GColors::getBufferEntry(unsigned char* colors)
{
    unsigned int red   = colors[0] ;
    unsigned int green = colors[1] ;
    unsigned int blue  = colors[2] ;
    unsigned int alpha = colors[3] ;

    unsigned int rgb = red << 16 | green << 8 | blue  ; 
    return rgb ; 
}




unsigned int GColors::parseHex(const char* hex_)
{
    unsigned int x ; 
    stringstream ss;
    ss << hex << hex_ ;
    ss >> x;
    return x ; 
}


unsigned int GColors::getNumColors()
{
    return m_name2hex.size();
}

unsigned int GColors::getNumBytes()
{
    return m_name2hex.size() * sizeof(unsigned char) * 4 ;
}

unsigned char* GColors::make_uchar4_buffer()
{
    unsigned int n = getNumColors();
    unsigned char* colors = new unsigned char[n*4] ; 
    unsigned char alpha = 0xFF ; 

    typedef std::map<std::string, std::string> MSS ; 

    unsigned int count(0);
    for(MSS::iterator it=m_name2hex.begin() ; it != m_name2hex.end() ; it++ ) 
    {
        string name = it->first ;
        string hex_ = it->second ;

        unsigned int color = parseHex(hex_.c_str()+1);
        unsigned int red   = (color & 0xFF0000) >> 16;
        unsigned int green = (color & 0x00FF00) >> 8 ;
        unsigned int blue  = (color & 0x0000FF)      ;

        unsigned int offset = count*4 ;  
        colors[offset + 0] = red ; 
        colors[offset + 1] = green ; 
        colors[offset + 2] = blue ;  
        colors[offset + 3] = alpha  ; 

        count++ ; 
    }   
    return colors ; 
}

