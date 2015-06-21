#include "GColors.hh"
#include "GBuffer.hh"

#include "jsonutil.hpp"
#include "assert.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>


#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


using namespace std ; 

const char* GColors::NAME = "GColors.json" ;

GColors* GColors::load(const char* dir, const char* name)
{
    if(!existsMap(dir, name))
    {
        LOG(warning) << "GColors::load FAILED no file at  " << dir << "/" << name ; 
        return NULL ;
    }

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
    //dumpMap<std::string, std::string>( m_name2hex, msg );

    typedef std::map<std::string, std::string> MSS ; 
    typedef std::vector<std::string> VS ; 

    VS names ; 
    for(MSS::iterator it=m_name2hex.begin() ; it != m_name2hex.end() ; it++ ) names.push_back(it->first) ;

    std::sort(names.begin(), names.end(), *this );

    for(VS::iterator it=names.begin() ; it != names.end() ; it++ )
    {
        std::string name = *it ; 
        const char* hex_ = getHex(name.c_str());
        unsigned int code = getCode(name.c_str());
        std::cout 
            << " name   " << std::setw(25) <<  name
            << " hex_   " << std::setw(10) <<  hex_
            << " code  "  << std::setw(10) << std::dec <<  code
            << " code  "  << std::setw(10) << std::hex <<  code
            << std::endl ; 
    }
}

bool GColors::operator() (const std::string& a, const std::string& b)
{
    // sort order for dump 
    return getCode(a.c_str(),0) < getCode(b.c_str(),0) ; 
}

unsigned int GColors::getCode(const char* name, unsigned int missing)
{
    if(!name) return missing ;
    const char* hex_ = getHex(name, NULL);
    if(!hex_) return missing ;
    assert(hex_[0] == '#');
    unsigned int code = parseHex( hex_ + 1 ); 
    return code ; 
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
    GBuffer* buffer = make_uchar4_buffer();
    unsigned char* data = (unsigned char*)buffer->getPointer();

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

        unsigned int rgb   = getBufferEntry(data+count*4) ; 

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

GBuffer* GColors::make_uchar4_buffer()
{
    std::vector<unsigned int> codes ; 
    typedef std::map<std::string, std::string> MSS ; 
    unsigned int count(0);
    for(MSS::iterator it=m_name2hex.begin() ; it != m_name2hex.end() ; it++ ) 
    {
        unsigned int code = getCode(it->first.c_str());
        codes.push_back(code);
    }   
    return make_uchar4_buffer( codes) ; 
}


GBuffer* GColors::make_uchar4_buffer(std::vector<unsigned int>& codes)
{
    unsigned int n = codes.size();

    unsigned char* colors = new unsigned char[n*4] ; 
    GBuffer* buffer = new GBuffer( sizeof(unsigned char)*n*4, colors, 4*sizeof(unsigned char), 1 );


    unsigned char alpha = 0xFF ; 
    typedef std::vector<unsigned int> VU ; 
    unsigned int count(0);
    for(VU::iterator it=codes.begin() ; it != codes.end() ; it++ ) 
    {
        unsigned int color = *it ;
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
    return buffer ; 
}


void GColors::dump_uchar4_buffer( GBuffer* buffer )
{
    unsigned char* data = (unsigned char*)buffer->getPointer();
    unsigned int numCols = buffer->getNumItems();
    for(unsigned int i=0 ; i < numCols ; i++)
    {
         unsigned int rgb = getBufferEntry(data+4*i) ;
         std::cout 
                   << std::setw(5)  << std::dec << i 
                   << std::setw(10) << std::dec << rgb 
                   << std::setw(10) << std::hex << rgb 
                   << std::endl ; 
    }
}


