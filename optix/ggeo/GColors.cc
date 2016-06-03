#include "GColors.hh"
#include "GBuffer.hh"

#include "jsonutil.hpp"
#include "assert.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "NSpectral.hpp"
#include "NLog.hpp"
#include "NPY.hpp"

using namespace std ; 

const char* GColors::NAME = "GColors.json" ;

GColors* GColors::load(const char* dir, const char* name)
{
    if(!existsPath(dir, name))
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


void GColors::sort()
{
    LOG(debug) << "GColors::sort" ; 

    typedef std::map<std::string, std::string> MSS ; 
    for(MSS::iterator it=m_name2hex.begin() ; it != m_name2hex.end() ; it++ ) m_psychedelic_names.push_back(it->first) ;
    std::sort(m_psychedelic_names.begin(), m_psychedelic_names.end(), *this );

    for(unsigned int i=0 ; i < m_psychedelic_names.size() ; i++)
    {
        string name = m_psychedelic_names[i] ;
        unsigned int code = getCode(name.c_str()); 
        m_psychedelic_codes.push_back(code); 
    } 
}

//void GColors::make_spectral_codes()
//{
//    m_spectral_codes = NSpectral::make_colors();
//}


void GColors::dump(const char* msg)
{
    typedef std::map<std::string, std::string> MSS ; 

    unsigned int num_colors = getNumColors();

    LOG(info) << msg 
              << " num_colors " << num_colors ; 

    for(unsigned int i=0 ; i < num_colors ; i++)
    {
        std::string name = getNamePsychedelic(i) ; 
        const char* hex_ = getHex(name.c_str());
        unsigned int code = getCode(name.c_str());

        gfloat3 rgb = getColor(name.c_str());

        std::cout 
            << " " << std::setw(3) <<  i
            << " name   " << std::setw(25) <<  name
            << " hex_   " << std::setw(10) <<  hex_
            << " code  "  << std::setw(10) << dec <<  code
            << " code  "  << std::setw(10) << hex <<  code << dec
            << " r " << std::setw(10) << rgb.x 
            << " g " << std::setw(10) << rgb.y 
            << " b " << std::setw(10) << rgb.z 
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


gfloat3 GColors::getColor(const char* name, unsigned int missing)
{
    unsigned int code = getCode(name, missing);
    return makeColor(code) ;
}


const char* GColors::getNamePsychedelic(unsigned int index)
{
    if(m_psychedelic_names.size() == 0) sort();
    return m_psychedelic_names[index].c_str() ;
}


gfloat3 GColors::getPsychedelic(unsigned int num)
{
    unsigned int index = num % getNumColors() ;
    const char* cname = getNamePsychedelic(index);    
    return getColor( cname );
}

std::vector<unsigned int>& GColors::getPsychedelicCodes()
{
    if(m_psychedelic_codes.size() == 0) sort();
    return m_psychedelic_codes ;
}

std::vector<unsigned int>& GColors::getSpectralCodes()
{
    //if(m_spectral_codes.size() == 0) make_spectral_codes();
    return m_spectral_codes ;
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
             << setw(3)  << dec << count 
             << setw(20) << name  
             << setw(20) << hex_ 
             << setw(20) << dec << color 
             << setw(20) << hex << color
             << setw(4)  << hex << red 
             << setw(4)  << hex << green
             << setw(4)  << hex << blue 
             << setw(20) << hex << rgb << dec
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
    //unsigned int alpha = colors[3] ;

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
    //unsigned int count(0);
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




NPY<unsigned char>* GColors::make_buffer()
{
    std::vector<unsigned int> codes ; 
    typedef std::map<std::string, std::string> MSS ; 
    //unsigned int count(0);
    for(MSS::iterator it=m_name2hex.begin() ; it != m_name2hex.end() ; it++ ) 
    {
        unsigned int code = getCode(it->first.c_str());
        codes.push_back(code);
    }   
    return make_buffer( codes) ; 
}

NPY<unsigned char>* GColors::make_buffer(std::vector<unsigned int>& codes)
{
    unsigned int n = codes.size();
    unsigned char alpha = 0xFF ; 
    NPY<unsigned char>* buf = NPY<unsigned char>::make(n, 4 ); 

    for(unsigned int i=0 ; i < n ; i++)
    {
        unsigned int color = codes[i] ;
        unsigned int red   = (color & 0xFF0000) >> 16;
        unsigned int green = (color & 0x00FF00) >> 8 ;
        unsigned int blue  = (color & 0x0000FF)      ;

        buf->setValue( i, 0, 0, 0, red); 
        buf->setValue( i, 1, 0, 0, green); 
        buf->setValue( i, 2, 0, 0, blue); 
        buf->setValue( i, 3, 0, 0, alpha); 
    }  
    return buf ; 
}




void GColors::initCompositeColorBuffer(unsigned int max_colors)
{
    unsigned int itemsize = sizeof(unsigned char)*4 ;

    LOG(debug) << "GColors::initCompositeColorBuffer "
              << " max_colors " << max_colors 
              << " itemsize " << itemsize 
              ; 

    unsigned int n = max_colors*4 ;
    unsigned char* colors = new unsigned char[n] ; 
    while(n--) colors[n] = 0x44 ;  //  default to dull grey  


    m_composite = new GBuffer( itemsize*max_colors, colors, itemsize, 1 );



    std::vector<int> shape ; 
    shape.push_back(max_colors);
    shape.push_back(4);
    std::string metadata = "{}" ;

    m_composite_ = new NPY<unsigned char>(shape, colors, metadata  );
}

void GColors::addColors(std::vector<unsigned int>& codes, unsigned int start )
{
    unsigned int max_colors = m_composite->getNumItems();
    unsigned char* colors = (unsigned char*)m_composite->getPointer() ;

    unsigned char* colors_ = m_composite_->getValues() ;

    unsigned char alpha = 0xFF ; 
    typedef std::vector<unsigned int> VU ; 

    LOG(debug) << "GColors::addColors " 
              << " codes.size " << codes.size()
              << " start " << start 
              << " max_colors " << max_colors 
              ;

    unsigned int count = start ;  // color counting 
    for(VU::iterator it=codes.begin() ; it != codes.end() ; it++ ) 
    {
        unsigned int color = *it ;
        unsigned int red   = (color & 0xFF0000) >> 16;
        unsigned int green = (color & 0x00FF00) >> 8 ;
        unsigned int blue  = (color & 0x0000FF)      ;

        unsigned int offset = count*4 ;  

        assert( offset < 4*max_colors && " going over size of buffer" );

        colors[offset + 0] = red ; 
        colors[offset + 1] = green ; 
        colors[offset + 2] = blue ;  
        colors[offset + 3] = alpha  ; 

        colors_[offset + 0] = red ; 
        colors_[offset + 1] = green ; 
        colors_[offset + 2] = blue ;  
        colors_[offset + 3] = alpha  ; 

        count++ ; 
    } 
}

void GColors::dumpCompositeBuffer(const char* msg)
{
    LOG(info) << msg ; 
    dump_uchar4_buffer(m_composite);

    m_composite_->dump(msg);
}

void GColors::dump_uchar4_buffer( GBuffer* buffer )
{
    LOG(info)<<"GColors::dump_uchar4_buffer";
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



gfloat3 GColors::makeColor( unsigned int rgb )
{
    unsigned int red   =  ( rgb & 0xFF0000 ) >> 16 ;  
    unsigned int green =  ( rgb & 0x00FF00 ) >>  8 ;  
    unsigned int blue  =  ( rgb & 0x0000FF ) ;  

    float d(0xFF);
    float r = float(red)/d ;
    float g = float(green)/d ;
    float b = float(blue)/d ;

    return gfloat3( r, g, b) ;
}


guint4 GColors::getCompositeDomain()
{
    return m_composite_domain ; 
}

void GColors::setupCompositeColorBuffer(std::vector<unsigned int>&  material_codes, std::vector<unsigned int>& flag_codes)
{
    std::vector<unsigned int>& psychedelic_codes = getPsychedelicCodes();
    std::vector<unsigned int>& spectral_codes = getSpectralCodes();

    unsigned int colormax = COLORMAX ; 
    initCompositeColorBuffer(colormax);
    assert( m_composite->getNumItems() == colormax );
    assert( m_composite_->getNumItems() == colormax );

    unsigned int material_color_offset = MATERIAL_COLOR_OFFSET ; 
    unsigned int flag_color_offset     = FLAG_COLOR_OFFSET ; 
    unsigned int psychedelic_color_offset = PSYCHEDELIC_COLOR_OFFSET ; 
    unsigned int spectral_color_offset = SPECTRAL_COLOR_OFFSET ; 

    if(material_codes.size() > 0)
    {
        assert(material_codes.size() < 64 );
        addColors(material_codes,     material_color_offset ) ;
    }
    if(flag_codes.size() > 0)
    {
        assert(flag_color_offset + flag_codes.size() < psychedelic_color_offset );
        addColors(flag_codes, flag_color_offset ) ;  
    }
    if(psychedelic_codes.size() > 0)
    {
        assert(psychedelic_color_offset + psychedelic_codes.size() < spectral_color_offset );
        addColors(psychedelic_codes , psychedelic_color_offset ) ;  
    }
    if(spectral_codes.size() > 0)
    {
        assert(spectral_color_offset + spectral_codes.size() < colormax );
        addColors(spectral_codes , spectral_color_offset ) ;  
    }



    m_composite_domain.x = 0 ; 
    m_composite_domain.y = colormax ;
    m_composite_domain.z = m_psychedelic_codes.size() ; 
    m_composite_domain.w = psychedelic_color_offset ; 
  
    // used in 
    //        optixrap-/cu/color_lookup.h 
    //        oglrap-/gl/fcolor.h
    //
    //     fcolor = texture(Colors, (float(flq[0].x) + MATERIAL_COLOR_OFFSET - 1.0 + 0.5)/ColorDomain.y ) 
    //
}

