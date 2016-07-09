#include <cassert>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "BFile.hh"
#include "BMap.hh"

#include "NPY.hpp"
#include "NSpectral.hpp"

#include "OpticksColors.hh"
#include "PLOG.hh"

using namespace std ; 

//const char* OpticksColors::NAME = "GColors.json" ;
const char* OpticksColors::NAME = "OpticksColors.json" ;



OpticksColors::OpticksColors()  
    :
    m_composite(NULL)
{
   m_composite_domain.x = 0 ; 
   m_composite_domain.y = 0 ; 
   m_composite_domain.z = 0 ; 
   m_composite_domain.w = 0 ; 
}


nuvec4 OpticksColors::getCompositeDomain()
{
    return m_composite_domain ; 
}

NPY<unsigned char>* OpticksColors::getCompositeBuffer()
{
    return m_composite ;  
}



OpticksColors* OpticksColors::load(const char* dir, const char* name)
{
    OpticksColors* gc = new OpticksColors ; 
    if(!BFile::ExistsFile(dir, name))
    {
        LOG(warning) << "OpticksColors::load FAILED no file at  dir " << dir << " with name " << name ; 
    } 
    else
    {
        gc->loadMaps(dir);
    }
    return gc ; 
}

void OpticksColors::loadMaps(const char* dir)
{
    BMap<std::string, std::string>::load( &m_name2hex, dir, NAME );
}

void OpticksColors::sort()
{
    LOG(debug) << "OpticksColors::sort" ; 
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

bool OpticksColors::operator() (const std::string& a, const std::string& b)
{
    // sort order for dump 
    return getCode(a.c_str(),0) < getCode(b.c_str(),0) ; 
}

void OpticksColors::dump(const char* msg)
{
    unsigned int num_colors = getNumColors();
    LOG(info) << msg 
              << " num_colors " << num_colors ; 

    for(unsigned int i=0 ; i < num_colors ; i++)
    {
        std::string name = getNamePsychedelic(i) ; 
        const char* hex_ = getHex(name.c_str());
        unsigned int code = getCode(name.c_str());

        nvec3 rgb = getColor(name.c_str());

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


unsigned int OpticksColors::getCode(const char* name, unsigned int missing)
{
    if(!name) return missing ;
    const char* hex_ = getHex(name, NULL);
    if(!hex_) return missing ;
    assert(hex_[0] == '#');
    unsigned int code = parseHex( hex_ + 1 ); 
    return code ; 
}

const char* OpticksColors::getHex( const char* name, const char* missing)
{
    return m_name2hex.count(name) == 1 ? m_name2hex[name].c_str() : missing ;
}

nvec3 OpticksColors::getColor(const char* name, unsigned int missing)
{
    unsigned int code = getCode(name, missing);
    return makeColor(code) ;
}


const char* OpticksColors::getNamePsychedelic(unsigned int index)
{
    if(m_psychedelic_names.size() == 0 && m_name2hex.size() > 0) sort();
    return 
       index < m_psychedelic_names.size() 
             ?
             m_psychedelic_names[index].c_str() 
             :
             NULL  
             ;
}


nvec3 OpticksColors::getPsychedelic(unsigned int num)
{
    unsigned int num_colors = getNumColors();
    unsigned int index = num_colors > 0 ? num % num_colors : 0 ;
    const char* cname = getNamePsychedelic(index);    

    LOG(trace) << "OpticksColors::getPsychedelic"
              << " num " << num 
              << " index " << index
              << " num_colors " << num_colors 
              << " cname " << ( cname ? cname : "NULL" )
              ;

    return getColor( cname );
}

std::vector<unsigned int>& OpticksColors::getPsychedelicCodes()
{
    if(m_psychedelic_codes.size() == 0) sort();
    return m_psychedelic_codes ;
}

std::vector<unsigned int>& OpticksColors::getSpectralCodes()
{
    //if(m_spectral_codes.size() == 0) make_spectral_codes();
    return m_spectral_codes ;
}

const char* OpticksColors::getName( const char* hex_, const char* missing)
{
    typedef std::map<std::string, std::string> MSS ; 
    for(MSS::iterator it=m_name2hex.begin() ; it != m_name2hex.end() ; it++ ) 
        if(it->second == hex_) return it->first.c_str();
    return missing ; 
}

void OpticksColors::test(const char* msg)
{
    LOG(info) << msg ;  

    NPY<unsigned char>* buffer = make_buffer();
    buffer->save("$TMP/OpticksColors.npy");

    unsigned char* data = buffer->getValues();

    unsigned int nfail(0); 
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
             << " [ color  "
             << setw(20) << dec << color 
             << setw(20) << hex << color
             << " ] "
             << " [ rgb "
             << setw(20) << dec << rgb 
             << setw(20) << hex << rgb << dec
             << " ] "
             << setw(4)  << hex << red 
             << setw(4)  << hex << green
             << setw(4)  << hex << blue 
             << " " << ( color != rgb  ? "MISMATCH" : "OK" )
             << endl ; 

        if(color != rgb)
        {
            nfail +=  1; 
        }
        count++ ; 
    } 
    assert(nfail == 0);

    delete buffer ; 

}

unsigned int OpticksColors::getBufferEntry(unsigned char* colors)
{
    unsigned int red   = colors[0] ;
    unsigned int green = colors[1] ;
    unsigned int blue  = colors[2] ;
    //unsigned int alpha = colors[3] ;

    unsigned int rgb = red << 16 | green << 8 | blue  ; 
    return rgb ; 
}

unsigned int OpticksColors::parseHex(const char* hex_)
{
    unsigned int x ; 
    stringstream ss;
    ss << hex << hex_ ;
    ss >> x;
    return x ; 
}


unsigned int OpticksColors::getNumColors()
{
    return m_name2hex.size();
}

unsigned int OpticksColors::getNumBytes()
{
    return m_name2hex.size() * sizeof(unsigned char) * 4 ;
}



NPY<unsigned char>* OpticksColors::make_buffer()
{
    std::vector<unsigned int> codes ; 
    typedef std::map<std::string, std::string> MSS ; 
    for(MSS::iterator it=m_name2hex.begin() ; it != m_name2hex.end() ; it++ ) 
    {
        unsigned int code = getCode(it->first.c_str());
        codes.push_back(code);
    }   
    return make_buffer( codes) ; 
}

NPY<unsigned char>* OpticksColors::make_buffer(std::vector<unsigned int>& codes)
{
    unsigned int n = codes.size();
    unsigned char c_alpha = 0xFF ; 
    NPY<unsigned char>* buf = NPY<unsigned char>::make(n, 4 ); 
    buf->zero();    
    unsigned char* data = buf->getValues();

    LOG(trace) << "OpticksColors::make_buffer" 
               << " n " << n 
               ; 

    for(unsigned int i=0 ; i < n ; i++)
    {
        unsigned int color = codes[i] ;
        unsigned int red   = (color & 0xFF0000) >> 16;
        unsigned int green = (color & 0x00FF00) >> 8 ;
        unsigned int blue  = (color & 0x0000FF)      ;

        LOG(trace) << std::setw(4) << i 
                  << std::setw(10) << hex << color << dec
                  << std::setw(10) << hex << red << dec
                  << std::setw(10) << hex << green << dec
                  << std::setw(10) << hex << blue << dec
                  ;

        unsigned char c_red(red);
        unsigned char c_green(green);
        unsigned char c_blue(blue);

        *(data + i*4 + 0) = c_red ; 
        *(data + i*4 + 1) = c_green ; 
        *(data + i*4 + 2) = c_blue ; 
        *(data + i*4 + 3) = c_alpha ; 

    }  
    return buf ; 
}




void OpticksColors::initCompositeColorBuffer(unsigned int max_colors)
{

    LOG(debug) << "OpticksColors::initCompositeColorBuffer "
              << " max_colors " << max_colors 
              ; 

    unsigned int n = max_colors*4 ;
    unsigned char* colors = new unsigned char[n] ; 
    while(n--) colors[n] = 0x44 ;  //  default to dull grey  

    std::vector<int> shape ; 
    shape.push_back(max_colors);
    shape.push_back(4);
    std::string metadata = "{}" ;

    m_composite = new NPY<unsigned char>(shape, colors, metadata  );
}

void OpticksColors::addColors(std::vector<unsigned int>& codes, unsigned int start )
{
    unsigned int max_colors = m_composite->getNumItems();
    unsigned char* colors = m_composite->getValues() ;

    unsigned char alpha = 0xFF ; 
    typedef std::vector<unsigned int> VU ; 

    LOG(info) << "OpticksColors::addColors " 
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
        if(!( offset < 4*max_colors))
             LOG(fatal) << "OpticksColors::addColors out of range " << offset ;

        assert( offset < 4*max_colors && " going over size of buffer" );

        colors[offset + 0] = red ; 
        colors[offset + 1] = green ; 
        colors[offset + 2] = blue ;  
        colors[offset + 3] = alpha  ; 

        count++ ; 
    } 
}

void OpticksColors::dumpCompositeBuffer(const char* msg)
{
    LOG(info) << msg ; 

    m_composite->dump(msg);
}


nvec3 OpticksColors::makeColor( unsigned int rgb )
{
    unsigned int red   =  ( rgb & 0xFF0000 ) >> 16 ;  
    unsigned int green =  ( rgb & 0x00FF00 ) >>  8 ;  
    unsigned int blue  =  ( rgb & 0x0000FF ) ;  

    float d(0xFF);
    float r = float(red)/d ;
    float g = float(green)/d ;
    float b = float(blue)/d ;

    return make_nvec3(r,g,b);
}


void OpticksColors::setupCompositeColorBuffer(std::vector<unsigned int>&  material_codes, std::vector<unsigned int>& flag_codes)
{
    std::vector<unsigned int>& psychedelic_codes = getPsychedelicCodes();
    std::vector<unsigned int>& spectral_codes = getSpectralCodes();

    unsigned int colormax = COLORMAX ; 
    initCompositeColorBuffer(colormax);
    assert( m_composite->getNumItems() == colormax );

    unsigned int material_color_offset = MATERIAL_COLOR_OFFSET ; 
    unsigned int flag_color_offset     = FLAG_COLOR_OFFSET ; 
    unsigned int psychedelic_color_offset = PSYCHEDELIC_COLOR_OFFSET ; 
    unsigned int spectral_color_offset = SPECTRAL_COLOR_OFFSET ; 


    LOG(info) << "OpticksColors::setupCompositeColorBuffer"
              << " material_codes " << material_codes.size()
              << " flag_codes " << flag_codes.size()
              << " psychedelic_codes " << psychedelic_codes.size()
              << " spectral_codes " << spectral_codes.size()
              << " material_color_offset " << material_color_offset
              << " flag_color_offset " << flag_color_offset
              << " psychedelic_color_offset " << psychedelic_color_offset
              << " spectral_color_offset " << spectral_color_offset
              ;
              

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
    LOG(info) << "OpticksColors::setupCompositeColorBuffer DONE " ;
}

