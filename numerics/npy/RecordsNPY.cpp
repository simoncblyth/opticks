#include "RecordsNPY.hpp"

#include <glm/glm.hpp>
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"
#include "regexsearch.hh"

#include <sstream>
#include <iomanip>


void RecordsNPY::setDomains(NPY<float>* domains)
{
    domains->dump("RecordsNPY::setDomains");

    glm::vec4 ce = domains->getQuad(0,0);
    glm::vec4 td = domains->getQuad(1,0);
    glm::vec4 wd = domains->getQuad(2,0);

    print(ce, "RecordsNPY::setDomains ce");
    print(td, "RecordsNPY::setDomains td");
    print(wd, "RecordsNPY::setDomains wd");

    setCenterExtent(ce);    
    setTimeDomain(td);    
    setWavelengthDomain(wd);    
}

float RecordsNPY::unshortnorm(short value, float center, float extent )
{
/*
cu/photon.h::
 
     83 __device__ short shortnorm( float v, float center, float extent )
     84 {
     85     // range of short is -32768 to 32767
     86     // Expect no positions out of range, as constrained by the geometry are bouncing on,
     87     // but getting times beyond the range eg 0.:100 ns is expected
     88     //
     89     int inorm = __float2int_rn(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)
     90     return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;
     91 }

*/
    return float(value)*extent/32767.0f + center ; 
}

float RecordsNPY::unshortnorm_position(short v, unsigned int k )
{
    assert(k < 3 );
    return unshortnorm( v, m_center_extent[k], m_center_extent.w );
}

float RecordsNPY::unshortnorm_time(short v, unsigned int k )
{
    assert(k == 3 );
    return unshortnorm( v, m_time_domain.x,    m_time_domain.y );
}





void RecordsNPY::unpack_position_time(glm::vec4& post, unsigned int i, unsigned int j)
{
    glm::uvec4 v = m_records->getQuadU( i, j );
    post.x = unshortnorm_position(v.x, 0);
    post.y = unshortnorm_position(v.y, 1);
    post.z = unshortnorm_position(v.z, 2);
    post.w = unshortnorm_time(v.w, 3);
}

void RecordsNPY::unpack_polarization_wavelength(glm::vec4& polw, unsigned int i, unsigned int j, unsigned int k0, unsigned int k1)
{
    ucharfour v = m_records->getUChar4( i, j, k0, k1 ); 
    polw.x =  uncharnorm_polarization(v.x);  
    polw.y =  uncharnorm_polarization(v.y);  
    polw.z =  uncharnorm_polarization(v.z);  
    polw.w =  uncharnorm_wavelength(v.w);  
}

void RecordsNPY::unpack_material_flags(glm::uvec4& flag, unsigned int i, unsigned int j, unsigned int k0, unsigned int k1)
{
    ucharfour v = m_records->getUChar4( i, j, k0, k1 ); 
    flag.x =  v.x ;  
    flag.y =  v.y ;  
    flag.z =  v.z ;  
    flag.w =  v.w ;  
}



float RecordsNPY::uncharnorm(unsigned char value, float center, float extent, float bitmax )
{
/*
cu/photon.h::

    122     float nwavelength = 255.f*(p.wavelength - wavelength_domain.x)/wavelength_domain.w ; // 255.f*0.f->1.f 
    123
    124     qquad qpolw ;
    125     qpolw.uchar_.x = __float2uint_rn((p.polarization.x+1.f)*127.f) ;
    126     qpolw.uchar_.y = __float2uint_rn((p.polarization.y+1.f)*127.f) ;
    127     qpolw.uchar_.z = __float2uint_rn((p.polarization.z+1.f)*127.f) ;
    128     qpolw.uchar_.w = __float2uint_rn(nwavelength)  ;
    129 
    130     // tightly packed, polarization and wavelength into 4*int8 = 32 bits (1st 2 npy columns) 
    131     hquad polw ;     // lsb_              msb_
    132     polw.ushort_.x = qpolw.uchar_.x | qpolw.uchar_.y << 8 ;
    133     polw.ushort_.y = qpolw.uchar_.z | qpolw.uchar_.w << 8 ;

                             center   extent
     pol      range -1 : 1     0        2
     pol + 1  range  0 : 2     1        2
 
*/
   
    return float(value)*extent/bitmax - center ; 
}


float RecordsNPY::uncharnorm_polarization(unsigned char value)
{
    return uncharnorm(value, 1.f, 2.f, 254.f );
}
float RecordsNPY::uncharnorm_wavelength(unsigned char value)
{
    return uncharnorm(value, -m_wavelength_domain.x , m_wavelength_domain.w, 255.f );
}


void RecordsNPY::dumpRecord(unsigned int i, const char* msg)
{
    bool unset = m_records->isUnsetItem(i);
    if(unset) return ;

    glm::vec4  post ; 
    glm::vec4  polw ; 
    glm::uvec4 flag ; 


    unpack_position_time(           post, i, 0 );       // i,j 
    unpack_polarization_wavelength( polw, i, 1, 0, 1 ); // i,j,k0,k1
    unpack_material_flags(          flag, i, 1, 2, 3);  // i,j,k0,k1

    std::string m1 = m_types->findMaterialName(flag.x) ;
    std::string m2 = m_types->findMaterialName(flag.y) ;

    // flag.w is the result of ffs on a single set bit field, returning a 1-based bit position
    std::string history = m_types->getHistoryString( 1 << (flag.w-1) ); 

    assert(flag.z == 0);

    printf("%s %8u %s %s %25s %25s %s \n", 
                msg,
                i, 
                gpresent(post,2,11).c_str(),
                gpresent(polw,2,7).c_str(),
                m1.c_str(),
                m2.c_str(),
                history.c_str());

}


void RecordsNPY::dumpRecords(const char* msg, unsigned int ndump)
{
    if(!m_records) return ;

    unsigned int ni = m_records->m_len0 ;
    unsigned int nj = m_records->m_len1 ;
    unsigned int nk = m_records->m_len2 ;
    assert( nj == 2 && nk == 4 );

    printf("%s numrec %d maxrec %d \n", msg, ni, m_maxrec );
    unsigned int unrec = 0 ; 

    for(unsigned int i=0 ; i<ni ; i++ )
    {
        bool unset = m_records->isUnsetItem(i);
        if(unset) unrec++ ;
        if(unset) continue ; 
        bool out = i < ndump || i > ni-ndump ; 
        if(out) dumpRecord(i);
    }    
    printf("unrec %d/%d \n", unrec, ni );
}


std::string RecordsNPY::getSequenceString(unsigned int photon_id, Types::Item_t etype)
{
    // express variable length sequence of bit positions as string of 
    std::stringstream ss ; 
    for(unsigned int r=0 ; r<m_maxrec ; r++)
    {
        unsigned int record_id = photon_id*m_maxrec + r ;
        bool unset = m_records->isUnsetItem(record_id);
        if(unset) continue ; 

        glm::uvec4 flag ; 
        unpack_material_flags(flag, record_id, 1, 2, 3);  // i,j,k0,k1

        unsigned int bit(0) ; 
        switch(etype)
        {
            case Types::MATERIAL:bit = flag.x  ;break; 
            case Types::HISTORY: bit = flag.w  ;break; 
            case Types::MATERIALSEQ: assert(0) ;break; 
            case Types::HISTORYSEQ:  assert(0) ;break; 
        }  
        assert(bit < 32);

        //ss << std::hex << std::setw(2) << std::setfill('0') << bit ; 

        std::string label = m_types->getMaskString(bit, etype) ;
        ss << m_types->getAbbrev(label, etype) ; 

    }
    return ss.str();
}


std::string RecordsNPY::decodeSequenceString(std::string& seq, Types::Item_t etype)
{
    assert(seq.size() % 2 == 0);
    std::stringstream ss ;
    unsigned int nelem = seq.size()/2 ; 
    for(unsigned int i=0 ; i < nelem ; i++)
    {
        std::string sub = seq.substr(i*2, 2) ;

        std::string label = m_types->getAbbrevInvert(sub, etype);

        //unsigned int bit = hex_lexical_cast<unsigned int>(sub.c_str());
        //ss << sub << ":" << bit << ":" << getMaskString( 1 << (bit-1) , etype) << " "  ; 
        //ss << m_types->getMaskString( 1 << (bit-1) , etype) << " "  ; 

        ss  << label << " " ; 


    }  
    return ss.str();
}



void RecordsNPY::constructFromRecord(unsigned int photon_id, unsigned int& bounce, unsigned int& history, unsigned int& material)
{
    bounce = 0 ; 
    history = 0 ; 
    material = 0 ; 

    for(unsigned int r=0 ; r<m_maxrec ; r++)
    {
        unsigned int record_id = photon_id*m_maxrec + r ;
        bool unset = m_records->isUnsetItem(record_id);
        if(unset) continue ; 

        glm::uvec4 flag ; 
        unpack_material_flags(flag, record_id, 1, 2, 3);  // i,j,k0,k1

        bounce += 1 ; 
        unsigned int  s_history = 1 << (flag.w - 1) ; 
        history |= s_history ;

        unsigned int s_material1 = 1 << (flag.x - 1) ; 
        unsigned int s_material2 = 1 << (flag.y - 1) ; 

        material |= s_material1 ; 
        material |= s_material2 ; 
    } 
}


