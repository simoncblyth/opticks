#include "RecordsNPY.hpp"

#include <glm/glm.hpp>

//npy-
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"
#include "Index.hpp"

#include "regexsearch.hh"

#include <sstream>
#include <iomanip>
#include <algorithm>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal




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

void RecordsNPY::unpack_material_flags_i(glm::ivec4& flag, unsigned int i, unsigned int j, unsigned int k0, unsigned int k1)
{
    charfour v = m_records->getChar4( i, j, k0, k1 ); 
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


void RecordsNPY::tracePath(unsigned int photon_id, float& length, float& distance, float& duration )
{
    std::vector<glm::vec4> posts ; 
    for(unsigned int r=0 ; r < m_maxrec ; r++ )
    {
        unsigned int record_id = photon_id*m_maxrec + r ; 
        bool unset = m_records->isUnsetItem(record_id);
        if(unset) continue ; 

        glm::vec4 post ; 
        unpack_position_time( post, record_id, 0 ); // i,j 
        posts.push_back(post);
    }

    length = 0.f ;
    unsigned int last = posts.size() - 1 ; 
    for(unsigned int i=1 ; i <= last ; i++)
    {
        glm::vec4 step = posts[i] - posts[i-1];
        length += glm::distance( posts[i], posts[i-1] );
    } 

    distance = glm::distance( posts[last], posts[0] );
    duration = posts[last].w - posts[0].w ; 
}

glm::vec4 RecordsNPY::getLengthDistanceDuration(unsigned int photon_id)
{
    float length ;
    float distance ;
    float duration  ;
    tracePath(photon_id, length, distance, duration );
    return glm::vec4(length, distance, 0.f, duration );
}


glm::vec4 RecordsNPY::getCenterExtent(unsigned int photon_id)
{
    glm::vec4 min(FLT_MAX) ;
    glm::vec4 max(-FLT_MAX) ;

    for(unsigned int r=0 ; r < m_maxrec ; r++ )
    {
        unsigned int record_id = photon_id*m_maxrec + r ; 
        bool unset = m_records->isUnsetItem(record_id);
        if(unset) continue ; 

        glm::vec4 post ; 
        unpack_position_time( post, record_id, 0 ); // i,j 

        min.x = std::min( min.x, post.x);
        min.y = std::min( min.y, post.y);
        min.z = std::min( min.z, post.z);
        min.w = std::min( min.w, post.w);

        max.x = std::max( max.x, post.x);
        max.y = std::max( max.y, post.y);
        max.z = std::max( max.z, post.z);
        max.w = std::max( max.w, post.w);
    }

    glm::vec4 rng = max - min ; 
   
    //print(max, "RecordsNPY::getCenterExtent max");
    //print(min, "RecordsNPY::getCenterExtent min");
    //print(rng, "RecordsNPY::getCenterExtent rng");

    float extent = 0.f ; 
    extent = std::max( rng.x , extent );
    extent = std::max( rng.y , extent );
    extent = std::max( rng.z , extent );
    extent = extent / 2.0f ;    
 
    glm::vec4 center_extent((min.x + max.x)/2.0f, (min.y + max.y)/2.0f , (min.z + max.z)/2.0f, extent ); 
    return center_extent ; 
}


void RecordsNPY::dumpRecord(unsigned int i, const char* msg)
{
    bool unset = m_records->isUnsetItem(i);
    if(unset) return ;

    glm::vec4  post ; 
    glm::vec4  polw ; 
    glm::uvec4 flag ; 
    glm::ivec4 iflag ; 

    unpack_position_time(           post, i, 0 );       // i,j 
    unpack_polarization_wavelength( polw, i, 1, 0, 1 ); // i,j,k0,k1
    unpack_material_flags(          flag, i, 1, 2, 3);  // i,j,k0,k1
    unpack_material_flags_i(       iflag, i, 1, 2, 3);  // i,j,k0,k1

    std::string m1 = m_types->findMaterialName(flag.x) ;
    std::string m2 = m_types->findMaterialName(flag.y) ;

    // flag.w is the result of ffs on a single set bit field, returning a 1-based bit position
    std::string history = m_types->getHistoryString( 1 << (flag.w-1) ); 

    //assert(flag.z == 0);  now set to bounday integer for debug 

    printf("%s %8u %s %s flag.x/m1 %2d:%25s flag.y/m2 %2d:%25s iflag.z [%3d] %s \n", 
                msg,
                i, 
                gpresent(post,2,11).c_str(),
                gpresent(polw,2,7).c_str(),
                flag.x,m1.c_str(),
                flag.y,m2.c_str(),
                iflag.z,
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




NPY<unsigned long long>* RecordsNPY::makeSequenceArray(Types::Item_t etype)
{
    unsigned int size = m_records->getShape(0)/m_maxrec ; 
    unsigned long long* seqdata = new unsigned long long[size] ; 
    for(unsigned int i=0 ; i < size ; i++)
    {
        seqdata[i] = getSequence(i, etype) ;
    }
    return NPY<unsigned long long>::make(size, 1, 1, seqdata);
}




unsigned long long RecordsNPY::getSequence(unsigned int photon_id, Types::Item_t etype)
{
    unsigned long long seq = 0ull ; 
    for(unsigned int r=0 ; r<m_maxrec ; r++)
    {
        unsigned int record_id = photon_id*m_maxrec + r ;
        bool unset = m_records->isUnsetItem(record_id);
        if(unset) break ; 

        glm::uvec4 flag ; 
        unpack_material_flags(flag, record_id, 1, 2, 3);  // i,j,k0,k1

        unsigned long long bitpos(0ull) ; 
        switch(etype)
        {
            case     Types::MATERIAL: bitpos = flag.x ; assert(0) ;break; 
            case      Types::HISTORY: bitpos = flag.w  ;break; 
            case  Types::MATERIALSEQ: assert(0)        ;break; 
            case   Types::HISTORYSEQ: assert(0)        ;break; 
        }  
        assert(bitpos < 16);
        seq |= bitpos << (r*4) ; 
    }
    return seq ; 
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

        // NB over all slots even if no records are written there, so 
        //    must handle unset if the above unset check fails...
        //
        //    hmm looks like a change in the unset values in the buffer 
        //    not reflected by update of NPY template specialization UNSET
        //

        glm::uvec4 flag ; 
        unpack_material_flags(flag, record_id, 1, 2, 3); // flag from m_records->getUChar4( i, j, k0, k1 );

        unsigned int bitpos(0) ; 
        switch(etype)
        {
            case     Types::MATERIAL: bitpos = flag.x  ;break; 
            case      Types::HISTORY: bitpos = flag.w  ;break; 
            case  Types::MATERIALSEQ: assert(0)        ;break; 
            case   Types::HISTORYSEQ: assert(0)        ;break; 
        }  
        assert(bitpos < 32);

        unsigned int bitmask = bitpos == 0 ? 0 : 1 << (bitpos - 1); // added handling of 0 
        assert(ffs(bitmask) == bitpos);

        if(ffs(bitmask) != bitpos)
        {
             LOG(warning) << "RecordsNPY::getSequenceString"
                          << " UNEXPECTED ffs(bitmask) != bitpos "
                          << " bitmask " << std::hex << bitmask << std::dec
                          << " ffs(bitmask) " <<  ffs(bitmask)
                          << " bitpos " << bitpos 
                          ; 

/*
In [16]: "%x" % (1 << 31)
Out[16]: '80000000'

In [18]: ffs_(1<<31)
Out[18]: 32

In [20]: ffs_(0)
Out[20]: 0

*/
        }

        std::string label = m_types->getMaskString( bitmask, etype) ;
        std::string abbrev = m_types->getAbbrev(label, etype) ; 

        //if(photon_id == 0) printf("bitpos %u bitmask %x label %s abbrev %s \n", bitpos, bitmask, label.c_str(), abbrev.c_str());

        ss << abbrev ;

    }
    return ss.str();
}


void RecordsNPY::appendMaterials(std::vector<unsigned int>& materials, unsigned int photon_id)
{
    for(unsigned int r=0 ; r<m_maxrec ; r++)
    {
        unsigned int record_id = photon_id*m_maxrec + r ;
        bool unset = m_records->isUnsetItem(record_id);
        if(unset) break ; 
        glm::uvec4 flag ; 
        unpack_material_flags(flag, record_id, 1, 2, 3);  // i,j,k0,k1

        materials.push_back(flag.x); ; 
        materials.push_back(flag.y); ; 
    }
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


