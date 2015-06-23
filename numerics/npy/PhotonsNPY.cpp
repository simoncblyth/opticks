#include "PhotonsNPY.hpp"
#include "uif.h"
#include "NPY.hpp"



#include <set>
#include <map>
#include <iostream>
#include <iomanip>

#include <glm/glm.hpp>
#include "limits.h"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"

#include "jsonutil.hpp" 
#include "regexsearch.hh"


bool value_order(const std::pair<int,int>&a, const std::pair<int,int>&b)
{
    return a.second > b.second ;
}


const char* PhotonsNPY::PHOTONS_ = "photons" ;
const char* PhotonsNPY::RECORDS_ = "records" ;


const char* PhotonsNPY::getItemName(Item_t item)
{
    const char* name(NULL);
    switch(item)
    {
       case PHOTONS:name = PHOTONS_ ; break ; 
       case RECORDS:name = RECORDS_ ; break ; 
    }
    return name ; 
}


void PhotonsNPY::classify(bool sign)
{
    m_boundaries.clear();
    m_boundaries = findBoundaries(sign);
    delete m_boundaries_selection ; 
    m_boundaries_selection = initBooleanSelection(m_boundaries.size());
    //dumpBoundaries("PhotonsNPY::classify");
}

bool* PhotonsNPY::initBooleanSelection(unsigned int n)
{
    bool* selection = new bool[n];
    while(n--) selection[n] = false ; 
    return selection ;
}

glm::ivec4 PhotonsNPY::getSelection()
{
    // ivec4 containing 1st four boundary codes provided by the selection

    int v[4] ;
    unsigned int count(0) ; 
    for(unsigned int i=0 ; i < m_boundaries.size() ; i++)
    {
        if(m_boundaries_selection[i])
        {
            std::pair<int, std::string> p = m_boundaries[i];
            if(count < 4)
            {
                v[count] = p.first ; 
                count++ ; 
            }
            else
            {
                 break ;
            }
        }
    }  
    glm::ivec4 iv(-INT_MAX,-INT_MAX,-INT_MAX,-INT_MAX);   // zero tends to be meaningful, so bad default for "unset"
    if(count > 0) iv.x = v[0] ;
    if(count > 1) iv.y = v[1] ;
    if(count > 2) iv.z = v[2] ;
    if(count > 3) iv.w = v[3] ;
    return iv ;     
}



void PhotonsNPY::dumpBoundaries(const char* msg)
{
    printf("%s\n", msg);
    for(unsigned int i=0 ; i < m_boundaries.size() ; i++)
    {
         std::pair<int, std::string> p = m_boundaries[i];
         printf(" %2d : %s \n", p.first, p.second.c_str() );
    }
}


std::vector<std::pair<int, std::string> > PhotonsNPY::findBoundaries(bool sign)
{
    assert(m_photons);

    std::vector<std::pair<int, std::string> > boundaries ;  

    printf("PhotonsNPY::findBoundaries \n");


    std::map<int,int> uniqn = sign ? m_photons->count_uniquei(3,0,2,0) : m_photons->count_uniquei(3,0) ;

    // To allow sorting by count
    //      map<boundary_code, count> --> vector <pair<boundary_code,count>>

    std::vector<std::pair<int,int> > pairs ; 
    for(std::map<int,int>::iterator it=uniqn.begin() ; it != uniqn.end() ; it++) pairs.push_back(*it);
    std::sort(pairs.begin(), pairs.end(), value_order );


    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        std::pair<int,int> p = pairs[i]; 
        int code = p.first ;
        std::string name ;
        if(m_names.count(abs(code)) > 0) name = m_names[abs(code)] ; 

        char line[128] ;
        snprintf(line, 128, " %3d : %7d %s ", p.first, p.second, name.c_str() );
        boundaries.push_back( std::pair<int, std::string>( code, line ));
    }   

    return boundaries ;
}



unsigned char msb_( unsigned short x )
{
    return ( x & 0xFF00 ) >> 8 ;
}

unsigned char lsb_( unsigned short x )
{
    return ( x & 0xFF)  ;
}


float PhotonsNPY::uncharnorm(unsigned char value, float center, float extent, float bitmax )
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


float PhotonsNPY::uncharnorm_polarization(unsigned char value)
{
    return uncharnorm(value, 1.f, 2.f, 254.f );
}
float PhotonsNPY::uncharnorm_wavelength(unsigned char value)
{
    return uncharnorm(value, -m_wavelength_domain.x , m_wavelength_domain.w, 255.f );
}


float PhotonsNPY::unshortnorm(short value, float center, float extent )
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

float PhotonsNPY::unshortnorm_position(short v, unsigned int k )
{
    assert(k < 3 );
    return unshortnorm( v, m_center_extent[k], m_center_extent.w );
}

float PhotonsNPY::unshortnorm_time(short v, unsigned int k )
{
    assert(k == 3 );
    return unshortnorm( v, m_time_domain.x,    m_time_domain.y );
}





void PhotonsNPY::unpack_position_time(glm::vec4& post, unsigned int i, unsigned int j)
{
    glm::uvec4 v = m_records->getQuadU( i, j );
    post.x = unshortnorm_position(v.x, 0);
    post.y = unshortnorm_position(v.y, 1);
    post.z = unshortnorm_position(v.z, 2);
    post.w = unshortnorm_time(v.w, 3);
}

void PhotonsNPY::unpack_polarization_wavelength(glm::vec4& polw, unsigned int i, unsigned int j, unsigned int k0, unsigned int k1)
{
    ucharfour v = m_records->getUChar4( i, j, k0, k1 ); 
    polw.x =  uncharnorm_polarization(v.x);  
    polw.y =  uncharnorm_polarization(v.y);  
    polw.z =  uncharnorm_polarization(v.z);  
    polw.w =  uncharnorm_wavelength(v.w);  
}

void PhotonsNPY::unpack_material_flags(glm::uvec4& flag, unsigned int i, unsigned int j, unsigned int k0, unsigned int k1)
{
    ucharfour v = m_records->getUChar4( i, j, k0, k1 ); 
    flag.x =  v.x ;  
    flag.y =  v.y ;  
    flag.z =  v.z ;  
    flag.w =  v.w ;  
}

void PhotonsNPY::dumpRecord(unsigned int i, const char* msg)
{
    bool unset = m_records->isUnsetItem(i);
    if(unset) return ;

    glm::vec4  post ; 
    glm::vec4  polw ; 
    glm::uvec4 flag ; 

    unpack_position_time(           post, i, 0 );       // i,j 
    unpack_polarization_wavelength( polw, i, 1, 0, 1 ); // i,j,k0,k1
    unpack_material_flags(          flag, i, 1, 2, 3);  // i,j,k0,k1

    std::string m1 = findMaterialName(flag.x) ;
    std::string m2 = findMaterialName(flag.y) ;
    std::string history = getStepFlagString( flag.w );

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


void PhotonsNPY::dumpRecords(const char* msg, unsigned int ndump)
{
    if(!m_records) return ;

    unsigned int ni = m_records->m_len0 ;
    unsigned int nj = m_records->m_len1 ;
    unsigned int nk = m_records->m_len2 ;
    assert( nj == 2 && nk == 4 );

    printf("%s numrec %d \n", msg, ni );
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





void PhotonsNPY::dump(const char* msg)
{
    if(!m_photons) return ;
    printf("%s\n", msg);

    unsigned int ni = m_photons->m_len0 ;
    unsigned int nj = m_photons->m_len1 ;
    unsigned int nk = m_photons->m_len2 ;

    assert( nj == 4 && nk == 4 );

    std::vector<float>& data = m_photons->m_data ; 

    printf(" ni %u nj %u nk %u nj*nk %u \n", ni, nj, nk, nj*nk ); 

    uif_t uif ; 

    unsigned int check = 0 ;
    for(unsigned int i=0 ; i<ni ; i++ ){
    for(unsigned int j=0 ; j<nj ; j++ )
    {
       bool out = i == 0 || i == ni-1 ; 

       if(out) printf(" (%7u,%1u) ", i,j );

       for(unsigned int k=0 ; k<nk ; k++ )
       {
           unsigned int index = i*nj*nk + j*nk + k ;
           assert(index == m_photons->getValueIndex(i,j,k));

           if(out)
           {
               uif.f = data[index] ;
               if(      j == 3 && k == 0 ) printf(" %15d ",   uif.i );
               else if( j == 3 && k == 3 ) printf(" %15d ",   uif.u );
               else                        printf(" %15.3f ", uif.f );
           }
           assert(index == check);
           check += 1 ; 
       }
       if(out)
       {
           if( j == 0 ) printf(" position/time ");
           if( j == 1 ) printf(" direction/wavelength ");
           if( j == 2 ) printf(" polarization/weight ");
           if( j == 3 ) printf(" boundary/cos_theta/distance_to_boundary/flags ");

           printf("\n");
       }
    }
    }
}





void PhotonsNPY::readFlags(const char* path)
{
    // read photon header to get flag names and enum values
    enum_regexsearch( m_flags, path);
    m_flags_selection = initBooleanSelection(m_flags.size());
}

void PhotonsNPY::dumpFlags(const char* msg)
{
    printf("%s\n", msg);
    for(unsigned int i=0 ; i < m_flags.size() ; i++)
    {
         std::pair<unsigned int, std::string> p = m_flags[i];
         printf(" %10d : %10x :  %s  : %d \n", p.first, p.first,  p.second.c_str(), m_flags_selection[i] );
    }
}


void PhotonsNPY::readMaterials(const char* idpath, const char* name)
{
    loadMap<std::string, unsigned int>(m_materials, idpath, name);
}
void PhotonsNPY::dumpMaterials(const char* msg)
{
    dumpMap<std::string, unsigned int>(m_materials, msg);
}

std::string PhotonsNPY::findMaterialName(unsigned int index)
{
    std::stringstream ss ; 
    ss << "findMaterialName-failed-" << index  ;
    std::string name = ss.str() ;
    typedef std::map<std::string, unsigned int> MSU ; 
    for(MSU::iterator it=m_materials.begin() ; it != m_materials.end() ; it++)
    {
        if( it->second == index )
        {
            name = it->first ;
            break ; 
        }
    }
    return name ; 
}


std::string PhotonsNPY::getHistoryString(unsigned int flags)
{
    std::stringstream ss ; 
    std::vector<std::string> names ; 
    for(unsigned int i=0 ; i < m_flags.size() ; i++)
    {
        std::pair<unsigned int, std::string> p = m_flags[i];
        unsigned int mask = p.first ;
        if(flags & mask) ss << p.second << " " ; 
    }
    return ss.str() ; 
}

std::string PhotonsNPY::getStepFlagString(unsigned char flag)
{
   // flag is the result of ffs on the bit field returning a 1-based bit position
   return getHistoryString( 1 << (flag-1) ); 
}



glm::ivec4 PhotonsNPY::getFlags()
{
    int flags(0) ;
    for(unsigned int i=0 ; i < m_flags.size() ; i++)
    {
        if(m_flags_selection[i]) flags |= m_flags[i].first ; 
    } 
    return glm::ivec4(flags,0,0,0) ;     
}






void PhotonsNPY::examinePhotonHistories()
{
    // find counts of all histories 

    typedef std::map<unsigned int,unsigned int>  muu_t ; 
    typedef std::pair<unsigned int,unsigned int> puu_t ;

    NPYBase* npy = getItem(PHOTONS);
    muu_t uu = ((NPY<float>*)npy)->count_unique_u(3,3) ; 

    std::vector<puu_t> pairs ; 
    for(muu_t::iterator it=uu.begin() ; it != uu.end() ; it++) pairs.push_back(*it);
    std::sort(pairs.begin(), pairs.end(), value_order );

    std::cout << "PhotonsNPY::examinePhotonHistories : " << std::endl ; 

    unsigned int total(0);
    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        puu_t p = pairs[i];
        unsigned int flags = p.first ;
        unsigned int count = p.second ; 
        total += count ;  

        std::cout 
               << std::setw(5) << i 
               << " : "
               << std::setw(10) << std::hex << flags 
               << " : " 
               << std::setw(10) << std::dec << count 
               << " : "
               << getHistoryString(flags) 
               << std::endl ; 
    }
    std::cout << " total " << total << std::endl ; 
}


void PhotonsNPY::examineRecordHistories()
{
    unsigned int ni = m_records->m_len0 ;
    unsigned int j = 1 ; 
    unsigned int k = 3 ;  

    std::vector<short>& rdata = m_records->m_data ; 

    hui_t hui ;

    unsigned int irec(0);
    unsigned int history(0) ; 
    unsigned int p_history(0) ; 
    unsigned int bounce(0) ; 

    typedef std::pair<unsigned int, unsigned int> PUU ;
    typedef std::map<unsigned int, unsigned int> MUU ;
    typedef std::vector<unsigned int> VU ; 
    MUU uu ;  


    VU mismatch ; 

    for(unsigned int i=0 ; i<ni ; i++ )
    {
        if(i % m_maxrec == 0)  // record start 
        {
            history = 0 ; 
            bounce  = 0 ; 
        }

        unsigned int index = m_records->getValueIndex(i, j, k);
        short value = rdata[index] ;
        hui.short_ = value ; 
        bool unset = value == SHRT_MIN ; 

        if(!unset) 
        {   
            bounce += 1 ; 
            unsigned short uvalue = hui.ushort_ ;
            unsigned char  msb = msb_(uvalue); 
            //unsigned char  lsb = lsb_(uvalue); 

            unsigned char s_flag = msb ; 
            unsigned int  s_history = 1 << (s_flag - 1) ; 

            history |= s_history ;
        }


        if(i % m_maxrec == m_maxrec - 1) // record end
        {
            assert(bounce > 0 );
            p_history = m_photons->getUInt(irec, 3, 3);
            if(p_history != history)
            {
                mismatch.push_back(irec);  // all mismatches are bounce 10 
                std::cout << std::setw(10) << irec
                          << "[" << std::setw(3)  << bounce << "]" 
                          << std::setw(80) << getHistoryString( p_history ) 
                          << " =/= " 
                          << std::setw(80) << getHistoryString( history ) 
                          << std::endl ;

            }   

            if(uu.count(history)==0) uu[history] = 1 ; 
            else                     uu[history] += 1 ; 

            irec++ ; 
        }
    }


    std::cout << "mismatch count " << mismatch.size() << std::endl ; 
    
     


    std::vector<PUU> pairs ; 
    for(MUU::iterator it=uu.begin() ; it != uu.end() ; it++) pairs.push_back(*it);
    std::sort(pairs.begin(), pairs.end(), value_order );

    std::cout << "PhotonsNPY::examineRecordHistories : " << std::endl ; 

    unsigned int total(0);

    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        PUU p = pairs[i];
        unsigned int history = p.first ;
        unsigned int count = p.second ; 
        total += count ;  

        std::cout 
               << std::setw(5) << i 
               << " : "
               << std::setw(10) << std::hex << history 
               << " : " 
               << std::setw(10) << std::dec << count 
               << " : "
               << getHistoryString(history) 
               << std::endl ; 
    }

    std::cout << " total " << total << std::endl ; 


}




