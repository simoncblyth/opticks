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


bool second_value_order(const std::pair<int,int>&a, const std::pair<int,int>&b)
{
    return a.second > b.second ;
}

bool su_second_value_order(const std::pair<std::string,unsigned int>&a, const std::pair<std::string,unsigned int>&b)
{
    return a.second > b.second ;
}





const char* PhotonsNPY::PHOTONS_ = "photons" ;
const char* PhotonsNPY::RECORDS_ = "records" ;
const char* PhotonsNPY::HISTORY_ = "history" ;
const char* PhotonsNPY::MATERIAL_ = "material" ;



const char* PhotonsNPY::getItemName(Item_t item)
{
    const char* name(NULL);
    switch(item)
    {
       case PHOTONS:name = PHOTONS_ ; break ; 
       case RECORDS:name = RECORDS_ ; break ; 
       case HISTORY:name = HISTORY_ ; break ; 
       case MATERIAL:name = MATERIAL_ ; break ; 
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
    std::sort(pairs.begin(), pairs.end(), second_value_order );


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

    // flag.w is the result of ffs on a single set bit field, returning a 1-based bit position
    std::string history = getHistoryString( 1 << (flag.w-1) ); 

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



void PhotonsNPY::dumpPhotons(const char* msg, unsigned int ndump)
{
    if(!m_photons) return ;
    printf("%s\n", msg);

    unsigned int ni = m_photons->m_len0 ;
    unsigned int nj = m_photons->m_len1 ;
    unsigned int nk = m_photons->m_len2 ;
    assert( nj == 4 && nk == 4 );

    for(unsigned int i=0 ; i<ni ; i++ )
    {
        bool out = i < ndump || i > ni-ndump ; 
        if(out) dumpPhotonRecord(i);
    }
}


void PhotonsNPY::dumpPhotonRecord(unsigned int photon_id, const char* msg)
{
    printf("%s\n", msg);
    for(unsigned int r=0 ; r<m_maxrec ; r++)
    {
        unsigned int record_id = photon_id*m_maxrec + r ;
        dumpRecord(record_id);
    }  
    dumpPhoton(photon_id);
    printf("\n");
}


void PhotonsNPY::dumpPhoton(unsigned int i, const char* msg)
{
    unsigned int history = m_photons->getUInt(i, 3, 3);
    std::string phistory = getHistoryString( history );

    glm::vec4 post = m_photons->getQuad(i,0);
    glm::vec4 dirw = m_photons->getQuad(i,1);
    glm::vec4 polw = m_photons->getQuad(i,2);


    std::string seqmat = getSequenceString(i, MATERIAL) ;
    std::string seqhis = getSequenceString(i, HISTORY) ;

    std::string dseqmat = decodeSequenceString(seqmat, MATERIAL);
    std::string dseqhis = decodeSequenceString(seqhis, HISTORY);


    printf("%s %8u %s %s %25s %25s %s \n", 
                msg,
                i, 
                gpresent(post,2,11).c_str(),
                gpresent(polw,2,7).c_str(),
                seqmat.c_str(),
                seqhis.c_str(),
                phistory.c_str());

    printf("%s\n", dseqmat.c_str());
    printf("%s\n", dseqhis.c_str());

    

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

std::string PhotonsNPY::getMaterialString(unsigned int mask)
{
    std::stringstream ss ; 
    typedef std::map<std::string, unsigned int> MSU ; 
    for(MSU::iterator it=m_materials.begin() ; it != m_materials.end() ; it++)
    {
        unsigned int mat = it->second ;
        if(mask & (1 << (mat-1))) ss << it->first << " " ; 
    }
    return ss.str() ; 
}

std::string PhotonsNPY::getHistoryString(unsigned int flags)
{
    std::stringstream ss ; 
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
    typedef std::map<unsigned int,unsigned int>  MUU ; 

    MUU uu = m_photons->count_unique_u(3,3) ; 

    dumpMaskCounts("PhotonsNPY::examinePhotonHistories : ", HISTORY, uu, 1);
}

void PhotonsNPY::constructFromRecord(unsigned int photon_id, unsigned int& bounce, unsigned int& history, unsigned int& material)
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


std::string PhotonsNPY::getSequenceString(unsigned int photon_id, Item_t etype)
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
            case PHOTONS:               ;break; 
            case RECORDS:               ;break; 
            case MATERIAL:bit = flag.x  ;break; 
            case HISTORY: bit = flag.w  ;break; 
        }  
        assert(bit < 32);
        ss << std::hex << std::setw(2) << std::setfill('0') << bit ; 
    }
    return ss.str();
}

std::string PhotonsNPY::decodeSequenceString(std::string& seq, Item_t etype)
{
    assert(seq.size() % 2 == 0);
    std::stringstream ss ;
    unsigned int nelem = seq.size()/2 ; 
    for(unsigned int i=0 ; i < nelem ; i++)
    {
        std::string sub = seq.substr(i*2, 2) ;
        unsigned int bit = hex_lexical_cast<unsigned int>(sub.c_str());
        //ss << sub << ":" << bit << ":" << getMaskString( 1 << (bit-1) , etype) << " "  ; 
        ss << getMaskString( 1 << (bit-1) , etype) << " "  ; 
    }  
    return ss.str();
}

void PhotonsNPY::makeSequenceIndex(
       std::map<std::string, std::vector<unsigned int> >  mat, 
       std::map<std::string, std::vector<unsigned int> >  his 
)
{
    unsigned int ni = m_photons->m_len0 ;
    m_seqidx = NPY<unsigned char>::make_vec4(ni,1,0) ;

    for(unsigned int i=0 ; i < ni ; i++)
    { 
         unsigned int photon_id = i ; 
         glm::uvec4 seqidx ;

         // loop over important indices checking if photon_id is there

         // hmm need to sort the counts to find the important sequences
         // and then can construct indices (<255) and index structure like GItemIndex 
         // for common seqs

         m_seqidx->setQuad(photon_id, 0, seqidx );
    }
}

void PhotonsNPY::prepSequenceIndex()
{
    unsigned int nr = m_records->m_len0 ;
    unsigned int ni = m_photons->m_len0 ;

    unsigned int history(0) ;
    unsigned int bounce(0) ;
    unsigned int material(0) ;

    typedef std::map<std::string, unsigned int>  MSU ;
    typedef std::map<std::string, std::vector<unsigned int> >  MSV ;
    typedef std::map<unsigned int, unsigned int> MUU ;

    std::vector<unsigned int> mismatch ;
    MUU uuh ;  
    MUU uum ;  
    MSU sum ;  
    MSU suh ;  
    MSV svh ;  
    MSV svm ;  

    for(unsigned int i=0 ; i < ni ; i++)
    { 
         unsigned int photon_id = i ; 

         constructFromRecord(photon_id, bounce, history, material);

         unsigned int phistory = m_photons->getUInt(photon_id, 3, 3);

         if(history != phistory) mismatch.push_back(photon_id);

         std::string seqmat = getSequenceString(photon_id, MATERIAL);

         std::string seqhis = getSequenceString(photon_id, HISTORY);

         assert(history == phistory);

         uuh[history] += 1 ; 

         uum[material] += 1 ; 

         suh[seqhis] += 1; 

         svh[seqhis].push_back(photon_id); 

         sum[seqmat] += 1 ; 

         svm[seqmat].push_back(photon_id); 

    }
    assert( mismatch.size() == 0);

    printf("PhotonsNPY::consistencyCheck photons %u records %u mismatch %lu \n", ni, nr, mismatch.size());
    dumpMaskCounts("PhotonsNPY::consistencyCheck histories", HISTORY, uuh, 1 );
    dumpMaskCounts("PhotonsNPY::consistencyCheck materials", MATERIAL, uum, 1000 );
    dumpSequenceCounts("PhotonsNPY::consistencyCheck seqhis", HISTORY, suh , svh, 1000);
    dumpSequenceCounts("PhotonsNPY::consistencyCheck seqmat", MATERIAL, sum , svm, 1000);

    makeSequenceIndex( svm, svh );
}



std::string PhotonsNPY::getMaskString(unsigned int mask, Item_t etype)
{
    std::string mstr ;
    switch(etype)
    {
       case HISTORY:mstr = getHistoryString(mask) ; break ; 
       case MATERIAL:mstr = getMaterialString(mask) ; break ; 
       case PHOTONS:mstr = "??" ; break ; 
       case RECORDS:mstr = "??" ; break ; 
    }
    return mstr ; 
}


void PhotonsNPY::dumpMaskCounts(const char* msg, Item_t etype, 
        std::map<unsigned int, unsigned int>& uu, 
        unsigned int cutoff)
{
    typedef std::map<unsigned int, unsigned int> MUU ;
    typedef std::pair<unsigned int, unsigned int> PUU ;

    std::vector<PUU> pairs ; 
    for(MUU::iterator it=uu.begin() ; it != uu.end() ; it++) pairs.push_back(*it);
    std::sort(pairs.begin(), pairs.end(), second_value_order );

    std::cout << msg << std::endl ; 

    unsigned int total(0);

    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        PUU p = pairs[i];
        total += p.second ;  

        if(p.second > cutoff) 
            std::cout 
               << std::setw(5) << i 
               << " : "
               << std::setw(10) << std::hex << p.first
               << " : " 
               << std::setw(10) << std::dec << p.second
               << " : "
               << getMaskString(p.first, etype) 
               << std::endl ; 
    }

    std::cout 
              << " total " << total 
              << " cutoff " << cutoff 
              << std::endl ; 
}




void PhotonsNPY::dumpSequenceCounts(const char* msg, Item_t etype, 
       std::map<std::string, unsigned int>& su,
       std::map<std::string, std::vector<unsigned int> >& sv,
       unsigned int cutoff
    )
{
    typedef std::map<std::string, unsigned int> MSU ;
    typedef std::pair<std::string, unsigned int> PSU ;

    std::vector<PSU> pairs ; 
    for(MSU::iterator it=su.begin() ; it != su.end() ; it++) pairs.push_back(*it);
    std::sort(pairs.begin(), pairs.end(), su_second_value_order );

    std::cout << msg << std::endl ; 

    unsigned int total(0);

    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        PSU p = pairs[i];
        total += p.second ;  

        assert( sv[p.first].size() == p.second );

        if(p.second > cutoff)
            std::cout 
               << std::setw(5) << i          
               << " : "
               << std::setw(30) << p.first
               << " : " 
               << std::setw(10) << std::dec << p.second
               << std::setw(10) << std::dec << sv[p.first].size()
               << " : "
               << decodeSequenceString(p.first, etype) 
               << std::endl ; 
    }

    std::cout 
              << " total " << total 
              << " cutoff " << cutoff 
              << std::endl ; 

}



