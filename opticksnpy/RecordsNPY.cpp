#include "RecordsNPY.hpp"

#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <glm/glm.hpp>

//brap-
#include "BBit.hh"

// npy-
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"
#include "NPY.hpp"
#include "Index.hpp"
#include "Typ.hpp"

#include "PLOG.hh"

RecordsNPY::RecordsNPY(NPY<short>* records, unsigned int maxrec)
    :
    m_records(records),
    m_maxrec(maxrec),
    m_flat(false),
    m_types(NULL),
    m_typ(NULL)
{
}

NPY<short>* RecordsNPY::getRecords()
{
    return m_records; 
}

bool RecordsNPY::isFlat()
{
    return m_flat ; 
}

unsigned int RecordsNPY::getMaxRec()
{
    return m_maxrec ; 
}


void RecordsNPY::setCenterExtent(glm::vec4& ce)
{
    m_center_extent = ce ; 
}
void RecordsNPY::setTimeDomain(glm::vec4& td)
{
    m_time_domain = td ; 
}
void RecordsNPY::setWavelengthDomain(glm::vec4& wd)
{
    m_wavelength_domain = wd ; 
}
void RecordsNPY::setTypes(Types* types)
{  
    m_types = types ; 
}
void RecordsNPY::setTyp(Typ* typ)
{  
    m_typ = typ ; 
}





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



#if defined(_MSC_VER)
// conversion from 'glm::uint' to 'short'
#pragma warning( disable : 4244 )
#endif



void RecordsNPY::unpack_position_time(glm::vec4& post, unsigned int i, unsigned int j, unsigned int k)
{
    /*
    1373     def rpost_(self, recs):
    1374         """
    1375         NB recs can be a slice, eg slice(0,5) for 1st 5 step records of each photon
    ....
    1390         """
    1391         center, extent = self.post_center_extent()
    1392         p = self.rx[:,recs,0].astype(np.float32)*extent/32767.0 + center
    1393         return p
    1394 
    */
    glm::uvec4 v = m_records->getQuadU( i, j, k);
    post.x = unshortnorm_position(v.x, 0);
    post.y = unshortnorm_position(v.y, 1);
    post.z = unshortnorm_position(v.z, 2);
    post.w = unshortnorm_time(v.w, 3);
}

void RecordsNPY::unpack_polarization_wavelength(glm::vec4& polw, unsigned int i, unsigned int j, unsigned int k, unsigned int l0, unsigned int l1)
{
    /*
    1203     def rpolw_(self, recs):
    1204         """
    1205         Unlike rpol_ this works with irec slices, 
    1206         BUT note that the wavelength returned in 4th column is 
    1207         not decompressed correctly.
    1208         Due to shape shifting it is not easy to remove
    1209         """
    1210         return self.rx[:,recs,1,0:2].copy().view(np.uint8).astype(np.float32)/127.-1.  
    */
    ucharfour v = m_records->getUChar4( i, j, k, l0, l1 ); 

    polw.x =  uncharnorm_polarization(v.x);  
    polw.y =  uncharnorm_polarization(v.y);  
    polw.z =  uncharnorm_polarization(v.z);  
    polw.w =  uncharnorm_wavelength(v.w);  


    /*
    LOG(info) << "RecordsNPY::unpack_polarization_wavelength" 
              << " ijk (" << i << "," <<  j << "," << k << ")"
              << " l0l1 [" << l0 << "," << l1 << "]"
              << " v.xyzw " << int(v.x) << "," << int(v.y) << "," << int(v.z)  << "," << int(v.w)  
              << " polw " << gformat(polw) 
              ; 

    */

}

void RecordsNPY::unpack_material_flags(glm::uvec4& flag, unsigned int i, unsigned int j, unsigned int k, unsigned int l0, unsigned int l1)
{
    ucharfour v = m_records->getUChar4( i, j, k, l0, l1 ); 
    flag.x =  v.x ;  
    flag.y =  v.y ;  
    flag.z =  v.z ;  
    flag.w =  v.w ;  
}

void RecordsNPY::unpack_material_flags_i(glm::ivec4& flag, unsigned int i, unsigned int j, unsigned int k, unsigned int l0, unsigned int l1)
{
    charfour v = m_records->getChar4( i, j, k, l0, l1 ); 
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
        unsigned int i = m_flat ? record_id : photon_id ;
        unsigned int j = m_flat ? 0         : r ;

        bool unset = m_records->isUnsetItem(i, j);
        if(unset) continue ; 

        glm::vec4 post ; 
        unpack_position_time( post, i, j,  0 );
        posts.push_back(post);
    }

    length = 0.f ;

    if(posts.size() == 0) 
    {
        LOG(warning) << "RecordsNPY::tracePath" 
                     << " no posts " ;
        return ;  
    }

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
    float length(0.f) ;
    float distance(0.f) ;
    float duration(0.f)  ;
    tracePath(photon_id, length, distance, duration );
    return glm::vec4(length, distance, 0.f, duration );
}


glm::vec4 RecordsNPY::getCenterExtent(unsigned int photon_id)
{
    glm::vec4 min(FLT_MAX) ;
    glm::vec4 max(-FLT_MAX) ;

    for(unsigned int r=0 ; r < m_maxrec ; r++ )
    {
        //unsigned int record_id = photon_id*m_maxrec + r ; 
        bool unset = m_records->isUnsetItem(photon_id, r);
        if(unset) continue ; 

        glm::vec4 post ; 


        unsigned int record_id = photon_id*m_maxrec + r ; 
        unsigned int i = m_flat ? record_id : photon_id ;
        unsigned int j = m_flat ? 0         : r ;

        unpack_position_time( post, i, j,  0 ); // i,j,k 

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



bool RecordsNPY::exists(unsigned int photon_id , unsigned int r )
{
    unsigned int record_id = photon_id*m_maxrec + r ;
    unsigned int i = m_flat ? record_id : photon_id ;
    unsigned int j = m_flat ? 0         : r ;
    bool unset = m_records->isUnsetItem(i, j);
    return !unset ;  
}

void RecordsNPY::unpack_material_flags(glm::uvec4& flag, unsigned int photon_id , unsigned int r )
{
    unsigned int record_id = photon_id*m_maxrec + r ;
    unsigned int i = m_flat ? record_id : photon_id ;
    unsigned int j = m_flat ? 0         : r ;

    bool unset = m_records->isUnsetItem(i, j);
    assert(!unset);

    unpack_material_flags(flag, i,j,1, 2, 3);  // i,j,k0,k1
}




/*


App::indexEvtOld dpho 0
rec        0 (       79.68      32.45     300.00         0.10) (    0.70  -0.90   1.01   438.51) flag.x/m1 72:                        ? flag.y/m2  1:                GdDopedLS iflag.z [ 79]  
rec        0 (      -25.06     251.95      32.85         1.09) (    0.00  -0.16  -0.97   101.73) flag.x/m1 228:                        ? flag.y/m2 13:                   Vacuum iflag.z [ -1]  
rec        0 (       79.68      32.45      97.79         1.10) (    0.70  -0.90   0.46   182.20) flag.x/m1 21:                 Nitrogen flag.y/m2 14:                    Pyrex iflag.z [ 79]  
rec        0 (      -25.06     251.95      11.85         0.98) (    0.00  -0.16  -0.89    74.90) flag.x/m1 130:                        ? flag.y/m2 12:                     Rock iflag.z [ 16]  
rec        0 (       79.84      32.51      91.68         1.13) (    0.76  -0.90  -0.76   176.24) flag.x/m1 118:                        ? flag.y/m2 14:                    Pyrex iflag.z [ 79]  
rec        0 (      -25.06     251.95      11.85         0.66) (    0.00  -0.16  -0.89    74.90) flag.x/m1 130:                        ? flag.y/m2  8:                DeadWater iflag.z [  0]  
[2016-Mar-11 11:28:34.840599]:fatal: RecordsNPY::getSequenceString bitpos out of range 72 bitmax 40 record 0 photon_id 0 flag 72,1,79,245 etype 1
[2016-Mar-11 11:28:34.840811]:fatal: RecordsNPY::getSequenceString bitpos out of range 118 bitmax 40 record 2 photon_id 0 flag 118,14,79,245 etype 1
[2016-Mar-11 11:28:34.840955]:fatal: RecordsNPY::getSequenceString bitpos out of range 245 bitmax 32 record 0 photon_id 0 flag 72,1,79,245 etype 0
[2016-Mar-11 11:28:34.841101]:fatal: RecordsNPY::getSequenceString bitpos out of range 245 bitmax 32 record 1 photon_id 0 flag 21,14,79,245 etype 0
[2016-Mar-11 11:28:34.841257]:fatal: RecordsNPY::getSequenceString bitpos out of range 245 bitmax 32 record 2 photon_id 0 flag 118,14,79,245 etype 0
pho        0 (       79.84      32.51      91.68         1.13) (   -0.38   0.93  -0.00   380.00)                    ERRERR                 ERRERRERR  



In [18]: evt.rpost_(slice(0,5)).shape
Out[18]: (500000, 5, 4)

In [19]: evt.rpost_(slice(0,5))[0]
Out[19]: 
A(    [[  79.681,   32.447,  300.   ,    0.1  ],
       [  79.681,   32.447,   97.79 ,    1.1  ],
       [  79.836,   32.511,   91.684,    1.13 ],
       [   0.   ,    0.   ,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ]])





*/

void RecordsNPY::dumpRecord(unsigned int i, unsigned int j, const char* msg)
{
    bool unset = m_records->isUnsetItem(i, j);
    if(unset) return ;

    glm::vec4  post ; 
    glm::vec4  polw ; 
    glm::uvec4 flag ; 
    glm::ivec4 iflag ; 

    // flat records means that the photon_id and record number occupy the i slot 
    // formerly records was flat 

    //LOG(info) << "RecordsNPY::dumpRecord ij " << i  << "," << j ;

    unpack_position_time(           post, i, j, 0 );       // i,j,k
    unpack_polarization_wavelength( polw, i, j, 1, 0, 1 ); // i,j,k,l0,l1

    unpack_material_flags(          flag, i, j, 1, 2, 3);  // i,j,k,l0,l1
    unpack_material_flags_i(       iflag, i, j, 1, 2, 3);  // i,j,k,l0,l1

    // for debug see npy-/evt.py 


    std::string m1 = m_typ ? m_typ->findMaterialName(flag.x) : "notyp" ;
    std::string m2 = m_typ ? m_typ->findMaterialName(flag.y) : "notyp" ;

    // flag.w is the result of ffs on a single set bit field, returning a 1-based bit position
    std::string history = m_types ? m_types->getHistoryString( 1 << (flag.w-1) ) : "notyps" ; 

    //assert(flag.z == 0);  now set to bounday integer for debug 

    printf("%s %8u post %s polw %s flag.x/m1 %2d:%25s flag.y/m2 %2d:%25s iflag.z [%3d] %s \n", 
                msg,
                i, 
                gpresent(post,2,11).c_str(),
                gpresent(polw,2,7).c_str(),
                flag.x,m1.c_str(),
                flag.y,m2.c_str(),
                iflag.z,
                history.c_str());

}

/*


In [4]: r = np.load("/usr/local/env/opticks/dayabay/rxcerenkov/1.npy")

In [19]: r.reshape(-1,10,2,4)[4]
Out[19]: 
array([[[  -77,   119,   251,   140],
        [ 3178, 31054,  1542,   494]],

       [[ -575,   292,    54,   238],
        [-2147, 31131,  9478,  3309]],

       [[-1993,   784,  -507,   517],
        [-4929, 31089,  1542,  2322]],

       [[ -794,   187,   293,   789],
        [17319, 30998,  1542,  3310]],

       [[  366,  -391,  1068,  1052],
        [17319, 30998,  9478,  3247]],

       [[ 1463,  -937,  1801,  1300],
        [17319, 30998,  9478,  1199]],

       [[    0,     0,     0,     0],
        [    0,     0,     0,     0]],

       [[    0,     0,     0,     0],
        [    0,     0,     0,     0]],

       [[    0,     0,     0,     0],
        [    0,     0,     0,     0]],

       [[    0,     0,     0,     0],
        [    0,     0,     0,     0]]], dtype=int16)



*/






void RecordsNPY::dumpRecords(const char* msg, unsigned int ndump)
{
    if(!m_records) return ;

    unsigned int ni = m_records->m_ni ;
    unsigned int nj = m_records->m_nj ;
    unsigned int nk = m_records->m_nk ;
    unsigned int nl = m_records->m_nl ;
    assert( nk == 2 && nl == 4 );

    printf("%s numrec %d maxrec %d \n", msg, ni, m_maxrec );
    unsigned int unrec = 0 ; 

    for(unsigned int i=0 ; i<ni ; i++ )
    {
        for(unsigned int j=0 ; j<nj ; j++ )
        {
            bool unset = m_records->isUnsetItem(i, j);
            if(unset) unrec++ ;
            if(unset) continue ; 
            bool out = i < ndump || i > ni-ndump ; 
            if(out) dumpRecord(i,j);
        }
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
    NPY<unsigned long long>* npy = NPY<unsigned long long>::make(size, 1, 1);
    npy->setData(seqdata);
    return npy ; 
}






unsigned long long RecordsNPY::getSequence(unsigned int photon_id, Types::Item_t etype)
{
    unsigned long long seq = 0ull ; 
    for(unsigned int r=0 ; r<m_maxrec ; r++)
    {
        if(!exists(photon_id, r )) break ;

        glm::uvec4 flag ; 
        unpack_material_flags(flag, photon_id, r);

        unsigned long long bitpos(0ull) ; 
        switch(etype)
        {
            case     Types::MATERIAL: bitpos = flag.x ; assert(0) ;break; 
            case      Types::HISTORY: bitpos = flag.w  ;break; 
            case  Types::MATERIALSEQ: assert(0)        ;break; 
            case   Types::HISTORYSEQ: assert(0)        ;break; 
        }  


        if(bitpos >= 16)
        {
            LOG(fatal) << "RecordsNPY::getSequence"
                       << " UNEXPECTED bitpos " << bitpos
                        ; 
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
        if(!exists(photon_id, r )) continue ;

        glm::uvec4 flag ; 
        unpack_material_flags(flag, photon_id, r);

        unsigned int bitpos(0) ; 
        unsigned int bitmax(0) ; 
        switch(etype)
        {
            case     Types::MATERIAL: bitpos = flag.x ; bitmax = 40 ;break; 
            case      Types::HISTORY: bitpos = flag.w ; bitmax = 32  ;break; 
            case  Types::MATERIALSEQ: assert(0)        ;break; 
            case   Types::HISTORYSEQ: assert(0)        ;break; 
        }  

        // test materials are often exceeding the hard history limit of 32
        if(bitpos >= bitmax)
        {
            LOG(fatal) << "RecordsNPY::getSequenceString"
                       << " bitpos out of range "  << bitpos
                       << " bitmax " << bitmax
                       << " record " << r
                       << " photon_id " << photon_id
                       << " flag " << gformat(flag)
                       << " etype " << etype
                       ;
                

        } 
        //assert(bitpos < 32);

        std::string abbrev = "ERR" ; 
        if(bitpos < 32)
        { 
            unsigned int bitmask = bitpos == 0 ? 0 : 1 << (bitpos - 1); 
            unsigned int first = BBit::ffs(bitmask) ;
            assert(first == bitpos);
            if(first != bitpos)
            {
                 LOG(warning) << "RecordsNPY::getSequenceString"
                              << " UNEXPECTED ffs(bitmask) != bitpos "
                              << " bitmask " << std::hex << bitmask << std::dec
                              << " ffs(bitmask) " <<  BBit::ffs(bitmask)
                              << " bitpos " << bitpos 
                              ; 

            }
            std::string label = m_types->getMaskString( bitmask, etype) ;
            abbrev = m_types->getAbbrev(label, etype) ; 
        }

        //if(photon_id == 0) printf("bitpos %u bitmask %x label %s abbrev %s \n", bitpos, bitmask, label.c_str(), abbrev.c_str());

        ss << abbrev ;

    }
    return ss.str();
}


/*

32 bit limit::

    In [16]: "%x" % (1 << 31)
    Out[16]: '80000000'

    In [41]: ffs_??
    Type:       function
    String Form:<function <lambda> at 0x10575d7d0>
    File:       /usr/local/env/chroma_env/lib/python2.7/site-packages/env/numerics/npy/types.py
    Definition: ffs_(_)
    Source:     ffs_ = lambda _:libcpp.ffs(_)


    In [20]: ffs_(0)
    Out[20]: 0

    In [35]: ffs_( 1 << (31-1) )
    Out[35]: 31

    In [36]: ffs_( 1 << (32-1) )
    Out[36]: 32

    In [37]: ffs_( 1 << (33-1) )
    Out[37]: 0

    In [38]: ffs_( 1 << (34-1) )
    Out[38]: 0

    In [39]: ffs_( 1 << (35-1) )
    Out[39]: 0

    In [40]: ffs_( 1 << (64-1) )
    Out[40]: 0

*/





void RecordsNPY::appendMaterials(std::vector<unsigned int>& materials, unsigned int photon_id)
{
    for(unsigned int r=0 ; r<m_maxrec ; r++)
    {
        if(!exists(photon_id, r )) continue ;

        glm::uvec4 flag ; 
        unpack_material_flags(flag, photon_id, r);

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
        if(!exists(photon_id, r)) continue ;

        glm::uvec4 flag ; 
        unpack_material_flags(flag, photon_id, r);

        bounce += 1 ; 
        unsigned int  s_history = 1 << (flag.w - 1) ; 
        history |= s_history ;

        unsigned int s_material1 = 1 << (flag.x - 1) ; 
        unsigned int s_material2 = 1 << (flag.y - 1) ; 

        material |= s_material1 ; 
        material |= s_material2 ; 
    } 
}

