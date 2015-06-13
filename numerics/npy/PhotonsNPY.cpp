#include "PhotonsNPY.hpp"
#include "uif.h"
#include "NPY.hpp"



#include <set>
#include <map>
#include <iostream>
#include <iomanip>

#include <glm/glm.hpp>
#include "limits.h"

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


void PhotonsNPY::dumpRecords(const char* msg, unsigned int ndump, unsigned int maxrec)
{
    if(!m_records) return ;

    unsigned int ni = m_records->m_len0 ;
    unsigned int nj = m_records->m_len1 ;
    unsigned int nk = m_records->m_len2 ;

    assert( nj == 2 && nk == 4 );

    printf("%s numrec %d \n", msg, ni );

    std::vector<short>& data = m_records->m_data ; 

    hui_t hui ;
 
    unsigned short prior_flags(0); 
    unsigned int check = 0 ;
    for(unsigned int i=0 ; i<ni ; i++ ){


    if(i % maxrec == 0)
    {
         prior_flags = 0 ;
    }
    std::string history ;


    for(unsigned int j=0 ; j<nj ; j++ )
    {
       bool out = i < ndump || i > ni-ndump ; 

      
       

       if(out) printf(" (%7u,%1u) ", i,j );

       for(unsigned int k=0 ; k<nk ; k++ )
       {
           unsigned int index = i*nj*nk + j*nk + k ;
           assert(index == m_records->getValueIndex(i,j,k));

           short value = data[index] ;
           hui.short_ = value ; 

           unsigned short uvalue = hui.ushort_ ;

           bool unset = value == SHRT_MIN ; 

           if(out)
           {
               if( unset )  
               {
                    printf(" %15s ",  "..." );
               }
               else if( j == 1 && k == 3 )
               {
                    printf(" %15d ",   uvalue );
                    history = getHistoryString( uvalue ^ prior_flags ) ; // hmm does it make sense to do this bitfield delta-ing  on GPU 
                    prior_flags = uvalue ; 
               }
               else 
               {                       
                    printf(" %15d ",   value  );
               }

               if( k == nk - 1)
               {
                   if( j == 0 && !unset ) printf(" position/time (packed) ");
                   //if( j == 1 && !unset ) printf(" polarization/wavelength/boundary/flags (packed) ");
                   if( j == 1 && !unset ) printf("%s ", history.c_str());
                   printf("\n");
               }
           }
           assert(index == check);
           check += 1 ; 

       }
    }
    }
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



glm::ivec4 PhotonsNPY::getFlags()
{
    int flags(0) ;
    for(unsigned int i=0 ; i < m_flags.size() ; i++)
    {
        if(m_flags_selection[i]) flags |= m_flags[i].first ; 
    } 
    return glm::ivec4(flags,0,0,0) ;     
}






void PhotonsNPY::examineHistories(Item_t item)
{
    // find counts of all histories 

    typedef std::map<unsigned int,unsigned int>  muu_t ; 
    typedef std::pair<unsigned int,unsigned int> puu_t ;

    NPYBase* npy = getItem(item);
    muu_t uu ;
    switch(item)
    {
       case PHOTONS: uu = ((NPY<float>*)npy)->count_unique_u(3,3) ; break ;
       case RECORDS: uu = ((NPY<short>*)npy)->count_unique_u(1,3) ; break ;
    }

    std::vector<puu_t> pairs ; 
    for(muu_t::iterator it=uu.begin() ; it != uu.end() ; it++) pairs.push_back(*it);
    std::sort(pairs.begin(), pairs.end(), value_order );

    std::cout << "PhotonsNPY::examineHistories : " << getItemName(item) << std::endl ; 

    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        puu_t p = pairs[i];
        unsigned int flags = p.first ;
        std::cout 
               << std::setw(5) << i 
               << " : "
               << std::setw(10) << std::hex << flags 
               << " : " 
               << std::setw(10) << std::dec << p.second
               << " : "
               << getHistoryString(flags) 
               << std::endl ; 
    }

}


