#include "G4StepNPY.hpp"
#include "NPY.hpp"
#include "Lookup.hpp"

void G4StepNPY::dump(const char* msg)
{
    if(!m_npy) return ;

    printf("%s\n", msg);

    unsigned int ni = m_npy->m_len0 ;
    unsigned int nj = m_npy->m_len1 ;
    unsigned int nk = m_npy->m_len2 ;
    std::vector<float>& data = m_npy->m_data ; 

    printf(" ni %u nj %u nk %u nj*nk %u \n", ni, nj, nk, nj*nk ); 

    uif_t uif ; 

    unsigned int check = 0 ;
    for(unsigned int i=0 ; i<ni ; i++ ){
    for(unsigned int j=0 ; j<nj ; j++ )
    {
       bool out = i == 0 || i == ni-1 ; 
       if(out) printf(" (%5u,%5u) ", i,j );
       for(unsigned int k=0 ; k<nk ; k++ )
       {
           unsigned int index = i*nj*nk + j*nk + k ;
           if(out)
           {
               uif.f = data[index] ;
               if( j == 0 || (j == 3 && k == 0)) printf(" %15d ",   uif.i );
               else         printf(" %15.3f ", uif.f );
           }
           assert(index == check);
           check += 1 ; 
       }
       if(out)
       {
           if( j == 0 ) printf(" sid/parentId/materialIndex/numPhotons ");
           if( j == 1 ) printf(" position/time ");
           if( j == 2 ) printf(" deltaPosition/stepLength ");
           if( j == 3 ) printf(" code ");

           printf("\n");
       }
    }
    }
}




bool G4StepNPY::applyLookup(unsigned int index)
{
    assert(m_lookup);


    std::vector<float>& data = m_npy->m_data ; 
    uif_t uif ; 

    uif.f = data[index] ;

    unsigned int acode = uif.u  ;
    int bcode = m_lookup->a2b(acode) ;

    if(true)
    {
        std::string aname = m_lookup->acode2name(acode) ;
        std::string bname = m_lookup->bcode2name(bcode) ;
        assert(aname == bname);
        //printf("  %3u -> %3d  [%s] \n", acode, bcode, aname.c_str() );
    }

    if( bcode > -1 )
    {
        uif.u = bcode ; 
        data[index] = uif.f ;
        m_lines.insert(bcode);
    }
    else
    {
        std::string aname = m_lookup->acode2name(acode) ;
        printf("G4StepNPY::applyLookup failed to translate acode %u : %s \n", acode, aname.c_str() );
    }

    return bcode > -1 ; 
}


int G4StepNPY::getStepId(unsigned int i)
{
    return m_npy->getInt(i,0,0);
}
bool G4StepNPY::isCerenkovStep(unsigned int i)
{
    return getStepId(i) < 0 ; 
}
bool G4StepNPY::isScintillationStep(unsigned int i)
{
    return getStepId(i) > 0 ; 
}



void G4StepNPY::applyLookup(unsigned int jj, unsigned int kk)
{
    unsigned int nfail = 0 ;

    unsigned int ni = m_npy->m_len0 ;
    unsigned int nj = m_npy->m_len1 ;
    unsigned int nk = m_npy->m_len2 ;

    for(unsigned int i=0 ; i<ni ; i++ ){
    for(unsigned int j=0 ; j<nj ; j++ ){
    for(unsigned int k=0 ; k<nk ; k++ ){

        if( j == jj && k == kk ) 
        { 
            unsigned int index = i*nj*nk + j*nk + k ;
            bool ok = applyLookup(index);
            if(!ok) nfail += 1 ;
        }
    }
    }
    }
    assert(nfail == 0);
}


void G4StepNPY::dumpLines(const char* msg)
{
    printf("%s\n", msg);
    for(Set_t::iterator it=m_lines.begin() ; it != m_lines.end() ; it++ )
    {
         unsigned int bcode = *it  ;
         std::string bname = m_lookup->bcode2name(bcode) ;
         printf("... %u [%s] \n", bcode, bname.c_str() ); 
    }   
}


