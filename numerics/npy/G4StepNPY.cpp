#include "G4StepNPY.hpp"
#include "NPY.hpp"
#include "Lookup.hpp"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal




void G4StepNPY::dump(const char* msg)
{
    if(!m_npy) return ;

    printf("%s\n", msg);

    unsigned int ni = m_npy->m_ni ;
    unsigned int nj = m_npy->m_nj ;
    unsigned int nk = m_npy->m_nk ;
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




void G4StepNPY::relabel(int label)
{

/*
Scintillation and Cerenkov genstep files contain a pre-label of
a signed integer.::   

    In [7]: sts_(1).view(np.int32)[:,0,0]
    Out[7]: array([    1,     2,     3, ..., 13896, 13897, 13898], dtype=int32)

    In [8]: stc_(1).view(np.int32)[:,0,0]
    Out[8]: array([   -1,    -2,    -3, ..., -7834, -7835, -7836], dtype=int32)

Having only 2 types of gensteps is too limiting for example 
when generating test photons corresponding to a light source. 
So *G4StepNPY::relabel* rejigs the markers to a enumerated code.  
The genstep index is still available from the photon buffer, and this is 
written into the *Id* of GPU structs.

*/
    LOG(info)<<"G4StepNPY::relabel" ;
    for(unsigned int i=0 ; i<m_npy->m_ni ; i++ )
    {
        int code = m_npy->getInt(i,0u,0u);
        if(i % 1000 == 0) printf("G4StepNPY::relabel (%u) %d -> %d \n", i, code, label );
        m_npy->setInt(i,0u,0u,0u, label);
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
        //printf(" G4StepNPY::applyLookup  %3u -> %3d  a[%s] b[%s] \n", acode, bcode, aname.c_str(), bname.c_str() );
        assert(aname == bname);
    }

    if( bcode > -1 )
    {
        //unsigned int code = onebased ? bcode + 1 : bcode  ; 
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

    unsigned int ni = m_npy->m_ni ;
    unsigned int nj = m_npy->m_nj ;
    unsigned int nk = m_npy->m_nk ;

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


