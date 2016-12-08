#include <cassert>
#include <iostream>
#include <iomanip>



#include "uif.h"
#include "NPY.hpp"
#include "NLookup.hpp"
#include "G4StepNPY.hpp"
#include "NPY.hpp"


#include "PLOG.hh"

G4StepNPY::G4StepNPY(NPY<float>* npy) 
       :  
       m_npy(npy),
       m_lookup(NULL),
       m_total_photons(0)
{
}

NPY<float>* G4StepNPY::getNPY()
{
    return m_npy ; 
}
void G4StepNPY::setLookup(NLookup* lookup)
{
    m_lookup = lookup ;
} 
NLookup* G4StepNPY::getLookup()
{
    return m_lookup ;
} 


unsigned G4StepNPY::getNumSteps()
{
    return m_npy->getShape(0);
}
unsigned G4StepNPY::getNumPhotons(unsigned i)
{
    unsigned ni = getNumSteps();
    assert(i < ni);
    int numPhotons = m_npy->getInt(i,0u,3u);
    return numPhotons ; 
}
unsigned G4StepNPY::getGencode(unsigned i)
{
    unsigned ni = getNumSteps();
    assert(i < ni);
    unsigned gencode = m_npy->getInt(i,0u,0u);
    return gencode  ; 
}


unsigned G4StepNPY::getNumPhotonsTotal()
{
    unsigned ni = getNumSteps();
    unsigned total(0);
    for(unsigned i=0 ; i<ni ; i++) total+= getNumPhotons(i) ;
    return total ; 
}

unsigned* G4StepNPY::makePhotonSeedArray()
{
    unsigned nstep = getNumSteps();
    unsigned nseed = getNumPhotonsTotal();
    unsigned* seeds = new unsigned[nseed] ; 
    unsigned offset(0);
    for(unsigned s=0 ; s < nstep ; s++)
    {
        unsigned npho = getNumPhotons(s);
        for(unsigned ipho=0 ; ipho<npho ; ipho++) seeds[offset + ipho] = s ; 
        offset += npho ; 
    } 
    return seeds ; 
   // genstep id for each photon
}

 

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

void G4StepNPY::countPhotons()
{
    for(unsigned int i=0 ; i<m_npy->m_ni ; i++ )
    {
        int label      = m_npy->getInt(i,0u,0u);
        int numPhotons = m_npy->getInt(i,0u,3u);
        if(m_photons.count(label) == 0) m_photons[label] = 0 ;
        m_photons[label] += numPhotons ; 
        m_total_photons += numPhotons ; 
    }
}


void G4StepNPY::checkCounts(std::vector<int>& counts, const char* msg)
{
    LOG(info) << msg 
              << " compare *seqCounts* (actual photon counts from propagation sequence data SeqNPY ) "
              << " with *stepCounts* (expected photon counts from input G4StepNPY )  "
              ;
    assert(counts.size() == 16);

    int mismatch(0);
    for(unsigned i=0 ; i < counts.size() ; i++)
    {
         int label = i == 0 ? 0 : 0x1 << (i - 1) ;  // layout of OpticksPhoton flags  

         int count = counts[i] ;
         int xpect = getNumPhotons(label) ; 
         if(count != xpect) mismatch += 1 ; 

         std::cout 
              << " bpos(hex) " << std::setw(10) << std::hex << i  << std::dec
              << " seqCounts " << std::setw(10) << count
              << " flagLabel " <<  std::setw(10) << label
              << " stepCounts " << std::setw(10) << xpect 
              << std::endl ; 
    }

    if(mismatch > 0)
          LOG(fatal) << "G4StepNPY::checkCounts MISMATCH between steps and propagation photon counts  "
                     << " mismatch " << mismatch
                     ; 

    assert(mismatch==0);
}




std::string G4StepNPY::description()
{
    std::stringstream ss ; 
    int total(0) ; 
    for(std::map<int,int>::const_iterator it=m_photons.begin() ; it != m_photons.end() ; it++)
    {
        int label = it->first ; 
        int numPhotons = it->second ; 
        total += numPhotons ; 
        ss   << " [ "
             << std::setw(10) << label 
             << std::setw(10) << numPhotons
             << " ] "
             ;
    }

    assert(total == m_total_photons);
    ss       << " [ "
             << std::setw(10) << "total"
             << std::setw(10) << m_total_photons
             << " ] " 
             ;

    return ss.str();
}

void G4StepNPY::Summary(const char* msg)
{
    LOG(info) << msg << description() ; 

}

int G4StepNPY::getNumPhotonsCounted(int label)
{
    return m_photons.count(label) == 0 ? 0 : m_photons[label] ; 
}
int G4StepNPY::getNumPhotonsCounted()
{
    return m_total_photons ; 
}


void G4StepNPY::checklabel(int xlabel, int ylabel)
{
    unsigned numStep = m_npy->getNumItems();
    unsigned mismatch = 0 ;  

    for(unsigned int i=0 ; i<numStep ; i++ )
    {
        int label = m_npy->getInt(i,0u,0u);

        if(xlabel > -1 && ylabel > -1)
        {
            if(xlabel == label || ylabel == label )
                 continue ;
            else 
                 mismatch += 1 ;  
        }
        else if(xlabel > -1 )
        {
            if(xlabel == label )
                 continue ;
            else 
                 mismatch += 1 ;  
        } 
        else if(ylabel > -1 )
        {
            if(ylabel == label )
                 continue ;
            else 
                 mismatch += 1 ;  
        } 
    }

    if(mismatch > 0) 
         LOG(fatal)<<"G4StepNPY::checklabel FAIL" 
                   << " xlabel " << xlabel 
                   << " ylabel " << ylabel 
                   << " numStep " << numStep
                   << " mismatch " << mismatch ; 
                   ;
  
    assert(mismatch == 0 );
}



void G4StepNPY::relabel(int cerenkov_label, int scintillation_label)
{

/*
Scintillation and Cerenkov genstep files contain a pre-label of
a signed 1-based integer index ::   

    In [5]: a = np.load(os.path.expanduser("/Users/blyth/opticksdata/gensteps/dayabay/cerenkov/1.npy"))
    In [8]: a[:,0,0].view(np.int32)
    Out[8]: array([   -1,    -2,    -3, ..., -7834, -7835, -7836], dtype=int32)

    In [9]: b = np.load(os.path.expanduser("/Users/blyth/opticksdata/gensteps/dayabay/scintillation/1.npy"))
    In [11]: b[:,0,0].view(np.int32)
    Out[11]: array([    1,     2,     3, ..., 13896, 13897, 13898], dtype=int32)


Having only 2 types of gensteps, indicated by +ve and -ve indices, 
is too limiting for example when generating test photons corresponding to a light source. 
So *G4StepNPY::relabel* rejigs the markers to the OpticksPhoton.h enumerated code of the source.  
The genstep index is still available from the photon buffer, and this is 
written into the *Id* of GPU structs.



G4gun gensteps look to already have been relabled 


In [8]: np.count_nonzero(evt.gs[:,0,0].view(np.int32) == 1)
Out[8]: 5080

In [9]: np.count_nonzero(evt.gs[:,0,0].view(np.int32) == 2)
Out[9]: 339530


simon:opticksnpy blyth$ opticks-find relabel
./optickscore/OpticksEvent.cc:    m_g4step->relabel(CERENKOV, SCINTILLATION);    // 1, 2 
./opticksnpy/G4StepNPY.cpp:void G4StepNPY::relabel(int cerenkov_label, int scintillation_label)
./opticksnpy/G4StepNPY.cpp:So *G4StepNPY::relabel* rejigs the markers to the OpticksPhoton.h enumerated code of the source.  
./opticksnpy/G4StepNPY.cpp:    LOG(info)<<"G4StepNPY::relabel" ;
./opticksnpy/G4StepNPY.cpp:        if(i % 1000 == 0) printf("G4StepNPY::relabel (%u) %d -> %d \n", i, code, label );
./opticksnpy/G4StepNPY.hpp:       void relabel(int cerenkov_label, int scintillation_label);




*/
    LOG(info)<<"G4StepNPY::relabel" ;
    for(unsigned int i=0 ; i<m_npy->m_ni ; i++ )
    {
        int code = m_npy->getInt(i,0u,0u);

        int label = code < 0 ? cerenkov_label : scintillation_label ; 

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
        printf(" G4StepNPY::applyLookup  %3u -> %3d  a[%s] b[%s] \n", acode, bcode, aname.c_str(), bname.c_str() );
        assert(aname == bname);
    }

    if( bcode > -1 )
    {
        //unsigned int code = onebased ? bcode + 1 : bcode  ; 
        uif.u = bcode ; 
        data[index] = uif.f ;
        m_lines.insert(bcode);

        if(m_lookup_ok.count(acode) == 0) m_lookup_ok[acode] = 0 ; 
        m_lookup_ok[acode]++ ; 
    }
    else
    {
        std::string aname = m_lookup->acode2name(acode) ;
        printf("G4StepNPY::applyLookup failed to translate acode %u : %s \n", acode, aname.c_str() );

        if(m_lookup_fails.count(acode) == 0) m_lookup_fails[acode] = 0 ; 
        m_lookup_fails[acode]++ ; 

    }

    return bcode > -1 ; 
}


void G4StepNPY::dumpLookupFails(const char* msg)
{
    LOG(info) << msg 
              << " lookup_fails " << m_lookup_fails.size()
              << " lookup_ok " << m_lookup_ok.size()
              ;

    typedef std::map<int,int> MII ;
    for(MII::const_iterator it=m_lookup_fails.begin() ; it != m_lookup_fails.end() ; it++)
    {
         std::cout << " acode " <<  std::setw(10) << it->first 
                   << " lookup_fails " << std::setw(10) << it->second
                   << std::endl 
                   ;

    }

    for(MII::const_iterator it=m_lookup_ok.begin() ; it != m_lookup_ok.end() ; it++)
    {
         std::cout << " acode " <<  std::setw(10) << it->first 
                   << " lookup_ok " << std::setw(10) << it->second
                   << std::endl 
                   ;

    }

}




int G4StepNPY::getStepId(unsigned int i)
{
    return m_npy->getInt(i,0,0);
}

/*

no longer valid as using the enum codes on collection

bool G4StepNPY::isCerenkovStep(unsigned int i)  
{
    return getStepId(i) < 0 ; 
}
bool G4StepNPY::isScintillationStep(unsigned int i)
{
    return getStepId(i) > 0 ; 
}
*/



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

    if(nfail > 0)
    {
       LOG(fatal) << "G4StepNPY::applyLookup"
                  << " shape " << m_npy->getShapeString()
                  << " nfail " << nfail 
                  ;
       NPYBase::setGlobalVerbose(true);
       m_npy->save("$TMP/G4StepNPY_applyLookup_FAIL.npy");

       dumpLookupFails("G4StepNPY::applyLookup");

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


