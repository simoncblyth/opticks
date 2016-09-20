#include "NGLM.hpp"
#include "NPY.hpp"
#include "G4StepNPY.hpp"
#include "TrivialCheckNPY.hpp"

#include "PLOG.hh"

TrivialCheckNPY::TrivialCheckNPY(NPY<float>* photons, NPY<float>* gensteps, char entryCode)
    :
    m_entryCode(entryCode),
    m_photons(photons),
    m_gensteps(gensteps),
    m_g4step(new G4StepNPY(m_gensteps))
{
    assert(m_entryCode == 'T' || m_entryCode == 'D');
}

int TrivialCheckNPY::check(const char* msg)
{
    dump(msg);

    int fail(0);

    checkGensteps(m_gensteps);

    unsigned nstep = m_g4step->getNumSteps();

    unsigned photon_offset(0);
    for(unsigned istep=0 ; istep < nstep ; istep++)
    { 
        unsigned numPhotonsForStep = m_g4step->getNumPhotons(istep);
        unsigned gencode = m_g4step->getGencode(istep);

        fail += checkPhotons(istep, m_photons, photon_offset, photon_offset+numPhotonsForStep, gencode, numPhotonsForStep);

        photon_offset += numPhotonsForStep ;  
    }
    return fail ; 
}

void TrivialCheckNPY::checkGensteps(NPY<float>* gs)
{
    unsigned ni = gs->getShape(0) ;
    unsigned nj = gs->getShape(1) ;
    unsigned nk = gs->getShape(2) ;

    assert(ni > 0 && nj == 6 && nk == 4);

    //m_g4step->dump("TrivialCheckNPY::checkGensteps");
}


int TrivialCheckNPY::checkPhotons(unsigned istep, NPY<float>* photons, unsigned i0, unsigned i1, unsigned gencode, unsigned numPhotons )
{
    LOG(debug) << "TrivialCheckNPY::checkPhotons" 
              << " istep " << istep 
              << " i0 " << i0
              << " i1 " << i1
              << " gencode " << gencode
              << " numPhotons " << numPhotons
              ;

    int fail(0);

    if(m_entryCode == 'T')
    {
        fail += checkItemValue( istep, photons, i0, i1, 2, 0, "(ghead.u.x)gencode"         , IS_UCONSTANT,     gencode, 0 );
        fail += checkItemValue( istep, photons, i0, i1, 2, 3, "(ghead.u.w)numPhotons"      , IS_UCONSTANT,  numPhotons, 0 );
    }

    unsigned PNUMQUAD = 4 ; 
    unsigned GNUMQUAD = 6 ; 

 
    if(m_entryCode == 'T' || m_entryCode == 'D')
    {
        fail += checkItemValue( istep, photons, i0, i1, 3, 0, "(indices.u.x)photon_id"     , IS_UINDEX,              -1,        0 );
        fail += checkItemValue( istep, photons, i0, i1, 3, 1, "(indices.u.y)photon_offset" , IS_UINDEX_SCALED,       -1, PNUMQUAD );
        fail += checkItemValue( istep, photons, i0, i1, 3, 2, "(indices.u.z)genstep_id"    , IS_UCONSTANT,        istep,        0 );
        fail += checkItemValue( istep, photons, i0, i1, 3, 3, "(indices.u.w)genstep_offset", IS_UCONSTANT_SCALED, istep, GNUMQUAD );
    }

    return fail ; 
}

void TrivialCheckNPY::dump(const char* msg)
{
    LOG(info) << msg 
              << " entryCode " << m_entryCode
              << " photons " << m_photons->getShapeString()
              << " gensteps " << m_gensteps->getShapeString()
              ;

}


int TrivialCheckNPY::checkItemValue(unsigned istep, NPY<float>* npy, unsigned i0, unsigned i1, unsigned jj, unsigned kk, const char* label, int expect, int constant, int scale )
{
    unsigned uconstant(constant);

    unsigned ni = npy->getShape(0) ;
    unsigned nj = npy->getShape(1) ;
    unsigned nk = npy->getShape(2) ;

    assert(i0 < i1 && i0 < ni && i1 <= ni );
    assert(nj == 4 && nk == 4);  // photon buffer

    float* values = npy->getValues();

    uif_t uif ; 

    int fail(0);
    for(unsigned i=i0 ; i<i1 ; i++ )
    {
        unsigned index = i*nj*nk + jj*nk + kk ;
        uif.f = values[index];

        unsigned u = uif.u ;   

        //LOG(info) << " i " << std::setw(6) << i << " u " << std::setw(6) << u ;  

        if(     expect == IS_UINDEX)    
        {
            if(u != i )
            {
                if(fail < 10)
                LOG(warning) << "FAIL checkItemValue IS_UINDEX "
                             << " istep:" << istep 
                             << " label:" << label 
                             << " i:" << i 
                             << " u:" << u 
                             ;
                fail += 1 ;   
            }
        }
        else if(expect == IS_UINDEX_SCALED)
        {
            if(u != i*scale )
            {
                if(fail < 10)
                LOG(warning) << "FAIL checkItemValue IS_UINDEX_SCALED " 
                             << " istep:" << istep 
                             << " label:" << label 
                             << " i:" << i 
                             << " u:" << u 
                             << " i*s:" << i*scale
                             ;

                fail += 1 ;   
            }
        }
        else if(expect == IS_UCONSTANT)  
        {
            if(u != uconstant )
            {
                if(fail < 10)
                LOG(warning) << "FAIL checkItemValue IS_UCONSTANT " 
                             << " istep:" << istep 
                             << " label:" << label 
                             << " i:" << i 
                             << " u:" << u 
                             << " uconstant:" << uconstant
                             ; 
                fail += 1 ;   
            }
        }
        else if(expect == IS_UCONSTANT_SCALED)  
        {
            if(u != uconstant*scale )
            {
                if(fail < 10)
                LOG(warning) << "FAIL checkItemValue IS_UCONSTANT_SCALED " 
                             << " istep:" << istep 
                             << " label:" << label 
                             << " i:" << i 
                             << " u:" << u 
                             << " uconstant*s:" << uconstant*scale
                             ;

                fail += 1 ;   
            }
        }


    }

    if(fail == 0)
    {
        LOG(debug) 
            << " step " << istep
            << "[:," << jj << "," << kk << "] " 
            << std::setw(30) << label 
            << ( fail == 0 ? " OK " : " FAIL " ) << fail  ; 
    }
    else
    {
        LOG(fatal) 
            << " step " << istep
            << "[:," << jj << "," << kk << "] " 
            << std::setw(30) << label 
            << ( fail == 0 ? " OK " : " FAIL " ) << fail  ; 
    }


    return fail ; 
}


