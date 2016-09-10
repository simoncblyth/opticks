#include "NGLM.hpp"
#include "NPY.hpp"
#include "TrivialCheckNPY.hpp"

#include "PLOG.hh"

TrivialCheckNPY::TrivialCheckNPY(NPY<float>* photons, NPY<float>* gensteps)
    :
    m_photons(photons),
    m_gensteps(gensteps)
{
}

void TrivialCheckNPY::dump(const char* msg)
{
    LOG(info) << msg ; 
}

