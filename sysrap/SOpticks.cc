#include "SArgs.hh"
#include "SOpticks.hh"


SOpticks::SOpticks(int argc, char** argv, const char* argforced)
   :
   m_sargs(new SArgs(argc, argv, argforced))
{
}

bool SOpticks::hasArg(const char* arg) const
{
    return m_sargs->hasArg(arg);
}

bool SOpticks::isEnabledMergedMesh(unsigned ridx)
{
    return true ; 
}

std::vector<unsigned>&  SOpticks::getSolidSelection() 
{
    return m_solid_selection ; 
}

const std::vector<unsigned>&  SOpticks::getSolidSelection() const 
{
    return m_solid_selection ; 
}

int SOpticks::getRaygenMode() const 
{
    return 0 ; 
}



Composition*  SOpticks::getComposition() const 
{
    return nullptr ; 
}
const char* SOpticks::getOutDir() const 
{
    return nullptr ; 
}







