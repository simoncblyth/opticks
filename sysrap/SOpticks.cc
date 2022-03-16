#include "SArgs.hh"
#include "STime.hh"
#include "SStr.hh"
#include "SOpticks.hh"

#include "PLOG.hh"

const plog::Severity SOpticks::LEVEL = PLOG::EnvLevel("SOpticks", "DEBUG"); 


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


const char* SOpticks::CFBaseScriptPath() 
{
    std::stringstream ss ; 
    ss << "/tmp/CFBASE.sh" ; 
    std::string s = ss.str(); 
    return strdup(s.c_str()); 
}

std::string SOpticks::CFBaseScriptString(const char* cfbase, const char* msg )
{
    std::stringstream ss ;
    ss
        << "# " << msg
        << std::endl
        << "# " << STime::Stamp()
        << std::endl
        << "export CFBASE=" << cfbase
        << std::endl
        << "cfcd(){ cd " << cfbase << "/CSGFoundry ; pwd ; } "    // dont assume the envvar still same when function used
        << std::endl
        << "# "
        << std::endl
        ;

    std::string s = ss.str();
    return s ;
}


void SOpticks::WriteCFBaseScript(const char* cfbase, const char* msg) 
{
    const char* sh_path = CFBaseScriptPath() ;
    std::string sh = CFBaseScriptString(cfbase, msg);

    LOG(info)
        << "writing sh_path " << sh_path << std::endl
        << "sh [" << std::endl
        << sh
        << "]"
        ;

    SStr::Save(sh_path, sh.c_str()) ;
}








