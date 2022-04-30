#include "SArgs.hh"
#include "STime.hh"
#include "SStr.hh"
#include "SProc.hh"
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




/**
SOpticks::WriteOutputDirScript
------------------------------

When an output directory is determined by an executable it 
is useful to write a small script with an export line showing 
the output directory such that scripts and executables that are subsequently 
run can access the output without having pre-knowledge of the directory.  

**/

void SOpticks::WriteOutputDirScript(const char* outdir) // static
{
    const char* exename = SProc::ExecutableName() ;
    const char* envvar = SStr::Concat(exename,  "_OUTPUT_DIR" ); 
    const char* sh_path = SStr::Concat(exename, "_OUTPUT_DIR" , ".sh")   ;    

    std::stringstream ss ; 
    ss   
        << "# Opticks::writeOutputDirScript " << std::endl 
        << "# " << STime::Stamp() << std::endl 
        << std::endl 
        << "export " << envvar << "=" << outdir  
        << std::endl 
        ;    
  
    std::string sh = ss.str(); 

    LOG(info) 
        << "writing sh_path " << sh_path << std::endl
        << "sh [" << std::endl
        << sh   
        << "]"  
        ;    

    int create_dirs = 0 ;  
    SStr::Save(sh_path, sh.c_str(), create_dirs ) ;  
}




