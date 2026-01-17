#include <iostream>
#include <iomanip>
#include <sstream>
#include <bitset>
#include <algorithm>
#include <cstring>

#include "ssys.h"
#include "sstr.h"

#include "SBit.hh"   // TODO: sbit.h
#include "SBitSet.h"

#include "SGeoConfig.hh"
#include "SName.h"
#include "SLabel.h"

#include "SLOG.hh"

const plog::Severity SGeoConfig::LEVEL = SLOG::EnvLevel("SGeoConfig", "DEBUG");


const char* SGeoConfig::_GEOM = ssys::getenvvar(kGEOM, nullptr );
unsigned long long SGeoConfig::_EMM = SBit::FromEString(kEMM, "~0");
const char* SGeoConfig::_ELVSelection   = ssys::getenvvar(kELVSelection, nullptr );
const char* SGeoConfig::_SolidSelection = ssys::getenvvar(kSolidSelection, nullptr );
const char* SGeoConfig::_SolidTrimesh   = ssys::getenvvar(kSolidTrimesh, nullptr );
const char* SGeoConfig::_FlightConfig   = ssys::getenvvar(kFlightConfig  , nullptr );
const char* SGeoConfig::_ArglistPath    = ssys::getenvvar(kArglistPath  , nullptr );
const char* SGeoConfig::_CXSkipLV       = ssys::getenvvar(kCXSkipLV  , nullptr );
const char* SGeoConfig::_CXSkipLV_IDXList = ssys::getenvvar(kCXSkipLV_IDXList, nullptr );

void SGeoConfig::SetELVSelection(  const char* es){  _ELVSelection   = es ? strdup(es) : nullptr ; }
void SGeoConfig::SetSolidSelection(const char* ss){  _SolidSelection = ss ? strdup(ss) : nullptr ; }
void SGeoConfig::SetSolidTrimesh(  const char* st){  _SolidTrimesh   = st ? strdup(st) : nullptr ; }
void SGeoConfig::SetFlightConfig(  const char* fc){  _FlightConfig   = fc ? strdup(fc) : nullptr ; }
void SGeoConfig::SetArglistPath(   const char* ap){  _ArglistPath    = ap ? strdup(ap) : nullptr ; }
void SGeoConfig::SetCXSkipLV(      const char* cx){  _CXSkipLV       = cx ? strdup(cx) : nullptr ; }

unsigned long long SGeoConfig::EnabledMergedMesh(){  return _EMM ; }
const char* SGeoConfig::SolidSelection(){ return _SolidSelection ; }
const char* SGeoConfig::SolidTrimesh(){   return _SolidTrimesh ; }
const char* SGeoConfig::FlightConfig(){   return _FlightConfig ; }
const char* SGeoConfig::ArglistPath(){    return _ArglistPath ; }
const char* SGeoConfig::CXSkipLV(){       return _CXSkipLV ? _CXSkipLV : "" ; }
const char* SGeoConfig::CXSkipLV_IDXList(){  return _CXSkipLV_IDXList ? _CXSkipLV_IDXList : "" ; }


const char* SGeoConfig::GEOM(){           return _GEOM ; }
const char* SGeoConfig::ELVSelection(){   return _ELVSelection ; }


/**
SGeoConfig::ELVSelection
--------------------------

SGeoConfig::ELVSelection converts a comma delimited list of lv solid names optionally
with special prefixes "t:" or "filepath:" into a comma delimited list of lvid integers.


Examples using "filepath:" prefix to select volumes via solid names(aka meshnames) listed in files::

    GEOM cf
    cp meshname.txt /tmp/elv.txt
    vi /tmp/elv.txt
    ELV=filepath:/tmp/elv.txt MOI=sTarget:0:-1 ~/o/cx.sh

    grep Surftube meshname.txt > /tmp/elv.txt
    echo sTarget >> /tmp/elv.txt
    ELV=filepath:/tmp/elv.txt MOI=sTarget:0:-1 ~/o/cx.sh

Note that the file is read via sstr::SplitTrimSuppress which will ignore lines in the file
starting with "#", so another way::

    GEOM cf
    cp meshname.txt /tmp/elv.txt
    vi /tmp/elv.txt ## comment names to skip with "#"
    ELV=filepath:/tmp/elv.txt MOI=sTarget:0:-1 ~/o/cx.sh

    grep \# /tmp/elv.txt
    #HamamatsuR12860sMask_virtual
    #NNVTMCPPMTsMask_virtual
    #mask_PMT_20inch_vetosMask_virtual


Examples using "t:" prefix to exclude volumes::

   ELV=t:HamamatsuR12860sMask_virtual,NNVTMCPPMTsMask_virtual MOI=NNVTMCPPMTsMask:0:-2  ~/o/cx.sh

   DISPLAY=:1 ELV=t:HamamatsuR12860sMask_virtual,NNVTMCPPMTsMask_virtual MOI=NNVTMCPPMTsMask:0:-2  ~/o/cx.sh



**/

const char* SGeoConfig::ELVSelection(const SName* id )
{
    const char* elv_selection_ = ELVSelection() ;
    const char* elv = nullptr ;
    char delim = ',' ;
    bool VERBOSE = ssys::getenvbool(ELVSelection_VERBOSE);

    if(VERBOSE) std::cerr
        << "SGeoConfig::ELVSelection.0."
        << " [" << ELVSelection_VERBOSE << "] "
        << " elv_selection_ " << ( elv_selection_ ? elv_selection_ : "-" )
        << std::endl
        ;



    bool allow_missing_names = true ;

    if( elv_selection_ )
    {
        const char* prefix = ELVPrefix(elv_selection_);

        if(VERBOSE) std::cerr
            << "SGeoConfig::ELVSelection.1."
            << " prefix " << ( prefix ? prefix : "-" )
            << " strlen(prefix) " << ( prefix ? strlen(prefix) : 0 )
            << "\n"
            ;


        if( SName::Has_STARTING( elv_selection_))  // skip the hasNames check when using STARTING_
        {
            std::vector<std::string>* qq_missing = nullptr ;
            elv = id->getIDXListFromNames(elv_selection_, delim, prefix, qq_missing );
        }
        else
        {
            if( allow_missing_names )
            {
                std::vector<std::string>* qq_missing = new std::vector<std::string> ;
                elv = id->getIDXListFromNames(elv_selection_, delim, prefix, qq_missing );

                int num_missing_names = qq_missing ? qq_missing->size() : -1 ;

                if(VERBOSE) std::cerr
                    << "SGeoConfig::ELVSelection.5."
                    << " elv_selection_[" << elv_selection_ << "]"
                    << " allow_missing_names " << ( allow_missing_names ? "YES" : "NO " )
                    << " num_missing_names " << num_missing_names
                    << "\n"
                    ;

                 if(num_missing_names > 0) for(int i=0 ; i < num_missing_names ; i++ ) std::cerr
                    << "SGeoConfig::ELVSelection.6. missing_name[" << (*qq_missing)[i] << "]\n" ;
            }
            else
            {
                std::vector<std::string>* qq_missing = nullptr ;
                std::stringstream ss ;
                bool has_names = id->hasNames(elv_selection_, delim, prefix, qq_missing, &ss );

                if(VERBOSE) std::cerr
                    << "SGeoConfig::ELVSelection.2."
                    << " elv_selection_[" << elv_selection_ << "]"
                    << " has_names " << ( has_names ? "YES" : "NO " )
                    << " allow_missing_names " << ( allow_missing_names ? "YES" : "NO " )
                    << " qq_missing.size " << ( qq_missing ? qq_missing->size() : -1 )
                    << "\n"
                    ;

                if(!has_names) std::cout
                    << "SGeoConfig::ELVSelection.3."
                    << " has_names " << ( has_names ? "YES" : "NO " ) << "\n"
                    << " qq_missing.size " << ( qq_missing ? qq_missing->size() : -1 )
                    << "[haslog[\n"
                    << ss.str()
                    << "]haslog[\n"
                    << "[id.detail\n"
                    << id->detail()
                    << "]id.detail\n"
                    ;

                if(has_names)
                {
                    elv = id->getIDXListFromNames(elv_selection_, delim, prefix, qq_missing );
                }
                else
                {
                    elv = elv_selection_ ;  // eg when just numbers
                }
            }
       }
    }
    return elv ;
}


/**
SGeoConfig::ELVPrefix
-----------------------

As the above ELVSelection has a problem for solids
with names starting with "t" are starting to transition
to separate the modifier from the list of ints or solid names.

DONE : adopt "t:" prefix for the tilde modifier.

*/

const char* SGeoConfig::ELVPrefix(const char* elvarg)
{
    if(elvarg == nullptr) return nullptr ;
    if(strlen(elvarg) == 0) return nullptr ;

    std::string prefix ;
    if(sstr::StartsWith(elvarg, "t:") || sstr::StartsWith(elvarg, "~:"))
    {
        prefix.assign("t:") ;
    }
    else if( sstr::StartsWith(elvarg,"t" ) || sstr::StartsWith(elvarg,"~") )
    {
        prefix.assign("t") ;
    }
    return prefix.empty() ? nullptr : strdup(prefix.c_str()) ;
}



/**
SGeoConfig::ELVString (formerly CSGFoundry::ELVString)
-------------------------------------------------------

This is used from SGeoConfig::ELV to create the SBitSet of included/excluded LV.

String configuring dynamic shape selection of form : t:110,117,134 or null when
there is no selection.  The value is obtained from:

* SGeoConfig::ELVSelection() which defaults to the OPTICKS_ELV_SELECTION envvar value
  and can be changed by the SGeoConfig::SetELVSelection static, a comma delimited list of
  mesh names is expected, for example:
  "NNVTMCPPMTsMask_virtual0x,HamamatsuR12860sMask_virtual0x,mask_PMT_20inch_vetosMask_virtual0x"

  If any of the names are not found in the geometry the selection request is ignored.

**/

const char* SGeoConfig::ELVString(const SName* id)
{
    const char* elv = ELVSelection(id) ;
    LOG(LEVEL) << " elv " << ( elv ? elv : "-" ) ;
    return elv ;
}


/**
SGeoConfig::ELV (formerly CSGFoundry::ELV)
--------------------------------------------

Sensitive to OPTICKS_ELV_SELECTION

**/

const SBitSet* SGeoConfig::ELV(const SName* id)
{
    unsigned num_meshname = id->getNumName();
    const char* elv_ = ELVString(id);
    SBitSet* elv = elv_ ? SBitSet::Create(num_meshname, elv_ ) : nullptr ;

    LOG(LEVEL)
       << " num_meshname " << num_meshname
       << " elv_ " << ( elv_ ? elv_ : "-" )
       << " elv " << ( elv ? elv->desc() : "-" )
       ;

    return elv ;
}











std::string SGeoConfig::Desc()
{
    std::stringstream ss ;
    ss << std::endl ;
    ss << std::setw(25) << kGEOM             << " : " << ( _GEOM   ? _GEOM   : "-" ) << std::endl ;
    ss << std::setw(25) << kEMM              << " : " << SBit::HexString(_EMM) << " 0x" << std::hex << _EMM << std::dec << std::endl ;
    ss << std::setw(25) << kELVSelection     << " : " << ( _ELVSelection   ? _ELVSelection   : "-" ) << std::endl ;
    ss << std::setw(25) << kSolidSelection   << " : " << ( _SolidSelection ? _SolidSelection : "-" ) << std::endl ;
    ss << std::setw(25) << kSolidTrimesh     << " : " << ( _SolidTrimesh   ? _SolidTrimesh   : "-" ) << std::endl ;
    ss << std::setw(25) << kFlightConfig     << " : " << ( _FlightConfig   ? _FlightConfig   : "-" ) << std::endl ;
    ss << std::setw(25) << kArglistPath      << " : " << ( _ArglistPath    ? _ArglistPath    : "-" ) << std::endl ;
    ss << std::setw(25) << kCXSkipLV         << " : " << CXSkipLV() << std::endl ;
    ss << std::setw(25) << kCXSkipLV_IDXList << " : " << CXSkipLV_IDXList() << std::endl ;
    std::string str = ss.str();
    return str ;
}

std::string SGeoConfig::desc() const
{
    std::stringstream ss ;
    ss << "SGeoConfig::desc\n" ;
    std::string str = ss.str();
    return str ;
}



bool SGeoConfig::IsEnabledMergedMesh(unsigned mm) // static
{
    bool emptylistdefault = true ;
    bool emm = true ;
    if(mm < 64)
    {
        std::bitset<64> bs(_EMM);
        emm = bs.count() == 0 ? emptylistdefault : bs[mm] ;
    }
    return emm ;
}


std::string SGeoConfig::DescEMM()
{
    std::stringstream ss ;
    ss << "SGeoConfig::DescEMM " ;
    ss << std::setw(25) << kEMM              << " : " << SBit::HexString(_EMM) << " 0x" << std::hex << _EMM << std::dec << std::endl ;

    for(unsigned i=0 ; i < 64 ; i++)
    {
        bool emm = SGeoConfig::IsEnabledMergedMesh(i) ;
        if(emm) ss << i << " " ;
    }
    std::string s = ss.str();
    return s ;
}


std::vector<std::string>*  SGeoConfig::Arglist()
{
    return sstr::LoadList( _ArglistPath, '\n' );
}



/**
SGeoConfig::CXSkipLV_IDXList
-----------------------------

Translates names in a comma delimited list into indices according to SName.

**/
void SGeoConfig::SetCXSkipLV_IDXList(const SName* id)
{
    const char* cxskiplv_ = CXSkipLV() ;
    bool has_names = cxskiplv_ ? id->hasNames(cxskiplv_ ) : false ;
    _CXSkipLV_IDXList = has_names ? id->getIDXListFromNames(cxskiplv_, ',' ) : nullptr ;
}

/**
SGeoConfig::IsCXSkipLV
------------------------

This controls mesh/solid skipping during GGeo to CSGFoundry
translation as this is called from:

1. CSG_GGeo_Convert::CountSolidPrim
2. CSG_GGeo_Convert::convertSolid

For any skips to be applied the below SGeoConfig::GeometrySpecificSetup
must have been called.

For example this is used for long term skipping of Water///Water
virtual solids that are only there for Geant4 performance reasons,
and do nothing useful for Opticks.

Note that ELVSelection does something similar to this, but
that is applied at every CSGFoundry::Load providing dynamic prim selection.
As maintaining consistency between results and geometry is problematic
with dynamic prim selection it is best to only use the dynamic approach
for geometry render scanning to find bottlenecks.

When creating longer lived geometry for analysis with multiple executables
it is more appropriate to use CXSkipLV to effect skipping at translation.

**/

bool SGeoConfig::IsCXSkipLV(int lv) // static
{
    if( _CXSkipLV_IDXList == nullptr ) return false ;
    std::vector<int> cxskip ;
    sstr::split<int>(cxskip, _CXSkipLV_IDXList, ',' );

    return std::count( cxskip.begin(), cxskip.end(), lv ) == 1 ;
}


/**
SGeoConfig::GeometrySpecificSetup
-----------------------------------

TODO: compare GPU performance with and without these virtual skips

This is invoked from:

1. [DEAD CODE LOCATION] CSG_GGeo_Convert::init prior to GGeo to CSGFoundry translation
2. [NOT MAINLINE CODE, ONLY TESTING/VIZ] argumentless CSGFoundry::Load

The SName* id argument passes the meshnames (aka solid names)
allowing detection of where a geometry appears to be JUNO by
the presence of a collection of solid names within it.
If JUNO is detected some JUNO specific static method calls are made.
This avoids repeating these settings in tests or fiddling
with envvars to configure these things.

Previously did something similar using metadata in geocache
or from the Opticks setup code within detector specific code.
However do not want to require writing cache and prefer to minimize
detector specific Opticks setup code as it is much easier
to test in isolation than as an appendage to a detector framework.

**AVOID "0x" SUFFIXES IN DECISION STRINGS**

The "0x" address suffix on names are only present when running
from a Geant4 geometry that was booted from GDML (assuming sane naming).
Hence including "0x" in decision strings such as skip lists would
cause different geometry when running from GDML and when running
live which is to be avoided.

**/
void SGeoConfig::GeometrySpecificSetup(const SName* id)  // static
{
    bool _JUNO_Detected = JUNO_Detected(id) ;
    LOG(LEVEL) << " _JUNO_Detected " << _JUNO_Detected ;
    if(_JUNO_Detected)
    {
        //const char* skips = "NNVTMCPPMTsMask_virtual,HamamatsuR12860sMask_virtual,mask_PMT_20inch_vetosMask_virtual" ;
        const char* skips = nullptr ;

        SetCXSkipLV(skips);
        SetCXSkipLV_IDXList(id);

        // USING dynamic ELVSelection here would be inappropriate : as dynamic selection
        // means the persisted geometry does not match the used geometry.
    }
}

bool SGeoConfig::JUNO_Detected(const SName* id)
{
    const char* JUNO_names = "HamamatsuR12860sMask,HamamatsuR12860_PMT_20inch,NNVTMCPPMT_PMT_20inch" ;
    bool with_JUNO_names = id ? id->hasNames(JUNO_names) : false ;
    return with_JUNO_names ;
}


