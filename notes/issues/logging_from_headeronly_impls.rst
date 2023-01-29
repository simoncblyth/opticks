logging_from_headeronly_impls
=================================

Issue
-------

Is it possible to arrange lightweight logging "slog.h" ?
for use from headeronly impls where cannot use 
the more heavyweight SLOG.hh : because cannot 
initialize static from envvar in C++11 


Heavyweight SLOG.hh/.cc approach relies on static initialization from envvar
-----------------------------------------------------------------------------

::

     14 #include "plog/Severity.h"
     ...
     30 
     31 #include "U4_API_EXPORT.hh"
     32 
     33 struct U4_API U4Physics : public G4VUserPhysicsList
     34 {
     35     static const plog::Severity LEVEL ;


::

    010 #include "SLOG.hh"
     11 const plog::Severity U4Physics::LEVEL = SLOG::EnvLevel("U4Physics", "DEBUG") ;

    182     LOG(LEVEL) << "G4OpAbsorption_DISABLE      : " << G4OpAbsorption_DISABLE ;



Expt with direct plog usage, or lightweight slog.h
--------------------------------------------------------

Or perhaps lower level direct use of plog is possible ?::

   #include <plog/Log.h>
   PLOGI << " hello " ; 

   LOG(plog::info) << " hello " ; 
   LOG(level) << " hello " ; 

How to control logging with static challenged header-onlys ?

* could set object level (not class LEVEL static) at instanciation  


Attempt to do use lite approach in U4Solid.h with U4TreeCreateTest.cc 
has the familiar mis-behaviour of uncontrolled log level. Presumably 
because the header is getting compiled with the main. 



U4Solid.h::

    386 
    387     PLOGI  << "U4Solid::init_Ellipsoid : PLOGI "  ;
    388     LOG(info)  << "U4Solid::init_Ellipsoid : LOG(info) "  ;
    389     LOG(debug) << "U4Solid::init_Ellipsoid : LOG(debug) "  ;
    390     LOG(level) << "U4Solid::init_Ellipsoid : LOG(level) " ;
    391     std::cerr << slog::Desc(level) ;
    392 


Logging without U4Solid envvar set::

    U4Solid::init_Ellipsoid@387: U4Solid::init_Ellipsoid : PLOGI 
    U4Solid::init_Ellipsoid@388: U4Solid::init_Ellipsoid : LOG(info) 
    U4Solid::init_Ellipsoid@389: U4Solid::init_Ellipsoid : LOG(debug) 
    U4Solid::init_Ellipsoid@390: U4Solid::init_Ellipsoid : LOG(level) 
    slog::Desc level:5 plog::severityToString(level):DEBUG

Logging with U4Solid envvar INFO::

    log::envlevel adjusting loglevel by envvar   key U4Solid val INFO fallback DEBUG level INFO severity 4
    U4Solid::init_Ellipsoid@387: U4Solid::init_Ellipsoid : PLOGI 
    U4Solid::init_Ellipsoid@388: U4Solid::init_Ellipsoid : LOG(info) 
    U4Solid::init_Ellipsoid@389: U4Solid::init_Ellipsoid : LOG(debug) 
    U4Solid::init_Ellipsoid@390: U4Solid::init_Ellipsoid : LOG(level) 
    slog::Desc level:4 plog::severityToString(level):INFO


* the level gets set based on the envvar : but there is no logging cut being applied so all logging shows
* familiar issue of logging in main 



