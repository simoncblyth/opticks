LHCb_RICH_YunlongLi
======================




> Hi Simon,
>
> Thank you very much for your reply. 
>
> Since the gdml and log files are too large to deliver as attachments, I
> uploaded the gdml file of the RICH detector and the log files to bitbucket
> ( https://bitbucket.org/yl-li/opticks_lhcb_rich/src/master/), so that you can
> reproduce the problems I mentioned before.
> ...
> I executed the command: 
>
>
>     OKX4Test --deletegeocache \
>              --gdmlpath ~/liyu/geometry/rich1_new.gdml \
>              --cvd 1 --rtx 1 \
>              --envkey --xanalytic \
>              --timemax 400 --animtimemax 400 \
>              --target 1 --eye -1,-1,-1 \
>              --SYSRAP debug
>
>
> and from the OKX4Test_SAbbrev.log (as attached to this email) in the last few
> lines, you can see [PipeBeTV56] and [PipeTitaniumG5] give the same abbrievation
> as [P5]. And after changing the code a bit as I wrote before, this problem can
> be solved.


Regarding your commandline notice that controlling the logging level 
at the project level with "--SYSRAP debug" tends to yield huge amounts of output.
Instead of doing that you can control the logging level for each class/struct 
by setting envvars named after the class/struct.  For example::

    export SAbbrev=INFO

Also note that the option "--cvd 1" is internally setting CUDA_VISIBLE_DEVICES envvar 
to 1 which will only work if you have more than one GPU attached.  
In this case with OKX4Test that does not matter, as the GPU is not being used, 
but in other cases using an inappropriate "--cvd" will cause crashes.  

Regarding the SAbbrev assert I added SAbbrevTest.cc test_3 for the issue::

void test_3()
{
     SAbbrev::FromString(R"LITERAL(
Copper
PipeAl6061
C4F10
PipeAl2219F
VeloStainlessSteel
Vacuum
PipeBeTV56
PipeSteel316LN
PipeBe
Celazole
PipeTitaniumG5
AW7075
PipeAl6082
FutureFibre
Technora
Brass
PipeSteel
BakeOutAerogel
Rich2CarbonFibre
RichSoftIron
Rich1GasWindowQuartz
Kovar
HpdIndium
HpdWindowQuartz
HpdS20PhCathode
HpdChromium
HpdKapton
Supra36Hpd
RichHpdSilicon
RichHpdVacuum
Rich1Nitrogen
Rich1MirrorCarbonFibre
R1RadiatorGas
Rich1MirrorGlassSimex
Rich1Mirror2SupportMaterial
Rich1G10
Rich1PMI
Rich1DiaphramMaterial
Air
)LITERAL")->dump() ; 

} 

And I fixed the issue by using random abbreviations when the usual attempts 
to abbreviate failed to come up with something unique.  




> 2. In this file only the polished, polished front-painted and ground mirrors
> are considered, other cases will cause the assertion in line 239 failed. Are
> you planning to handle other types of mirrors?
>
>
>   I have no plan to implement more surface types until I need them.
>
>   I am very willing to incorporate your pull requests with more surface types added.
>   However I suggest you discuss with me how you plan to do that first to ensure your
>   work can be incorporated into Opticks.
>
> The main reason why I asked this problem is that in this gdml file, there are some ground frontpainted mirrors (type 4), which can cause the command
>
> OKX4Test --deletegeocache --gdmlpath ~/liyu/geometry/rich1_new.gdml --cvd 1 --rtx 1 --envkey --xanalytic --timemax 400 --animtimemax 400 --target 1 --eye -1,-1,-1
>
> failed (See OKX4Test_GOpticalSurface.log). Right now, in order to make the test running, we just added
> if(strncmp(m_finish,"4",strlen(m_finish))==0)  return false ;
> to GOpticalSurface::isSpecular() function.





https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/extg4/X4Solid.cc#lines-1105,

3. In this file why are the startphi and deltaphi not allowed to be 0 and 360
at the same time? I see in G4Polycone class, such case is allowed.


   1091 void X4Solid::convertPolycone()
   1092 {
   1093     // G4GDMLWriteSolids::PolyconeWrite
   1094     // G4GDMLWriteSolids::ZplaneWrite
   1095     // ../analytic/gdml.py
   1096
   1097     //LOG(error) << "START" ;
   1098
   1099     const G4Polycone* const solid = static_cast<const G4Polycone*>(m_solid);
   1100     assert(solid);
   1101     const G4PolyconeHistorical* ph = solid->GetOriginalParameters() ;
   1102
   1103     float startphi = ph->Start_angle/degree ;
   1104     float deltaphi = ph->Opening_angle/degree ;
   1105     assert( startphi == 0.f && deltaphi == 360.f );
   1106


   The assertion on line 1105 is requiring that startphi=0 and deltaphi=360 constraining that
   there is no phi segment applied to the polycone.

   The assert is there just because that has not been needed in the geometries so far faced.
   You are very welcome to do the development work of adding that in a pull request. Make
   sure to include a unit test that tests the new functionality you are adding.

This case exists in this gdml file. if you correct all the things above and run the command:
OKX4Test --deletegeocache --gdmlpath ~/liyu/geometry/rich1_new.gdml --cvd 1 --rtx 1 --envkey --xanalytic --timemax 400 --animtimemax 400 --target 1 --eye -1,-1,-1 --X4 debug
the assertion here will fail (see OKX4Test_X4Solid.log file).

At present, we just remove this assertion and I am willing to find a better solution here.

https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/extg4/X4PhysicalVolume.cc#lines-1398,


4. In this file the names of the inner material and outer material are
extracted and then used in line 1524, 1530, 1536 for GBndLib->addBoundary
function.  In extg4/X4PhysicalVolume.cc, omat and imat are directly extracted
from logical volumes, and may follow this style "_dd_Materials_Air",
"_dd_Materials_Vacuum" But in GBndLib::add function, omat and imat are
extracted from GMaterialLib according to their indexes, and follow this style
"Air", "Vacuum".  Such difference can cause an assertion failed.


   The geometries I work with currently do not have prefixes such as "/dd/Material/"
   on material names, so there could well be a missing X4::BaseName or equivalent somewhere ?
   However the way you reported the issue makes me unsure of what the issue is !

Sorry if my description confuses you. You can refer to OKX4Test_GBndLIb.log file, which are generated by this command
OKX4Test --deletegeocache --gdmlpath ~/liyu/geometry/rich1_new.gdml --cvd 1 --rtx 1 --envkey --xanalytic --timemax 400 --animtimemax 400 --target 1 --eye -1,-1,-1 --X4 debug.
In line 126191, you can see the names of omat and imat with prefixed as "_dd_Materials".

Let's see if you can reproduce these problems and then we can deal with others.

Thank you very much for your help and patience.

Best wishes,

Yunlong






Hi Yunlong, 

> I hope all is well with you. 

Thanks, I'm well. I hope all is well with you too. 

> From our recent studies about Opticks using LHCb RICH detector and other
> simplied geometries, we found some issues and would like to seek for your help.
> Sorry I don't put these issues on groups.io, because they are related to
> different topics.
>
> https://bitbucket.org/simoncblyth/opticks/src/48b41f66c8b0c821e9458e36568d9daf4350bf29/sysrap/SAbbrev.cc#lines-44, 
> 
> 1. In this file it gives the abbreviations of material names which are used by
> GPropertyLib.  But if names are, i.e, "PipeSteel" and “PipeStainlessSteel”,
> which give the same abbreviations, the assertion in line 106 will fail.


See my update to the test sysrap/tests/SAbbrevTest.cc:test_2, that shows that different abbreviations 
are obtained and there is no assert.::

    sysrap/tests/SAbbrevTest.cc:test_2

    111 void test_2()
    112 {
    113     LOG(info);
    114     std::vector<std::string> ss = {
    115         "PipeSteel",
    116         "PipeStainlessSteel"
    117     };
    118     SAbbrev ab(ss);
    119     ab.dump();
    120 }

Running that test::

    SAbbrevTest 

    2021-09-30 19:56:16.207 INFO  [12432035] [test_2@113] 
                         PipeSteel : PS
                PipeStainlessSteel : Pl


I guess your set of material names has a problem but your idea of what the problem is, 
is not correct. 

The best way to investigate and report issues is to add a test to the unit test 
for the relevant class that captures the issue that you are seeing.

Runnable code provides a much more precise, effective and faster way to communicate issues than words. 
Also it is the best way to investigate issues.
 
When I can see the actual problem you are facing via a failing test, 
I can then consider how to fix it.

> But why do we need to use the abbreviations instead of full names?


The OpenGL GUI and also analysis python provides material history sequence tables 
with the material at every step of the photon presented. 
For those tables to be readable a 2 character abbreviation is needed. 

The abbreviation code could definitely be improved to avoid asserts, 
provide me with the set of names in a test that asserts and I will do so.
For example by doing something like you suggest below or even by forming 
random two character abbreviations until a unique one is found.

> A possible way is to change lines 73~86 to::
>
>       if( n->upper == 1 && n->number > 0 ) // 1 or more upper and number
>       {
>           int iu = n->first_upper_index ;
>           int in = n->first_number_index ;
>           ab = n->getTwoChar( iu < in ? iu : in ,  iu < in ? in : iu  );
>       }
>       else if( n->upper >= 2 ) // more than one uppercase : form abbrev from first two uppercase chars
>       {
>           ab = n->getFirstUpper(n->upper) ;
>       }
>       else
>       {
>           ab = n->getFirst(2) ;
>       }




> https://bitbucket.org/simoncblyth/opticks/src/7ebbd54d88ded3b5b713b3133c653012656dc582/ggeo/GOpticalSurface.cc#lines-228, 
> 
> 2. In this file only the polished, polished front-painted and ground mirrors
> are considered, other cases will cause the assertion in line 239 failed. Are
> you planning to handle other types of mirrors?
>

I have no plan to implement more surface types until I need them. 

I am very willing to incorporate your pull requests with more surface types added.  
However I suggest you discuss with me how you plan to do that first to ensure your 
work can be incorporated into Opticks.

However note that Opticks will soon undergo an enormous transition for compatibility 
with the all new NVIDIA OptiX 7 API. 
This transition  means that all GPU code must be re-architected. It is far from 
being a simple transition, the OptiX 7 API is totally different to OptiX 6.5 
As a result the below packages will be removed::

   cudarap
   thrustrap
   optixrap
   okop

With the below packages added::

   QUDARap  : pure CUDA photon generation, no OptiX dependency 
   CSG      : shared CPU/GPU geometry model 
   CSG_GGeo : conversion of GGeo geometry model into CSG 
   CSGOptiX : OptiX 7 ray tracing 
  
A focus for the new architecture is to provide fine-grained modular testing of GPU code. 

Given the tectonic shifts that Opticks will soon undergo, I think it makes
more sense to do things like implement more surface types after the 
dust has settled in the new architecture. 



> https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/extg4/X4Solid.cc#lines-1105, 
>
> 3. In this file why are the startphi and deltaphi not allowed to be 0 and 360
> at the same time? I see in G4Polycone class, such case is allowed.  


    1091 void X4Solid::convertPolycone()
    1092 {
    1093     // G4GDMLWriteSolids::PolyconeWrite
    1094     // G4GDMLWriteSolids::ZplaneWrite
    1095     // ../analytic/gdml.py 
    1096 
    1097     //LOG(error) << "START" ; 
    1098 
    1099     const G4Polycone* const solid = static_cast<const G4Polycone*>(m_solid);
    1100     assert(solid);
    1101     const G4PolyconeHistorical* ph = solid->GetOriginalParameters() ;
    1102 
    1103     float startphi = ph->Start_angle/degree ;
    1104     float deltaphi = ph->Opening_angle/degree ;
    1105     assert( startphi == 0.f && deltaphi == 360.f );
    1106 


The assertion on line 1105 is requiring that startphi=0 and deltaphi=360 constraining that 
there is no phi segment applied to the polycone.

The assert is there just because that has not been needed in the geometries so far faced.  
You are very welcome to do the development work of adding that in a pull request. Make 
sure to include a unit test that tests the new functionality you are adding. 

Again after you have thought about how you want to implement this and done
some preliminary development make sure to discuss your approach with me to 
ensure that your work can be incorporated into Opticks.
I think I have implemented similar things somewhere via CSG intersection with a phi 
segment shape.

The sample problem with the impending shift in Opticks applies however. There is 
little point in doing any developments in the packages that do not have long to live.



> https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/extg4/X4PhysicalVolume.cc#lines-1398, 

>
> 4. In this file the names of the inner material and outer material are
> extracted and then used in line 1524, 1530, 1536 for GBndLib->addBoundary
> function.  In extg4/X4PhysicalVolume.cc, omat and imat are directly extracted
> from logical volumes, and may follow this style "_dd_Materials_Air",
> "_dd_Materials_Vacuum" But in GBndLib::add function, omat and imat are
> extracted from GMaterialLib according to their indexes, and follow this style
> "Air", "Vacuum".  Such difference can cause an assertion failed. 


The geometries I work with currently do not have prefixes such as "/dd/Material/"
on material names : so your problem suggests there is a missing X4::BaseName somewhere ? 
Tell me where and I will add it. 

1384 unsigned X4PhysicalVolume::addBoundary(const G4VPhysicalVolume* const pv, const G4VPhysicalVolume* const pv_p )
1385 {
1386     const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
1387     const G4LogicalVolume* const lv_p = pv_p ? pv_p->GetLogicalVolume() : NULL ;
1388 
1389     // GDMLName adds pointer suffix to the object name, returns null when object is null : eg parent of world 
1390 
1391     const char* _pv = X4::GDMLName(pv) ;
1392     const char* _pv_p = X4::GDMLName(pv_p) ;
1393 
1394 
1395     const G4Material* const imat_ = lv->GetMaterial() ;
1396     const G4Material* const omat_ = lv_p ? lv_p->GetMaterial() : imat_ ;  // top omat -> imat 
1397 
1398     const char* omat = X4::BaseName(omat_) ;
1399     const char* imat = X4::BaseName(imat_) ;
1400 
....
1513     unsigned boundary = 0 ;
1514     if( g_sslv == NULL && g_sslv_p == NULL  )   // no skin surface on this or parent volume, just use bordersurface if there are any
1515     {
1516 
1517 #ifdef OLD_ADD_BOUNDARY
1518         const char* osur = X4::BaseName( osur_ );
1519         const char* isur = X4::BaseName( isur_ );
1520 #else
1521         const char* osur = osur_ ? osur_->getName() : nullptr ;
1522         const char* isur = isur_ ? isur_->getName() : nullptr ;
1523 #endif
1524         boundary = m_blib->addBoundary( omat, osur, isur, imat );
1525     }
1526     else if( g_sslv && !g_sslv_p )   // skin surface on this volume but not parent : set both osur and isur to this 
1527     {
1528         const char* osur = g_sslv->getName();
1529         const char* isur = osur ;
1530         boundary = m_blib->addBoundary( omat, osur, isur, imat );
1531     }
1532     else if( g_sslv_p && !g_sslv )  // skin surface on parent volume but not this : set both osur and isur to this
1533     {
1534         const char* osur = g_sslv_p->getName();
1535         const char* isur = osur ;
1536         boundary = m_blib->addBoundary( omat, osur, isur, imat );
1537     }
1538     else if( g_sslv_p && g_sslv )
1539     {
1540         assert( 0 && "fabled double skin found : see notes/issues/ab-blib.rst  " );
1541     }
1542 
1543     return boundary ;
1544 }

>
>
> A possible way is to deal with omat and imat in the same way as GPropertyMap::FindShortName, change lines 1398~1399 in extg4/X4PhysicalVolume.cc to::
>
>       const char* omat_name = X4::BaseName(omat_);
>       const char* imat_name = X4::BaseName(imat_);
>       const char* omat = NULL;
>       const char* imat = NULL;
>       if( omat_name[0] == '_')
>       {
>           const char* p = strrchr(omat_name, '_') ; 
>           omat = strdup(p+1) ;
>       }
>       else
>       {
>           omat = strdup(omat_name);
>       }
>       if( imat_name[0] == '_')
>       {
>           const char* p = strrchr(imat_name, '_') ; 
>           imat = strdup(p+1) ;
>       }
>       else
>       {
>            imat = strdup(imat_name);
>       }


This way is special casing prefixed names. 

It would be simpler to regularize the names by stripping the prefixes first, 
which is easier to understand and better because it takes less code. 

>
> The same issue exist in 
>
> * https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/extg4/X4MaterialLib.cc#lines-135,

Whats the issue here ? m4_name_base is the name with prefix removed 

::

    129     for(unsigned i=0 ; i < num_materials ; i++)
    130     {
    131         GMaterial*  pmap = m_mlib->getMaterial(i);
    132         G4Material* m4 = (*m_mtab)[i] ;
    133         assert( pmap && m4 );
    134 
    135         const char* pmap_name = pmap->getName();
    136         const std::string& m4_name = m4->GetName();
    137 
    138         bool has_prefix = strncmp( m4_name.c_str(), DD_MATERIALS_PREFIX, strlen(DD_MATERIALS_PREFIX) ) == 0 ;
    139         const char* m4_name_base = has_prefix ? m4_name.c_str() + strlen(DD_MATERIALS_PREFIX) : m4_name.c_str() ;
    140         bool name_match = strcmp( m4_name_base, pmap_name) == 0 ;
    141 
    142         LOG(info)
    143              << std::setw(5) << i
    144              << " ok pmap_name " << std::setw(30) << pmap_name
    145              << " g4 m4_name  " << std::setw(30) << m4_name
    146              << " g4 m4_name_base  " << std::setw(30) << m4_name_base
    147              << " has_prefix " << has_prefix
    148              ;




> * https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/cfg4/CGDMLDetector.cc#lines-206
> * https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/cfg4/CGDMLDetector.cc#lines-206.

Line 206 strips the prefix from the G4Material name if there is one and the lookup 
for the GMaterial is using that unprefixed shortname. What is the issue ?

::

    201     for(unsigned int i=0 ; i < nmat_without_mpt ; i++)
    202     {
    203         G4Material* g4mat = m_traverser->getMaterialWithoutMPT(i) ;
    204         const char* name = g4mat->GetName() ;
    205 
    206         const std::string base = BFile::Name(name);
    207         const char* shortname = base.c_str();
    208 
    209         const GMaterial* ggmat = m_mlib->getMaterial(shortname);
    210         assert(ggmat && strcmp(ggmat->getShortName(), shortname)==0 && "failed to find corresponding G4DAE material") ;
    211 
    212         LOG(verbose)
    213             << " g4mat " << std::setw(45) << name
    214             << " shortname " << std::setw(25) << shortname
    215             ;
    216 

    421 std::string BFile::Name(const char* path)
    422 {
    423     fs::path fsp(path);
    424     std::string name = fsp.filename().string() ;
    425     return name ;
    426 }



Using X4::BaseName on the original material name should get rid of the prefix, see X4Test::

    epsilon:extg4 blyth$ X4Test 
    2021-09-30 20:31:06.725 INFO  [12460728] [test_Name@31] 
     name      : /dd/material/Water
     Name      : /dd/material/Water
     ShortName : /dd/material/Water
     BaseName  : Water

 75 template<typename T>
 76 const char* X4::BaseName( const T* const obj )
 77 {
 78     if(obj == NULL) return NULL ;
 79     const std::string& name = obj->GetName();
 80     return BaseName(name);
 81 }


 40 const char* X4::ShortName( const std::string& name )
 41 {
 42     char* shortname = BStr::trimPointerSuffixPrefix(name.c_str(), NULL) ;
 43     return strdup( shortname );
 44 }
 45 
 46 const char* X4::Name( const std::string& name )
 47 {
 48     return strdup( name.c_str() );
 49 }
 50 
 51 const char* X4::BaseName( const std::string& name)
 52 {
 53     const std::string base = BFile::Name(name.c_str());
 54     return ShortName(base) ;
 55 }


>
>
> https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/ggeo/GMeshLib.cc#lines-193, 
>
> 5. In this file mesh->getAlt can be NULL because it's allowed in line 159, but
> it can cause the following assertion failed. A possible way is to add one line
> after line 193::
>
>       if( mesh->getAlt()==NULL ) continue ; // To be consistent with GMeshLib::saveAltReferences() 
>
> These are some problems we found until now. 


Thank you for working with Opticks.

Life is too short to worry about "theoretical" problems with code, 
there are more than enough real problems.  

So if you have a real issues please report them in a way that I can reproduce them.

Making changes based on code "reading" and possibly incomplete ideas 
of what is happening (or what might happen) is an unwise way to 
direct development efforts. 

I prefer a more traditional approach:

1. you exercise the code and find issues
2. you share the issues in a way that enables me to reproduce them
3. I (or you) try to fix them, preferably by writing simple tests that exercises the code 

For simple issues you could add a unit test that captures the problem, if more complex
you can share some GDML (preferably simplified) that tickles the issue.


> And we are glad to share you some
> pictures of the visualizations of LHCb RICH I geometry and the simplified
> geometry, as attached to this email.

Thank you for sharing the images. Those are very useful to include in presentations 
to enable me to demonstrate all the experiements that are evaluating Opticks
and encourage more adoption.

If you create any more detector geometry and photon path images or movies 
created with Opticks please remember to share them with me.  

>
> Thank you very much for building such an excellent software and look forward to your comments.
>

You are very welcome. 

Simon


