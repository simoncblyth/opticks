U4Touchable_not_compiling_with_Geant4_1120
=============================================


::

    epsilon:issues blyth$ cd /tmp/geant4/
    epsilon:geant4 blyth$ find . -name G4VTouchable.hh
    ./source/geometry/management/include/G4VTouchable.hh
    epsilon:geant4 blyth$ vi source/geometry/management/include/G4VTouchable.hh
    epsilon:geant4 blyth$ 

1120::

    034 #ifndef G4VTOUCHABLE_HH
     35 #define G4VTOUCHABLE_HH 1
     36 
     37 #include "G4TouchableHistory.hh"
     38 
     39 using G4VTouchable = G4TouchableHistory;
     40 
     41 #endif


1042::

    107 class G4VTouchable
    108 {
    109 
    110  public:  // with description
    111 
    112   G4VTouchable();
    113   virtual ~G4VTouchable();
    114     // Constructor and destructor.
    115 








::

    [ 68%] Linking CXX executable G4ThreeVectorTest
    [ 68%] Built target U4PhysicalConstantsTest
    In file included from /home/simon/opticks/u4/tests/U4TouchableTest.cc:1:
    /home/simon/opticks/u4/U4Touchable.h:33:51: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
       33 |     static const G4VPhysicalVolume* FindPV( const G4VTouchable* touch, const char* qname, int mode=MATCH_ALL );
          |                                                   ^~~~~~~~~~~~
          |                                                   U4Touchable
    /home/simon/opticks/u4/U4Touchable.h:35:45: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
       35 |     static int ImmediateReplicaNumber(const G4VTouchable* touch );
          |                                             ^~~~~~~~~~~~
          |                                             U4Touchable
    /home/simon/opticks/u4/U4Touchable.h:36:44: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
       36 |     static int AncestorReplicaNumber(const G4VTouchable* touch, int d=1 );
          |                                            ^~~~~~~~~~~~
          |                                            U4Touchable
    /home/simon/opticks/u4/U4Touchable.h:39:36: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
       39 |     static int ReplicaNumber(const G4VTouchable* touch, const char* replica_name_select ) ;
          |                                    ^~~~~~~~~~~~
          |                                    U4Touchable
    /home/simon/opticks/u4/U4Touchable.h:40:35: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
       40 |     static int ReplicaDepth(const G4VTouchable* touch, const char* replica_name_select ) ;
          |                                   ^~~~~~~~~~~~
          |                                   U4Touchable
    /home/simon/opticks/u4/U4Touchable.h:41:33: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
       41 |     static int TouchDepth(const G4VTouchable* touch );
          |                                 ^~~~~~~~~~~~
          |                                 U4Touchable
    [ 68%] Built target U4Debug_Test
    /home/simon/opticks/u4/U4Touchable.h:44:36: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
       44 |     static std::string Brief(const G4VTouchable* touch );
          |                                    ^~~~~~~~~~~~
          |                                    U4Touchable
    [ 68%] Built target U4PMTFastSimTest
    /home/simon/opticks/u4/U4Touchable.h:45:35: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
       45 |     static std::string Desc(const G4VTouchable* touch, int so_wid=20, int pv_wid=20);
          |                                   ^~~~~~~~~~~~
          |                                   U4Touchable
    /home/simon/opticks/u4/U4Touchable.h:83:60: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
       83 | inline const G4VPhysicalVolume* U4Touchable::FindPV( const G4VTouchable* touch, const char* qname, int mode )
          |                                                            ^~~~~~~~~~~~
          |                                                            U4Touchable
    /home/simon/opticks/u4/U4Touchable.h: In static member function ‘static const G4VPhysicalVolume* U4Touchable::FindPV(const int*, const char*, int)’:
    /home/simon/opticks/u4/U4Touchable.h:85:21: error: request for member ‘GetHistoryDepth’ in ‘* touch’, which is of non-class type ‘const int’
       85 |     int nd = touch->GetHistoryDepth();
          |                     ^~~~~~~~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h:90:47: error: request for member ‘GetVolume’ in ‘* touch’, which is of non-class type ‘const int’
       90 |         const G4VPhysicalVolume* dpv = touch->GetVolume(d);
          |                                               ^~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h: At global scope:
    /home/simon/opticks/u4/U4Touchable.h:126:54: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
      126 | inline int U4Touchable::ImmediateReplicaNumber(const G4VTouchable* touch )
          |                                                      ^~~~~~~~~~~~
          |                                                      U4Touchable
    /home/simon/opticks/u4/U4Touchable.h: In static member function ‘static int U4Touchable::ImmediateReplicaNumber(const int*)’:
    /home/simon/opticks/u4/U4Touchable.h:128:25: error: request for member ‘GetReplicaNumber’ in ‘* touch’, which is of non-class type ‘const int’
      128 |     int copyNo = touch->GetReplicaNumber(1);
          |                         ^~~~~~~~~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h:129:37: error: request for member ‘GetReplicaNumber’ in ‘* touch’, which is of non-class type ‘const int’
      129 |     if(copyNo <= 0) copyNo = touch->GetReplicaNumber(2);
          |                                     ^~~~~~~~~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h: At global scope:
    /home/simon/opticks/u4/U4Touchable.h:145:53: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
      145 | inline int U4Touchable::AncestorReplicaNumber(const G4VTouchable* touch, int d )
          |                                                     ^~~~~~~~~~~~
          |                                                     U4Touchable
    /home/simon/opticks/u4/U4Touchable.h: In static member function ‘static int U4Touchable::AncestorReplicaNumber(const int*, int)’:
    /home/simon/opticks/u4/U4Touchable.h:147:24: error: request for member ‘GetHistoryDepth’ in ‘* touch’, which is of non-class type ‘const int’
      147 |     int depth = touch->GetHistoryDepth();
          |                        ^~~~~~~~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h:151:25: error: request for member ‘GetReplicaNumber’ in ‘* touch’, which is of non-class type ‘const int’
      151 |         copyNo = touch->GetReplicaNumber(d);
          |                         ^~~~~~~~~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h: At global scope:
    /home/simon/opticks/u4/U4Touchable.h:158:45: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
      158 | inline int U4Touchable::ReplicaNumber(const G4VTouchable* touch, const char* replica_name_select )  // static
          |                                             ^~~~~~~~~~~~
          |                                             U4Touchable
    /home/simon/opticks/u4/U4Touchable.h: In static member function ‘static int U4Touchable::ReplicaNumber(const int*, const char*)’:
    /home/simon/opticks/u4/U4Touchable.h:162:32: error: request for member ‘GetReplicaNumber’ in ‘* touch’, which is of non-class type ‘const int’
      162 |     int repno = found ? touch->GetReplicaNumber(d) : d  ;
          |                                ^~~~~~~~~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h: At global scope:
    /home/simon/opticks/u4/U4Touchable.h:198:44: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
      198 | inline int U4Touchable::ReplicaDepth(const G4VTouchable* touch, const char* replica_name_select )   // static
          |                                            ^~~~~~~~~~~~
          |                                            U4Touchable
    /home/simon/opticks/u4/U4Touchable.h: In static member function ‘static int U4Touchable::ReplicaDepth(const int*, const char*)’:
    /home/simon/opticks/u4/U4Touchable.h:200:21: error: request for member ‘GetHistoryDepth’ in ‘* touch’, which is of non-class type ‘const int’
      200 |     int nd = touch->GetHistoryDepth();
          |                     ^~~~~~~~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h:220:47: error: request for member ‘GetVolume’ in ‘* touch’, which is of non-class type ‘const int’
      220 |         const G4VPhysicalVolume* dpv = touch->GetVolume(d);
          |                                               ^~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h:221:47: error: request for member ‘GetVolume’ in ‘* touch’, which is of non-class type ‘const int’
      221 |         const G4VPhysicalVolume* mpv = touch->GetVolume(d+1);
          |                                               ^~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h: At global scope:
    /home/simon/opticks/u4/U4Touchable.h:279:42: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
      279 | inline int U4Touchable::TouchDepth(const G4VTouchable* touch ) // static
          |                                          ^~~~~~~~~~~~
          |                                          U4Touchable
    /home/simon/opticks/u4/U4Touchable.h: In static member function ‘static int U4Touchable::TouchDepth(const int*)’:
    /home/simon/opticks/u4/U4Touchable.h:281:43: error: request for member ‘GetVolume’ in ‘* touch’, which is of non-class type ‘const int’
      281 |     const G4VPhysicalVolume* tpv = touch->GetVolume() ;
          |                                           ^~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h:283:30: error: request for member ‘GetHistoryDepth’ in ‘* touch’, which is of non-class type ‘const int’
      283 |     for(int i=0 ; i < touch->GetHistoryDepth() ; i++)
          |                              ^~~~~~~~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h:285:47: error: request for member ‘GetVolume’ in ‘* touch’, which is of non-class type ‘const int’
      285 |         const G4VPhysicalVolume* ipv = touch->GetVolume(i) ;
          |                                               ^~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h: At global scope:
    /home/simon/opticks/u4/U4Touchable.h:349:45: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
      349 | inline std::string U4Touchable::Brief(const G4VTouchable* touch )
          |                                             ^~~~~~~~~~~~
          |                                             U4Touchable
    /home/simon/opticks/u4/U4Touchable.h: In static member function ‘static std::string U4Touchable::Brief(const int*)’:
    /home/simon/opticks/u4/U4Touchable.h:353:55: error: request for member ‘GetHistoryDepth’ in ‘* touch’, which is of non-class type ‘const int’
      353 |        << " HistoryDepth " << std::setw(2) <<  touch->GetHistoryDepth()
          |                                                       ^~~~~~~~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h: At global scope:
    /home/simon/opticks/u4/U4Touchable.h:360:44: error: ‘G4VTouchable’ does not name a type; did you mean ‘U4Touchable’?
      360 | inline std::string U4Touchable::Desc(const G4VTouchable* touch, int so_wid, int pv_wid )
          |                                            ^~~~~~~~~~~~
          |                                            U4Touchable
    /home/simon/opticks/u4/U4Touchable.h: In static member function ‘static std::string U4Touchable::Desc(const int*, int, int)’:
    /home/simon/opticks/u4/U4Touchable.h:362:32: error: request for member ‘GetHistoryDepth’ in ‘* touch’, which is of non-class type ‘const int’
      362 |     int history_depth = touch->GetHistoryDepth();
          |                                ^~~~~~~~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h:382:40: error: request for member ‘GetVolume’ in ‘* touch’, which is of non-class type ‘const int’
      382 |         G4VPhysicalVolume* pv = touch->GetVolume(i);
          |                                        ^~~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h:385:31: error: request for member ‘GetSolid’ in ‘* touch’, which is of non-class type ‘const int’
      385 |         G4VSolid* so = touch->GetSolid(i);
          |                               ^~~~~~~~
    /home/simon/opticks/u4/U4Touchable.h:386:27: error: request for member ‘GetReplicaNumber’ in ‘* touch’, which is of non-class type ‘const int’
      386 |         G4int cp = touch->GetReplicaNumber(i);
          |                           ^~~~~~~~~~~~~~~~
    [ 68%] Built target U4PMTAccessorTest
    [ 69%] Linking CXX executable U4RotationMatrixTest
    [ 70%] Linking CXX executable U4NistManagerTest
    [ 70%] Built target U4RandomMonitorTest
    [ 70%] Built target G4ThreeVectorTest
    make[2]: *** [tests/CMakeFiles/U4TouchableTest.dir/U4TouchableTest.cc.o] Error 1
    make[1]: *** [tests/CMakeFiles/U4TouchableTest.dir/all] Error 2
    make[1]: *** Waiting for unfinished jobs....
    [ 70%] Built target U4RandomTest

