G4TheRayTracer
=================


::

    g4-;g4-cls G4RTSimpleScanner
    g4-;g4-cls G4VRTScanner





    tboolean-;tboolean-zsphere1 --okg4 --g4snap -D

    open /tmp/blyth/opticks/CRayTracer.jpeg 


* after apply VisAtt get smth
* need to hook up Composition to take the snap with standard viewpoint/camera config 



::

    (lldb) b "G4TheRayTracer::CreateBitMap"


    Process 24031 stopped
    * thread #1: tid = 0x5d6b73, 0x0000000104677477 libG4RayTracer.dylib`G4TheRayTracer::CreateBitMap(this=0x000000010f0578a0) + 23 at G4TheRayTracer.cc:194, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x0000000104677477 libG4RayTracer.dylib`G4TheRayTracer::CreateBitMap(this=0x000000010f0578a0) + 23 at G4TheRayTracer.cc:194
       191  G4bool G4TheRayTracer::CreateBitMap()
       192  {
       193    G4int iEvent = 0;
    -> 194    G4double stepAngle = viewSpan/100.;
       195    G4double viewSpanX = stepAngle*nColumn;
       196    G4double viewSpanY = stepAngle*nRow;
       197    G4bool succeeded;
    (lldb) p nRow
    (G4int) $0 = 640
    (lldb) p nColumn
    (G4int) $1 = 640
    (lldb) p viewSpan
    (G4double) $2 = 0.087266462599716474
    (lldb) 



    (lldb) c
    Process 24031 resuming
    Process 24031 stopped
    * thread #1: tid = 0x5d6b73, 0x0000000104677b99 libG4RayTracer.dylib`G4TheRayTracer::CreateBitMap(this=0x000000010f0578a0) + 1849 at G4TheRayTracer.cc:278, queue = 'com.apple.main-thread', stop reason = breakpoint 6.1
        frame #0: 0x0000000104677b99 libG4RayTracer.dylib`G4TheRayTracer::CreateBitMap(this=0x000000010f0578a0) + 1849 at G4TheRayTracer.cc:278
       275      theRayShooter->Shoot(anEvent,rayPosition,rayDirection.unit());
       276      theEventManager->ProcessOneEvent(anEvent);
       277      succeeded = GenerateColour(anEvent);
    -> 278      colorR[iCoord] = (unsigned char)(int(255*rayColour.GetRed()));
       279      colorG[iCoord] = (unsigned char)(int(255*rayColour.GetGreen()));
       280      colorB[iCoord] = (unsigned char)(int(255*rayColour.GetBlue()));
       281        } else {  // Ray does not intercept world at all.
    (lldb) p suceeded
    error: use of undeclared identifier 'suceeded'
    error: 1 errors parsing expression
    (lldb) p rayColour
    (G4Colour) $9 = (red = 1, green = 1, blue = 1, alpha = 1)
    (lldb) 

    (lldb) p trajectory
    (G4RayTrajectory *) $20 = 0x000000010c6d1db0
    (lldb) p *trajectory
    (G4RayTrajectory) $21 = {
      positionRecord = 0x00000001096327f0 size=2
    }
    (lldb) 



::

    g4-;g4-cls G4RayTrajectory 
    g4-;g4-cls G4RayTrajectoryPoint   # perhaps need to apply VisAttributes ?


::

     51 class G4RayTrajectoryPoint :public G4VTrajectoryPoint
     52 {
     53   public:
     54     G4RayTrajectoryPoint();
     55     virtual ~G4RayTrajectoryPoint();
     56 
     57     inline void *operator new(size_t);
     58     inline void operator delete(void *aTrajectoryPoint);
     59   //    inline int operator==(const G4RayTrajectoryPoint& right) const
     60   // { return (this==&right); };
     61 
     62   private:
     63     const G4VisAttributes* preStepAtt;
     64     const G4VisAttributes* postStepAtt;
     65     G4ThreeVector    surfaceNormal;
     66     G4double         stepLength;
     67 


::

    (lldb) b G4TheRayTracer::GetSurfaceColour

    (lldb) f 1
    frame #1: 0x0000000104677fb5 libG4RayTracer.dylib`G4TheRayTracer::GetSurfaceColour(this=0x000000010f0578a0, point=0x000000010c6d2ad0) + 37 at G4TheRayTracer.cc:344
       341  
       342  G4Colour G4TheRayTracer::GetSurfaceColour(G4RayTrajectoryPoint* point)
       343  {
    -> 344    const G4VisAttributes* preAtt = point->GetPreStepAtt();
       345    const G4VisAttributes* postAtt = point->GetPostStepAtt();
       346  
       347    G4bool preVis = ValidColour(preAtt);
    (lldb) p point
    (G4RayTrajectoryPoint *) $26 = 0x000000010c6d2ad0
    (lldb) p *point
    (G4RayTrajectoryPoint) $27 = {
      preStepAtt = 0x0000000000000000
      postStepAtt = 0x0000000000000000
      surfaceNormal = (dx = 1, dy = 0, dz = 0)
      stepLength = 2657.1261006483824
    }
    (lldb) 


::


    simon:geant4_10_02_p01 blyth$ find examples -type f -exec grep -H G4VisAttributes {} \;
    examples/advanced/air_shower/src/UltraDetectorConstruction.cc:#include "G4VisAttributes.hh"
    examples/advanced/air_shower/src/UltraDetectorConstruction.cc:   G4VisAttributes* UniverseVisAtt = new G4VisAttributes(G4Colour(1.0,1.0,1.0));
    examples/advanced/air_shower/src/UltraDetectorConstruction.cc:   World_log->SetVisAttributes (G4VisAttributes::Invisible);
    examples/advanced/air_shower/src/UltraDetectorConstruction.cc:G4VisAttributes* SurfaceVisAtt = new G4VisAttributes(G4Colour(0.0,0.0,1.0));
    examples/advanced/air_shower/src/UltraDetectorConstruction.cc:G4VisAttributes* SurfaceVisAtt = new G4VisAttributes(G4Colour(0.0,0.0,1.0));
    examples/advanced/air_shower/src/UltraDetectorConstruction.cc:  G4VisAttributes* PMTVisAtt   = new G4VisAttributes(true,G4Colour(0.0,0.0,1.0)) ;   



g4-;g4-cls G4TheRayTracer
--------------------------


setters for these::

    149     G4ThreeVector eyePosition;
    150     G4ThreeVector targetPosition;
    151     G4ThreeVector eyeDirection;
    152     G4ThreeVector lightDirection;
    153     G4ThreeVector up;
    154     G4double headAngle;
    155     G4double viewSpan; // Angle per 100 pixels
    156     G4double attenuationLength;
    157 


::

    191 G4bool G4TheRayTracer::CreateBitMap()
    192 {
    193   G4int iEvent = 0;
    194   G4double stepAngle = viewSpan/100.;
    195   G4double viewSpanX = stepAngle*nColumn;
    196   G4double viewSpanY = stepAngle*nRow;
    197   G4bool succeeded;
    198 
    199   G4VVisManager* visMan = G4VVisManager::GetConcreteInstance();
    200   visMan->IgnoreStateChanges(true);
    201 





::

    simon:geant4_10_02_p01 blyth$ g4-cc G4TheRayTracer
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4RayTracer.cc:#include "G4TheRayTracer.hh"
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4RayTracer.cc:  theRayTracer = new G4TheRayTracer;  // Establish default ray tracer.
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4RayTracerViewer.cc:#include "G4TheRayTracer.hh"
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4RayTracerViewer.cc: G4TheRayTracer* aTracer):
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4RayTracerViewer.cc:  if (!aTracer) theTracer = new G4TheRayTracer;
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4RayTracerXViewer.cc:#include "G4TheRayTracer.hh"
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4RayTracerXViewer.cc:          new G4TheRayTracer(new G4RTJpegMaker, new G4RTXScanner))
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4RTMessenger.cc:#include "G4TheRayTracer.hh"
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4RTMessenger.cc:(G4TheRayTracer* p1)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4RTMessenger.cc:G4RTMessenger::G4RTMessenger(G4TheRayTracer* p1)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4RTXScanner.cc:#include "G4TheRayTracer.hh"
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4TheMTRayTracer.cc:: G4TheRayTracer(figMaker,scanner)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4TheRayTracer.cc:// $Id: G4TheRayTracer.cc 86973 2014-11-21 11:57:27Z gcosmo $
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4TheRayTracer.cc:#include "G4TheRayTracer.hh"
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4TheRayTracer.cc:G4TheRayTracer::G4TheRayTracer(G4VFigureFileMaker* figMaker,
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4TheRayTracer.cc:G4TheRayTracer::~G4TheRayTracer()
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4TheRayTracer.cc:void G4TheRayTracer::Trace(const G4String& fileName)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4TheRayTracer.cc:void G4TheRayTracer::StoreUserActions()
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4TheRayTracer.cc:void G4TheRayTracer::RestoreUserActions()
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4TheRayTracer.cc:G4bool G4TheRayTracer::CreateBitMap()
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4TheRayTracer.cc:void G4TheRayTracer::CreateFigureFile(const G4String& fileName)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4TheRayTracer.cc:G4bool G4TheRayTracer::GenerateColour(G4Event* anEvent)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4TheRayTracer.cc:G4Colour G4TheRayTracer::GetMixedColour
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4TheRayTracer.cc:G4Colour G4TheRayTracer::GetSurfaceColour(G4RayTrajectoryPoint* point)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4TheRayTracer.cc:G4Colour G4TheRayTracer::Attenuate
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4TheRayTracer.cc:G4bool G4TheRayTracer::ValidColour(const G4VisAttributes* visAtt)
    simon:geant4_10_02_p01 blyth$ 



