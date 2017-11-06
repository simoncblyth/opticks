okg4-material-drastic-difference
=================================

::

    tboolean-;tboolean-media --okg4 
    tboolean-;tboolean-media --okg4 --load --vizg4

    tboolean-;tboolean-media --okg4 -D




Are Opticks MISS and Geant4 fWorldBoundary fully equivalent ?
-----------------------------------------------------------------

tboolean-media geometry is just a single cube "World" of Pyrex. 

* Opticks manages to BR off the edge of the World

  * kinda surprising : no m2 ?  
  * Actually there is an m2 with Opticks, because you set 
    boundaries onto volumes (here it is Rock///Pyrex) which 
    works even when there is one volume


Does an Opticks single volume geometry needs to be translated 
into a Geant4 two volume one ?  

Wont that just defer the issue ?

An Opticks MISS means just that, there was no geometry to hit 
in that direction. Presumbly within a volume based rep 
the containing volume needs to be infinite ? 

What needs to be done practically to get okg4 equivalence ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* make NCSG clone-able and nnode enlarge-able (can restrict imps to just box and sphere)

  * hmm can cheat the clone by a separate deserialize of the first (outer) tree
  * auto-container already has size to fit capability 
 

* then can dynamically (ie no persistent rep) create a wrapper CSG volume 
  (in CTestDetector) just to translate the opticks outer boundant omat 
  into a volume for G4 consumption 
 

* to manage this have expanded NCSGList, providing a context for the multi-tree NCSG statics 
 



cfg4 CRecorder missing most material/boundary info ?
---------------------------------------------------------

Unimplemented bits of CRecorder::RecordStepPoint.

::

    2017-11-06 19:50:44.458 INFO  [3282929] [OpticksEventDump::dump@79]  tagdir /tmp/blyth/opticks/evt/tboolean-media/torch/1 photon_id 23
    (      -74.25     333.02    -399.90         0.20)       (    0.00  -1.00   0.00   378.90)              14      12     124      13    TORCH          ?         ?
    (      -74.25     333.02     400.00         4.09)       (    0.00  -1.00   0.00   378.90)               0       0       0       3     MISS          ?         ?
     ph       23   ux 3264511140   fxyzw    -74.243    333.014    400.000      4.090 
    2017-11-06 19:50:44.458 INFO  [3282929] [OpticksEventDump::dump@79]  tagdir /tmp/blyth/opticks/evt/tboolean-media/torch/-1 photon_id 23
    (      -74.25     333.02    -399.90         0.20)       (    0.00  -1.00   0.00   378.90)              14       0       0      13    TORCH          ?         ?
    (      -74.25     333.02     400.00         4.09)       (    0.00  -1.00   0.00   378.90)              14       0       0       3     MISS          ?         ?
     ph       23   ux 3264511140   fxyzw    -74.243    333.014    400.000      4.090 



Simplify to the max : cube of vacuum, with single bit set in emitconfig.sheetmask
------------------------------------------------------------------------------------

Universe wrapper still disabled


::

    AB(1,torch,tboolean-media)  None 0 
    A tboolean-media/torch/  1 :  20171106-1843 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/1/fdom.npy 
    B tboolean-media/torch/ -1 :  20171106-1843 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/-1/fdom.npy 
    Rock///Vacuum
    /tmp/blyth/opticks/tboolean-media--
    .                seqhis_ana  1:tboolean-media   -1:tboolean-media        c2        ab        ba 
    .                             600000    600000         1.10/2 =  0.55  (pval:0.577 prob:0.423)  
    0000     599461    599445             0.00  TO MI
    0001        497       503             0.04  TO SC MI
    0002         42        52             1.06  TO AB
    .                             600000    600000         1.10/2 =  0.55  (pval:0.577 prob:0.423)  


With universe wrapper 
------------------------------------

* initially looked worse with universe wrapper, but that was 
  due to G4 volume hookup bug, was useful to use GDML export to debug that 

Universe Wrapper attempts to reconcile the 
opticks surface model with the G4 volume model
using a derived additional wrapper universe volume. 

It works for photons that dont cross the inner, but for those
that do have bookeeping difference G4 ends "BT MI" wheres OK just "MI"

Seems unavoidable, so just back to perfectAbsorberSurface : but
this time is should work with G4 because its not the actual 
world boundary.



::


      Opticks single volume with specified border eg Rock///Pyrex
      . 
         +----------+
         |          |
         |          |    Photons leaving the volume are MI for Opticks
         |          |
         |          |
         +----------+


       Geant4 same Pyrex inner volume, with thin Rock universe wrapper 

       +--------------+
       | +----------+ |
       | |          | |   Photons leaving are BT MI(fWorldBoundary) for G4  
       | |          | |  
       | |          | |   
       | |          | |
       | +----------+ |
       +--------------+

       Way to get matched behaviour is to put back perfectAbsorberSurface on both ..
       I commonly did that previously but stopped with okg4 because cannot
       have a border surface at the edge of the universe... but now 
       have uni-wrapper can put it back on the inner.

       This just means my test box has a "velvet" inner surface.

                         


::

    [2017-11-06 20:48:37,153] p58700 {/Users/blyth/opticks/ana/ab.py:137} INFO - AB.init_point DONE
    AB(1,torch,tboolean-media)  None 0 
    A tboolean-media/torch/  1 :  20171106-2047 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/1/fdom.npy 
    B tboolean-media/torch/ -1 :  20171106-2047 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/-1/fdom.npy 
    Rock///Pyrex
    /tmp/blyth/opticks/tboolean-media--
    .                seqhis_ana  1:tboolean-media   -1:tboolean-media        c2        ab        ba 
    .                             600000    600000    608996.86/15 = 40599.79  (pval:0.000 prob:1.000)  
    0000     299543         0        299543.00  TO MI
    0001          0    298231        298231.00  TO BT MI
    0002     289569    290483             1.44  TO AB
    0003          0      5448          5448.00  TO BR BT MI
    0004       5102      5286             3.26  TO BR AB
    0005       5233         0          5233.00  TO BR MI
    0006        152         0           152.00  TO SC MI
    0007          0       134           134.00  TO SC BT MI
    0008         89        99             0.53  TO BR BR AB
    0009          0        91            91.00  TO BR BR BT MI
    0010         82        88             0.21  TO SC AB
    0011         84         0            84.00  TO BR BR MI
    0012         40         0            40.00  TO SC BR MI
    0013          0        35            35.00  TO SC BR BT MI
    0014         32        31             0.02  TO SC BR AB
    0015         22        18             0.40  TO SC BR BR AB
    0016         13        13             0.00  TO SC BR BR BR AB
    0017          2        10             0.00  TO SC BR BR BR BR BR BR BR BR
    0018          9         4             0.00  TO SC BR BR BR BR AB
    0019          7         0             0.00  TO SC BR BR MI
    .                             600000    600000    608996.86/15 = 40599.79  (pval:0.000 prob:1.000)  



After fWorldBoundary -> MISS
--------------------------------


::

    AB(1,torch,tboolean-media)  None 0 
    A tboolean-media/torch/  1 :  20171105-1125 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/1/fdom.npy 
    B tboolean-media/torch/ -1 :  20171105-1125 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/-1/fdom.npy 
    Rock///Pyrex
    .                seqhis_ana  1:tboolean-media   -1:tboolean-media        c2        ab        ba 
    .                             600000    600000     10744.40/8 = 1343.05  (pval:0.000 prob:1.000)  
    0000     299543    308655           136.52  TO MI
    0001     289569    290952             3.29  TO AB
    0002       5233         0          5233.00  TO BR MI
    0003       5102         0          5102.00  TO BR AB
    0004        142       303            58.25  TO SC MI
    0005         98        90             0.34  TO SC AB
    0006         89         0            89.00  TO BR BR AB
    0007         84         0            84.00  TO BR BR MI
    0008         38         0            38.00  TO SC BR MI
    0009         30         0             0.00  TO SC BR AB
    0010         15         0             0.00  TO SC BR BR AB
    0011         12         0             0.00  TO SC BR BR BR AB
    0012         10         0             0.00  TO SC BR BR BR BR AB
    0013         10         0             0.00  TO SC BR BR MI
    0014          8         0             0.00  TO SC BR BR BR BR BR AB
    0015          5         0             0.00  TO SC BR BR BR BR BR BR BR BR
    0016          3         0             0.00  TO BR SC MI
    0017          2         0             0.00  TO SC BR BR BR BR BR BR AB
    0018          2         0             0.00  TO SC BR BR BR MI
    0019          2         0             0.00  TO BR BR BR MI
    .                             600000    600000     10744.40/8 = 1343.05  (pval:0.000 prob:1.000)  



After energy fix for input photons  : the about of bulk AB is close
---------------------------------------------------------------------

Vague recollection

* special cased CRecorder photons leaving world to be "SA" , so "SA == MI" here (TODO: check and get them the same)


::

    AB(1,torch,tboolean-media)  None 0 
    A tboolean-media/torch/  1 :  20171104-1920 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/1/fdom.npy 
    B tboolean-media/torch/ -1 :  20171104-1920 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/-1/fdom.npy 
    Rock///Pyrex
    .                seqhis_ana  1:tboolean-media   -1:tboolean-media        c2        ab        ba 
    .                             600000    600000    619192.64/10 = 61919.26  (pval:0.000 prob:1.000)  
    0000          0    308655        308655.00  TO SA
    0001     299543         0        299543.00  TO MI
    0002     289569    290952             3.29  TO AB
    0003       5233         0          5233.00  TO BR MI
    0004       5102         0          5102.00  TO BR AB
    0005          0       303           303.00  TO SC SA
    0006        142         0           142.00  TO SC MI
    0007         98        90             0.34  TO SC AB
    0008         89         0            89.00  TO BR BR AB
    0009         84         0            84.00  TO BR BR MI
    0010         38         0            38.00  TO SC BR MI
    0011         30         0             0.00  TO SC BR AB
    0012         15         0             0.00  TO SC BR BR AB
    0013         12         0             0.00  TO SC BR BR BR AB
    0014         10         0             0.00  TO SC BR BR BR BR AB
    0015         10         0             0.00  TO SC BR BR MI
    0016          8         0             0.00  TO SC BR BR BR BR BR AB
    0017          5         0             0.00  TO SC BR BR BR BR BR BR BR BR
    0018          3         0             0.00  TO BR SC MI
    0019          2         0             0.00  TO SC BR BR BR BR BR BR AB
    .                             600000    600000    619192.64/10 = 61919.26  (pval:0.000 prob:1.000)  



FIXED : G4 immediate absorb
-------------------------------

::

    delta:issues blyth$ tboolean-;tboolean-media-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-media --tag 1
    ok.smry 1 
    [2017-11-04 18:03:30,763] p23511 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 1 src torch det tboolean-media c2max 2.0 ipython False 
    [2017-11-04 18:03:30,763] p23511 {/Users/blyth/opticks/ana/ab.py:80} INFO - AB.load START smry 1 
    [2017-11-04 18:03:30,819] p23511 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  x : -400.000 400.000 : tot 600000 over 50006 0.083  under 49705 0.083 : mi   -400.000 mx    400.000  
    [2017-11-04 18:03:30,827] p23511 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  y : -400.000 400.000 : tot 600000 over 49882 0.083  under 49906 0.083 : mi   -400.000 mx    400.000  
    [2017-11-04 18:03:30,838] p23511 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  z : -400.000 400.000 : tot 600000 over 50119 0.084  under 50035 0.083 : mi   -400.000 mx    400.000  
    [2017-11-04 18:03:30,845] p23511 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  t :   0.000  20.000 : tot 600000 over 3 0.000  under 0 0.000 : mi      0.200 mx     22.391  
    [2017-11-04 18:03:31,341] p23511 {/Users/blyth/opticks/ana/evt.py:504} WARNING - init_records tboolean-media/torch/ -1 :  finds too few (ph)seqhis uniques : 1 : EMPTY HISTORY
    [2017-11-04 18:03:31,341] p23511 {/Users/blyth/opticks/ana/evt.py:506} WARNING - init_records tboolean-media/torch/ -1 :  finds too few (ph)seqmat uniques : 1 : EMPTY HISTORY
    [2017-11-04 18:03:31,500] p23511 {/Users/blyth/opticks/ana/ab.py:96} INFO - AB.load DONE 
    [2017-11-04 18:03:31,506] p23511 {/Users/blyth/opticks/ana/ab.py:131} INFO - AB.init_point START
    [2017-11-04 18:03:31,509] p23511 {/Users/blyth/opticks/ana/ab.py:133} INFO - AB.init_point DONE
    AB(1,torch,tboolean-media)  None 0 
    A tboolean-media/torch/  1 :  20171104-1800 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/1/fdom.npy 
    B tboolean-media/torch/ -1 :  20171104-1800 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/-1/fdom.npy 
    Rock///Pyrex
    .                seqhis_ana  1:tboolean-media   -1:tboolean-media        c2        ab        ba 
    .                             600000    600000    418659.45/8 = 52332.43  (pval:0.000 prob:1.000)  
    0000     289569    600000        108330.45  TO AB
    0001     299543         0        299543.00  TO MI
    0002       5233         0          5233.00  TO BR MI
    0003       5102         0          5102.00  TO BR AB
    0004        142         0           142.00  TO SC MI
    0005         98         0            98.00  TO SC AB
    0006         89         0            89.00  TO BR BR AB
    0007         84         0            84.00  TO BR BR MI
    0008         38         0            38.00  TO SC BR MI
    0009         30         0             0.00  TO SC BR AB
    0010         15         0             0.00  TO SC BR BR AB
    0011         12         0             0.00  TO SC BR BR BR AB
    0012         10         0             0.00  TO SC BR BR BR BR AB
    0013         10         0             0.00  TO SC BR BR MI
    0014          8         0             0.00  TO SC BR BR BR BR BR AB
    0015          5         0             0.00  TO SC BR BR BR BR BR BR BR BR
    0016          3         0             0.00  TO BR SC MI
    0017          2         0             0.00  TO SC BR BR BR BR BR BR AB
    0018          2         0             0.00  TO SC BR BR BR MI
    0019          2         0             0.00  TO BR BR BR MI
    .                             600000    600000    418659.45/8 = 52332.43  (pval:0.000 prob:1.000)  




::

    (lldb) b "G4OpAbsorption::GetMeanFreePath(G4Track const&, double, G4ForceCondition*)" 


::

    g4-;g4-look G4OpAbsorption.cc:127



::

    (lldb) p aParticle
    error: Couldn't materialize: couldn't get the value of variable aParticle: variable not available
    Errored out in Execute, couldn't PrepareToExecuteJITExpression
    (lldb) p aTrack
    (const G4Track) $0 = {
      fCurrentStepNumber = 1
      fPosition = (dx = 118.3531494140625, dy = 242.328857421875, dz = -399.89999389648438)
      fGlobalTime = 0.20000000298023224
      fLocalTime = 0
      fTrackLength = 0
      fParentID = 0
      fTrackID = 10000
      fVelocity = 299.79245800000001
      fpTouchable = {
        fObj = 0x000000013512f010
      }
      fpNextTouchable = {
        fObj = 0x000000013512f010
      }
      fpOriginTouchable = {
        fObj = 0x000000013512f010
      }
      fpDynamicParticle = 0x000000013512e098
      fTrackStatus = fAlive
      fBelowThreshold = false
      fGoodForTracking = false
      fStepLength = 0
      fWeight = 1
      fpStep = 0x0000000111f1d7d0
      fVtxPosition = (dx = 118.3531494140625, dy = 242.328857421875, dz = -399.89999389648438)
      fVtxMomentumDirection = (dx = -0, dy = -0, dz = 1)
      fVtxKineticEnergy = 0.0000000000032627417774210467
      fpLVAtVertex = 0x0000000111f54080
      fpCreatorProcess = 0x0000000000000000
      fCreatorModelIndex = -1
      fpUserInformation = 0x0000000000000000
      prev_mat = 0x0000000111f4f8c0
      groupvel = 0x0000000111f53150
      prev_velocity = 205.61897277832031
      prev_momentum = 0.0000000000032627417774210467
      is_OpticalPhoton = true
      useGivenVelocity = true
      fpAuxiliaryTrackInformationMap = 0x0000000000000000
    }
    (lldb) 


Curious deep frames do not materialize, but higher ones do

::

    (lldb) p track->GetMaterial()
    (G4Material *) $6 = 0x0000000111f4f8c0
    (lldb) p *track->GetMaterial()
    (G4Material) $7 = {
      fName = (std::__1::string = "Pyrex")
      fChemicalFormula = (std::__1::string = "")
      fDensity = 0.00000062415096471204161
      fState = kStateGas
      fTemp = 293.14999999999998
      fPressure = 632420964.9944762
      maxNbComponents = 1
      fArrayLength = 1

::

    (lldb) p track->GetDynamicParticle()->GetTotalMomentum()
    (G4double) $10 = 0.0000000000032627417774210467


    (lldb) p track->GetMaterial()->GetMaterialPropertiesTable()
    (G4MaterialPropertiesTable *) $11 = 0x0000000111f523a0

    (lldb) p track->GetMaterial()->GetMaterialPropertiesTable()->GetProperty("ABSLENGTH")
    (G4MaterialPropertyVector *) $12 = 0x0000000111f51f50



    (lldb) p track->GetMaterial()->GetMaterialPropertiesTable()->GetProperty("ABSLENGTH")->Value(track->GetDynamicParticle()->GetTotalMomentum()*10000000.)
    (G4double) $17 = 1000

    (lldb) p track->GetMaterial()->GetMaterialPropertiesTable()->GetProperty("ABSLENGTH")->Value(track->GetDynamicParticle()->GetTotalMomentum()*1000000.)
    (G4double) $18 = 1209.2070312499993

    (lldb) p track->GetMaterial()->GetMaterialPropertiesTable()->GetProperty("ABSLENGTH")->Value(track->GetDynamicParticle()->GetTotalMomentum()*100000.)
    (G4double) $19 = 0.000099999997473787516

    (lldb) p track->GetMaterial()->GetMaterialPropertiesTable()->GetProperty("ABSLENGTH")->Value(track->GetDynamicParticle()->GetTotalMomentum())
    (G4double) $20 = 0.000099999997473787516


