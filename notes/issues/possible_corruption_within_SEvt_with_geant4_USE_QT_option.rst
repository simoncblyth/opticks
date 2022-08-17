possible_corruption_within_SEvt_with_geant4_USE_QT_option
===========================================================


USE_QT : potentially a header name clash : most likely with QUDARAP
-----------------------------------------------------------------------

source/visualization/OpenGL/src/G4OpenGLQtViewer.cc::

      53 #include "G4UnitsTable.hh"
      54 #include "G4OpenGLStoredQtViewer.hh"
      55 #include "G4Threading.hh"
      56 
      57 #include <CLHEP/Units/SystemOfUnits.h>
      58 
      59 #include <typeinfo>
      60 
      61 #include <qlayout.h>
      62 #include <qlabel.h>
      63 #include <qdialog.h>
      64 #include <qpushbutton.h>
      65 #include <qprocess.h>
      66 #include <qdesktopwidget.h>
      67 
      68 #include <qmenu.h>
      69 #include <qimagewriter.h>
      70 
      71 #include <qtextedit.h>
      72 #include <qtreewidget.h>
      73 #include <qapplication.h>
      74 #include <qmessagebox.h>
      75 #include <qfiledialog.h>
      76 #include <qprinter.h>
      77 #include <qdatetime.h>
      78 #include <qpainter.h>
      79 #include <qgl.h> // include <qglwidget.h>
      80 #include <qdialog.h>
      81 #include <qcolordialog.h>
      82 #include <qevent.h> //include <qcontextmenuevent.h>
      83 #include <qobject.h>
      84 #include <qgroupbox.h>
      85 #include <qcombobox.h>
      86 #include <qlineedit.h>
      87 #include <qsignalmapper.h>
      88 #include <qmainwindow.h>
      89 #include <qtablewidget.h>
      90 #include <qheaderview.h>
      91 #include <qscrollarea.h>


::

    epsilon:geant4.10.04.p02 blyth$ find . -name '*.cc' -exec grep -l QT {} \;
    ./source/visualization/management/src/G4VisManager.cc
    ./source/visualization/RayTracer/src/G4RTJpegCoder.cc
    ./source/visualization/OpenGL/src/G4OpenGLStoredQt.cc
    ./source/visualization/OpenGL/src/G4OpenGLQtViewer.cc
    ./source/visualization/OpenGL/src/G4OpenGLImmediateQt.cc
    ./source/visualization/OpenGL/src/G4OpenGLQtMovieDialog.cc
    ./source/visualization/OpenGL/src/G4OpenGLQt.cc
    ./source/visualization/OpenGL/src/G4OpenGLQtExportDialog.cc
    ./source/visualization/OpenGL/src/G4OpenGLViewer.cc
    ./source/visualization/OpenGL/src/G4OpenGLVboDrawer.cc
    ./source/visualization/OpenGL/src/G4OpenGLStoredQtViewer.cc
    ./source/visualization/OpenGL/src/G4OpenGLImmediateQtViewer.cc
    ./source/visualization/OpenGL/src/G4OpenGLStoredQtSceneHandler.cc
    ./source/processes/hadronic/models/lend/src/nf_gammaFunctions.cc
    ./source/processes/hadronic/models/parton_string/hadronization/src/G4VLongitudinalStringDecay.cc
    ./source/processes/hadronic/models/parton_string/hadronization/src/G4LundStringFragmentation.cc
    ./source/processes/electromagnetic/lowenergy/src/G4PenelopeIonisationModel.cc
    ./source/externals/clhep/src/RandGaussQ.cc
    ./source/global/HEPRandom/src/G4MTRandGaussQ.cc
    ./source/interfaces/basic/src/G4UIExecutive.cc
    ./source/interfaces/basic/src/G4UIQt.cc
    ./source/interfaces/common/src/G4Qt.cc
    ./examples/advanced/ChargeExchangeMC/ChargeExchangeMC.cc
    ./examples/advanced/ChargeExchangeMC/src/CexmcHistoManagerMessenger.cc
    ./examples/advanced/ChargeExchangeMC/src/CexmcHistoWidget.cc
    ./examples/advanced/ChargeExchangeMC/src/CexmcHistoManager.cc
    ./examples/extended/medical/dna/chem3/chem3.cc
    ./examples/extended/medical/dna/pdb4dna/pdb4dna.cc
    ./examples/extended/medical/dna/wholeNuclearDNA/wholeNuclearDNA.cc
    ./examples/extended/medical/dna/microdosimetry/microdosimetry.cc
    epsilon:geant4.10.04.p02 blyth$ 




1. Is the corruption issue reproducible ? When you put that back does it still happen ?
------------------------------------------------------------------------------------------

Yes this corruption issue is reproducible.

With help of Tao, we suspect the reason is that I open the USE_QT option to
compile the Geant4. Thus I close the option and rebuild it, Then it works.

The below is error output. Maybe opticks is confict with USE_QT ?


::

    *** Error in `./build/example': double free or corruption (out): 0x00007fffb3509970 ***
    ======= Backtrace: =========
    /lib64/libc.so.6(+0x81299)[0x7f9e88946299]
    /home/ihep/juno-cmake-version/JUNOSOFT/ExternalLibs/opticks/head/lib64/libSysRap.so(_ZN4SEvt6gatherEv+0x428)[0x7f9e8a5252aa]
    /home/ihep/juno-cmake-version/JUNOSOFT/ExternalLibs/opticks/head/lib64/libG4CX.so(_ZN11G4CXOpticks8simulateEv+0x353)[0x7f9e94e6ed93]
    ./build/example[0x4286e6]
    /home/ihep/juno-cmake-version/JUNOSOFT/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so(_ZN14G4EventManager12DoProcessingEP7G4Event+0x872)[0x7f9e920c09f2]
    /home/ihep/juno-cmake-version/JUNOSOFT/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so(_ZN12G4RunManager11DoEventLoopEiPKci+0xaf)[0x7f9e923571df]
    /home/ihep/juno-cmake-version/JUNOSOFT/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so(_ZN12G4RunManager6BeamOnEiPKci+0x5e)[0x7f9e9235509e]
    /home/ihep/juno-cmake-version/JUNOSOFT/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so(_ZN14G4RunMessenger11SetNewValueEP11G4UIcommand8G4String+0x571)[0x7f9e9236c5d1]
    /home/ihep/juno-cmake-version/JUNOSOFT/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4intercoms.so(_ZN11G4UIcommand4DoItE8G4String+0x53c)[0x7f9e8e2df90c]
    /home/ihep/juno-cmake-version/JUNOSOFT/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4intercoms.so(_ZN11G4UImanager12ApplyCommandEPKc+0xcd4)[0x7f9e8e2fcbd4]
    /home/ihep/juno-cmake-version/JUNOSOFT/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4intercoms.so(_ZN9G4UIbatch11ExecCommandERK8G4String+0x17)[0x7f9e8e2cf467]
    /home/ihep/juno-cmake-version/JUNOSOFT/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4intercoms.so(_ZN9G4UIbatch12SessionStartEv+0x5e)[0x7f9e8e2d100e]
    /home/ihep/juno-cmake-version/JUNOSOFT/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4intercoms.so(_ZN11G4UImanager16ExecuteMacroFileEPKc+0x32)[0x7f9e8e2fd712]
    /home/ihep/juno-cmake-version/JUNOSOFT/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4intercoms.so(_ZN20G4UIcontrolMessenger11SetNewValueEP11G4UIcommand8G4String+0x21a)[0x7f9e8e2ebb3a]
    /home/ihep/juno-cmake-version/JUNOSOFT/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4intercoms.so(_ZN11G4UIcommand4DoItE8G4String+0x53c)[0x7f9e8e2df90c]
    /home/ihep/juno-cmake-version/JUNOSOFT/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4intercoms.so(_ZN11G4UImanager12ApplyCommandEPKc+0xcd4)[0x7f9e8e2fcbd4]
    ./build/example[0x41235d]
    /lib64/libc.so.6(__libc_start_main+0xf5)[0x7f9e888e7555]
    ./build/example[0x413c47]
    ======= Memory map: ========
    00400000-004a2000 r-xp 00000000 fd:00 555238219                          /home/ihep/LS_Sim_v2/LS_Sim/build/example
    006a1000-006a5000 r--p 000a1000 fd:00 555238219                          /home/ihep/LS_Sim_v2/LS_Sim/build/example
    006a5000-006ab000 rw-p 000a5000 fd:00 555238219                          /home/ihep/LS_Sim_v2/LS_Sim/build/example
    006ab000-006ae000 rw-p 00000000 00:00 0 
    015e2000-0901e000 rw-p 00000000 00:00 0                                  [heap]
    200000000-200200000 ---p 00000000 00:00 0 
    200200000-200400000 rw-s 00000000 00:05 72363                            /dev/nvidiactl
    200400000-200600000 rw-s 00000000 00:05 27871                            /dev/nvidia0
    200600000-203e00000 rw-s 00000000 00:05 72363                            /dev/nvidiactl
    203e00000-204000000 rw-s 00000000 00:05 27871                            /dev/nvidia0
    204000000-204e00000 ---p 00000000 00:00 0 
    204e00000-205000000 rw-s 00000000 00:05 72363                            /dev/nvidiactl
    205000000-205200000 rw-s 00000000 00:05 72363                            /dev/nvidiactl
    205200000-205400000 rw-s 205200000 00:05 2221468                         /dev/nvidia-uvm
    205400000-205600000 rw-s 00000000 00:05 72363                            /dev/nvidiactl
    205600000-205800000 ---p 00000000 00:00 0 
    205800000-205a00000 rw-s 00000000 00:05 72363                            /dev/nvidiactl
    205a00000-205c00000 rw-s 00000000 00:04 56058810                         /dev/zero (deleted)
    205c00000-300200000 ---p 00000000 00:00 0 
    10000000000-10004000000 ---p 00000000 00:00 0 
    7f9d2c000000-7f9d2c021000 rw-p 00000000 00:00 0 
    7f9d2c021000-7f9d30000000 ---p 00000000 00:00 0 
    7f9d32000000-7f9d3e000000 ---p 00000000 00:00 0 





