
#include <set>
#include <string>


#include "Opticks.hh"
#include "GGeo.hh"
#include "GBndLib.hh"
#include "GSurfaceLib.hh"
#include "GMergedMesh.hh"


#include "GSurLib.hh"


#include "PLOG.hh"
#include "GGEO_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    GGEO_LOG__ ;

    Opticks ok(argc, argv);
    ok.configure();

    GGeo gg(&ok);
    gg.loadFromCache();
    gg.dumpStats();


    GSurLib gsl(&ok, &gg);

    gsl.close();
    gsl.dump();


    return 0 ; 
}

/**

2016-09-25 13:47:17.473 INFO  [288257] [GSurLib::collectSur@51]  nsur 48
2016-09-25 13:47:17.498 INFO  [288257] [GSurLib::dump@124] test_GSurLib

    Only three inner boundaries surfaces 

    2 (   2                 NearOWSLinerSurface)  ibnd   1 obnd   0 ivol   1 ovol   0 npvp   1 nlv   1 [ ibnd   14:Tyvek//NearOWSLinerSurface/OwsWater] 
    3 (   3               NearIWSCurtainSurface)  ibnd   1 obnd   0 ivol   1 ovol   0 npvp   1 nlv   1 [ ibnd   16:Tyvek//NearIWSCurtainSurface/IwsWater] 
    5 (   5                       SSTOilSurface)  ibnd   1 obnd   0 ivol   2 ovol   0 npvp   1 nlv   1 [ ibnd   19:StainlessSteel//SSTOilSurface/MineralOil] 

    0 (   0                NearPoolCoverSurface)  ibnd   0 obnd   1 ivol   0 ovol   1 npvp   1 nlv   1 [ obnd    3:Air/NearPoolCoverSurface//PPE] 
    1 (   1                NearDeadLinerSurface)  ibnd   0 obnd   1 ivol   0 ovol   1 npvp   1 nlv   1 [ obnd   13:DeadWater/NearDeadLinerSurface//Tyvek] 
    4 (   4                SSTWaterSurfaceNear1)  ibnd   0 obnd   1 ivol   0 ovol   1 npvp   1 nlv   1 [ obnd   18:IwsWater/SSTWaterSurfaceNear1//StainlessSteel] 

    6 (   6       lvPmtHemiCathodeSensorSurface)  ibnd   0 obnd   1 ivol   0 ovol 672 npvp   1 nlv   1 [ obnd   29:Vacuum/lvPmtHemiCathodeSensorSurface//Bialkali] 
    7 (   7     lvHeadonPmtCathodeSensorSurface)  ibnd   0 obnd   1 ivol   0 ovol  12 npvp   1 nlv   1 [ obnd   34:Vacuum/lvHeadonPmtCathodeSensorSurface//Bialkali] 

    8 (   8                        RSOilSurface)  ibnd   0 obnd   1 ivol   0 ovol  64 npvp  32 nlv   1 [ obnd   37:MineralOil/RSOilSurface//Acrylic] 
    9 (   9                    ESRAirSurfaceTop)  ibnd   0 obnd   1 ivol   0 ovol   2 npvp   1 nlv   1 [ obnd   39:Air/ESRAirSurfaceTop//ESR] 
   10 (  10                    ESRAirSurfaceBot)  ibnd   0 obnd   1 ivol   0 ovol   2 npvp   1 nlv   1 [ obnd   40:Air/ESRAirSurfaceBot//ESR] 

   All below surfaces are outer boundaries of bits of steel or PVC in water 

   11 (  11                  AdCableTraySurface)  ibnd   0 obnd   1 ivol   0 ovol   2 npvp   2 nlv   1 [ obnd   76:IwsWater/AdCableTraySurface//UnstStainlessSteel] 
   12 (  12                SSTWaterSurfaceNear2)  ibnd   0 obnd   1 ivol   0 ovol   1 npvp   1 nlv   1 [ obnd   80:IwsWater/SSTWaterSurfaceNear2//StainlessSteel] 
   13 (  13                 PmtMtTopRingSurface)  ibnd   0 obnd   2 ivol   0 ovol 288 npvp 288 nlv   1 [ obnd   82:IwsWater/PmtMtTopRingSurface//UnstStainlessSteel 100:OwsWater/PmtMtTopRingSurface//UnstStainlessSteel] 
   14 (  14                PmtMtBaseRingSurface)  ibnd   0 obnd   2 ivol   0 ovol 288 npvp 288 nlv   1 [ obnd   83:IwsWater/PmtMtBaseRingSurface//UnstStainlessSteel 101:OwsWater/PmtMtBaseRingSurface//UnstStainlessSteel] 
   15 (  15                    PmtMtRib1Surface)  ibnd   0 obnd   2 ivol   0 ovol 864 npvp 864 nlv   1 [ obnd   84:IwsWater/PmtMtRib1Surface//UnstStainlessSteel 102:OwsWater/PmtMtRib1Surface//UnstStainlessSteel] 
   16 (  16                    PmtMtRib2Surface)  ibnd   0 obnd   2 ivol   0 ovol 864 npvp 864 nlv   1 [ obnd   86:IwsWater/PmtMtRib2Surface//UnstStainlessSteel 104:OwsWater/PmtMtRib2Surface//UnstStainlessSteel] 
   17 (  17                    PmtMtRib3Surface)  ibnd   0 obnd   2 ivol   0 ovol 864 npvp 864 nlv   1 [ obnd   87:IwsWater/PmtMtRib3Surface//UnstStainlessSteel 105:OwsWater/PmtMtRib3Surface//UnstStainlessSteel] 
   18 (  18                  LegInIWSTubSurface)  ibnd   0 obnd   1 ivol   0 ovol   8 npvp   8 nlv   1 [ obnd   88:IwsWater/LegInIWSTubSurface//ADTableStainlessSteel] 
   19 (  19                   TablePanelSurface)  ibnd   0 obnd   1 ivol   0 ovol   2 npvp   2 nlv   1 [ obnd   89:IwsWater/TablePanelSurface//ADTableStainlessSteel] 
   20 (  20                  SupportRib1Surface)  ibnd   0 obnd   1 ivol   0 ovol   8 npvp   8 nlv   1 [ obnd   90:IwsWater/SupportRib1Surface//ADTableStainlessSteel] 
   21 (  21                  SupportRib5Surface)  ibnd   0 obnd   1 ivol   0 ovol   4 npvp   4 nlv   1 [ obnd   91:IwsWater/SupportRib5Surface//ADTableStainlessSteel] 
   22 (  22                    SlopeRib1Surface)  ibnd   0 obnd   1 ivol   0 ovol   8 npvp   8 nlv   1 [ obnd   92:IwsWater/SlopeRib1Surface//ADTableStainlessSteel] 
   23 (  23                    SlopeRib5Surface)  ibnd   0 obnd   1 ivol   0 ovol   8 npvp   8 nlv   1 [ obnd   93:IwsWater/SlopeRib5Surface//ADTableStainlessSteel] 
   24 (  24             ADVertiCableTraySurface)  ibnd   0 obnd   1 ivol   0 ovol   2 npvp   2 nlv   1 [ obnd   94:IwsWater/ADVertiCableTraySurface//UnstStainlessSteel] 
   25 (  25            ShortParCableTraySurface)  ibnd   0 obnd   1 ivol   0 ovol   2 npvp   2 nlv   1 [ obnd   95:IwsWater/ShortParCableTraySurface//UnstStainlessSteel] 
   26 (  26               NearInnInPiperSurface)  ibnd   0 obnd   1 ivol   0 ovol   1 npvp   1 nlv   1 [ obnd   96:IwsWater/NearInnInPiperSurface//PVC] 
   27 (  27              NearInnOutPiperSurface)  ibnd   0 obnd   1 ivol   0 ovol   1 npvp   1 nlv   1 [ obnd   97:IwsWater/NearInnOutPiperSurface//PVC] 
   28 (  28                  LegInOWSTubSurface)  ibnd   0 obnd   1 ivol   0 ovol   8 npvp   8 nlv   1 [ obnd  106:OwsWater/LegInOWSTubSurface//ADTableStainlessSteel] 
   29 (  29                 UnistrutRib6Surface)  ibnd   0 obnd   1 ivol   0 ovol  16 npvp  16 nlv   1 [ obnd  107:OwsWater/UnistrutRib6Surface//UnstStainlessSteel] 
   30 (  30                 UnistrutRib7Surface)  ibnd   0 obnd   1 ivol   0 ovol  16 npvp  16 nlv   1 [ obnd  108:OwsWater/UnistrutRib7Surface//UnstStainlessSteel] 
   31 (  31                 UnistrutRib3Surface)  ibnd   0 obnd   1 ivol   0 ovol  92 npvp  92 nlv   1 [ obnd  109:OwsWater/UnistrutRib3Surface//UnstStainlessSteel] 
   32 (  32                 UnistrutRib5Surface)  ibnd   0 obnd   1 ivol   0 ovol 192 npvp 192 nlv   1 [ obnd  110:OwsWater/UnistrutRib5Surface//UnstStainlessSteel] 
   33 (  33                 UnistrutRib4Surface)  ibnd   0 obnd   1 ivol   0 ovol 330 npvp 330 nlv   1 [ obnd  111:OwsWater/UnistrutRib4Surface//UnstStainlessSteel] 
   34 (  34                 UnistrutRib1Surface)  ibnd   0 obnd   1 ivol   0 ovol  16 npvp  16 nlv   1 [ obnd  112:OwsWater/UnistrutRib1Surface//UnstStainlessSteel] 
   35 (  35                 UnistrutRib2Surface)  ibnd   0 obnd   1 ivol   0 ovol  16 npvp  16 nlv   1 [ obnd  113:OwsWater/UnistrutRib2Surface//UnstStainlessSteel] 
   36 (  36                 UnistrutRib8Surface)  ibnd   0 obnd   1 ivol   0 ovol  32 npvp  32 nlv   1 [ obnd  114:OwsWater/UnistrutRib8Surface//UnstStainlessSteel] 
   37 (  37                 UnistrutRib9Surface)  ibnd   0 obnd   1 ivol   0 ovol  32 npvp  32 nlv   1 [ obnd  115:OwsWater/UnistrutRib9Surface//UnstStainlessSteel] 
   38 (  38            TopShortCableTraySurface)  ibnd   0 obnd   1 ivol   0 ovol   2 npvp   2 nlv   1 [ obnd  116:OwsWater/TopShortCableTraySurface//UnstStainlessSteel] 
   39 (  39           TopCornerCableTraySurface)  ibnd   0 obnd   1 ivol   0 ovol   4 npvp   4 nlv   1 [ obnd  117:OwsWater/TopCornerCableTraySurface//UnstStainlessSteel] 
   40 (  40               VertiCableTraySurface)  ibnd   0 obnd   1 ivol   0 ovol   8 npvp   8 nlv   1 [ obnd  118:OwsWater/VertiCableTraySurface//UnstStainlessSteel] 
   41 (  41               NearOutInPiperSurface)  ibnd   0 obnd   1 ivol   0 ovol   1 npvp   1 nlv   1 [ obnd  119:OwsWater/NearOutInPiperSurface//PVC] 
   42 (  42              NearOutOutPiperSurface)  ibnd   0 obnd   1 ivol   0 ovol   1 npvp   1 nlv   1 [ obnd  120:OwsWater/NearOutOutPiperSurface//PVC] 
   43 (  43                 LegInDeadTubSurface)  ibnd   0 obnd   1 ivol   0 ovol   8 npvp   8 nlv   1 [ obnd  121:DeadWater/LegInDeadTubSurface//ADTableStainlessSteel] 
   44 (  44                perfectDetectSurface)  ibnd   0 obnd   0 ivol   0 ovol   0 npvp   0 nlv   0
   45 (  45                perfectAbsorbSurface)  ibnd   0 obnd   0 ivol   0 ovol   0 npvp   0 nlv   0
   46 (  46              perfectSpecularSurface)  ibnd   0 obnd   0 ivol   0 ovol   0 npvp   0 nlv   0
   47 (  47               perfectDiffuseSurface)  ibnd   0 obnd   0 ivol   0 ovol   0 npvp   0 nlv   0


   All surfaces are single lv 

**/

