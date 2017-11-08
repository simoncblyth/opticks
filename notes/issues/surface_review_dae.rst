Surface Review DAE
===================

Q: Where is ptr in lv/pv names being trimmed  ?


::

    simon:GNodeLib blyth$ head -10 PVNames.txt 
    top
    __dd__Structure__Sites__db-rock0xc15d358
    __dd__Geometry__Sites__lvNearSiteRock--pvNearHallTop0xbf89820
    __dd__Geometry__Sites__lvNearHallTop--pvNearTopCover0xc23f9b8
    __dd__Geometry__Sites__lvNearHallTop--pvNearTeleRpc--pvNearTeleRpc..10xc245d38
    __dd__Geometry__RPC__lvRPCMod--pvRPCFoam0xbf1a820
    __dd__Geometry__RPC__lvRPCFoam--pvBarCham14Array--pvBarCham14ArrayOne..1--pvBarCham14Unit0xc1264d0
    __dd__Geometry__RPC__lvRPCBarCham14--pvRPCGasgap140xc1257a0
    __dd__Geometry__RPC__lvRPCGasgap14--pvStrip14Array--pvStrip14ArrayOne..1--pvStrip14Unit0xc311da0
    __dd__Geometry__RPC__lvRPCGasgap14--pvStrip14Array--pvStrip14ArrayOne..2--pvStrip14Unit0xc125cf8
    simon:GNodeLib blyth$ 
    simon:GNodeLib blyth$ 
    simon:GNodeLib blyth$ head -10 LVNames.txt 
    World0xc15cfc0
    __dd__Geometry__Sites__lvNearSiteRock0xc030350
    __dd__Geometry__Sites__lvNearHallTop0xc136890
    __dd__Geometry__PoolDetails__lvNearTopCover0xc137060
    __dd__Geometry__RPC__lvRPCMod0xbf54e60
    __dd__Geometry__RPC__lvRPCFoam0xc032c88
    __dd__Geometry__RPC__lvRPCBarCham140xbf4c6a0
    __dd__Geometry__RPC__lvRPCGasgap140xbf98ae0
    __dd__Geometry__RPC__lvRPCStrip0xc2213c0
    __dd__Geometry__RPC__lvRPCStrip0xc2213c0
    simon:GNodeLib blyth$ 


rgd::

    854388       <skinsurface name="Tube_surf" surfaceproperty="TubeSurface">
    854389         <volumeref ref="lSurftube0x254b8d0"/>
    854390       </skinsurface>
    854391       <bordersurface name="UpperChimneyTyvekSurface" surfaceproperty="UpperChimneyTyvekOpticalSurface">
    854392         <physvolref ref="pUpperChimneyLS0x2547680"/>
    854393         <physvolref ref="pUpperChimneyTyvek0x2547de0"/>
    854394       </bordersurface>



::

    0848 GSolid* AssimpGGeo::convertStructureVisit(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* /*parent*/)


     923 
     924     GSolid* solid = new GSolid(nodeIndex, gtransform, mesh, UINT_MAX, NULL ); // sensor starts NULL
     925     solid->setLevelTransform(ltransform);
     926 
     927     const char* lv   = node->getName(0);
     928     const char* pv   = node->getName(1);
     929     const char* pv_p   = pnode->getName(1);
     930 
     931     gg->countMeshUsage(msi, nodeIndex, lv, pv);
     932 
     933     GBorderSurface* obs = gg->findBorderSurface(pv_p, pv);  // outer surface (parent->self) 
     934     GBorderSurface* ibs = gg->findBorderSurface(pv, pv_p);  // inner surface (self->parent) 
     935     GSkinSurface*   sks = gg->findSkinSurface(lv);
     936 

    1035     if(m_volnames)
    1036     {
    1037         solid->setPVName(pv);
    1038         solid->setLVName(lv);
    1039     }
    1040 
    1041 

