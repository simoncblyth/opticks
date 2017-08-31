j1707_250k_volumes_render_slowly_with_full_tri
===================================================

Issue
--------

::

    op --j1707 --gltf 3 --tracer


Huge geometry, > 250k Volumes, 18k instances + 38k instances

* performant in bbox mode, 60-30 fps gives nice interactivity
* (hit B key to shift mode) bogs down in full tri mode ~3 fps gives painful interactivity

This is without any culling or LOD.



LODCull for all instances or just the PMTs ? JUST PMTs
-----------------------------------------------------------

* switching between bbox and inst rendering for the sFasteners and sStrut 
  makes no difference to interactivity... 

* almost certainly applying LODCull to only 480 instances would not be beneficial : 
  just not enough instances to see any benefit, using INSTANCE_MINIMUM = 10000
  

::

   2017-08-28 11:52:34.748 INFO  [1181753] [NScene::dumpRepeatCount@1429] NScene::dumpRepeatCount m_verbosity 1
     ridx   1 count 182860   ## 36572*(4+1) = 182860   PMT_3inch_pmt_solid0x1c9e270    (progeny 4)
     ridx   2 count 106434   ## 17739*(5+1) = 106434   sMask_virtual0x18163c0          (progeny 5) 
     ridx   3 count   480    ##   480*(0+1) =    480   sFasteners0x1506180             (progeny 0)
     ridx   4 count   480    ##   480*(0+1) =    480   sStrut0x14ddd50                 (progeny 0)

     **  ##  idx   0 pdig 68a31892bccd1741cc098d232c702605 num_pdig  36572 num_progeny      4 NScene::meshmeta mesh_id  22 lvidx  20 height  1 soname        PMT_3inch_pmt_solid0x1c9e270 lvname              PMT_3inch_log0x1c9ef80
     **      idx   1 pdig 683529bb1b0fedc340f2ebce47468395 num_pdig  36572 num_progeny      0 NScene::meshmeta mesh_id  26 lvidx  19 height  0 soname       PMT_3inch_cntr_solid0x1c9e640 lvname         PMT_3inch_cntr_log0x1c9f1f0
     **      idx   2 pdig c81fb13777b701cb8ce6cdb7f0661f1b num_pdig  36572 num_progeny      0 NScene::meshmeta mesh_id  25 lvidx  17 height  0 soname PMT_3inch_inner2_solid_ell_helper0x1c9e5d0 lvname       PMT_3inch_inner2_log0x1c9f120
     **      idx   3 pdig 83a5a282f092aa7baf6982b54227bb54 num_pdig  36572 num_progeny      0 NScene::meshmeta mesh_id  24 lvidx  16 height  0 soname PMT_3inch_inner1_solid_ell_helper0x1c9e510 lvname       PMT_3inch_inner1_log0x1c9f050
     **      idx   4 pdig 50308873a9847d1c2c2029b6c9de7eeb num_pdig  36572 num_progeny      2 NScene::meshmeta mesh_id  23 lvidx  18 height  0 soname PMT_3inch_body_solid_ell_ell_helper0x1c9e4a0 lvname         PMT_3inch_body_log0x1c9eef0
     **      idx   5 pdig 27a989a1aeab2b96cedd2b6c4a7cba2f num_pdig  17739 num_progeny      0 NScene::meshmeta mesh_id  17 lvidx  10 height  2 soname                      sMask0x1816f50 lvname                      lMask0x18170e0
     **      idx   6 pdig e39a411b54c3ce46fd382fef7f632157 num_pdig  17739 num_progeny      0 NScene::meshmeta mesh_id  21 lvidx  12 height  4 soname    PMT_20inch_inner2_solid0x1863010 lvname      PMT_20inch_inner2_log0x1863310
     **      idx   7 pdig 74d8ce91d143cad52fad9d3661dded18 num_pdig  17739 num_progeny      0 NScene::meshmeta mesh_id  20 lvidx  11 height  4 soname    PMT_20inch_inner1_solid0x1814a90 lvname      PMT_20inch_inner1_log0x1863280
     **      idx   8 pdig a80803364fbf92f1b083ebff420b6134 num_pdig  17739 num_progeny      2 NScene::meshmeta mesh_id  19 lvidx  13 height  3 soname      PMT_20inch_body_solid0x1813ec0 lvname        PMT_20inch_body_log0x1863160
     **      idx   9 pdig 6b1283d04ffc8a27e19f84e2bec2ddd6 num_pdig  17739 num_progeny      3 NScene::meshmeta mesh_id  18 lvidx  14 height  3 soname       PMT_20inch_pmt_solid0x1813600 lvname             PMT_20inch_log0x18631f0
     **  ##  idx  10 pdig 8cbe68d7d5c763820ff67b8088e0de98 num_pdig  17739 num_progeny      5 NScene::meshmeta mesh_id  16 lvidx  15 height  0 soname              sMask_virtual0x18163c0 lvname               lMaskVirtual0x1816910
     **  ##  idx  11 pdig ad8b68a55505a09ac7578f32418904b3 num_pdig    480 num_progeny      0 NScene::meshmeta mesh_id  15 lvidx   9 height  2 soname                 sFasteners0x1506180 lvname                 lFasteners0x1506370
     **  ##  idx  12 pdig f93b8bbbac89ea22bac0bf188ba49a61 num_pdig    480 num_progeny      0 NScene::meshmeta mesh_id  14 lvidx   8 height  1 soname                     sStrut0x14ddd50 lvname                     lSteel0x14dde40




How to integrate something like env-/instcull-/LODCullShader into oglrap ?
----------------------------------------------------------------------------

Differences, 

* UBO rather than lots of little uniform calls


LODCullShader via transform feedback and geometry shader forks an original 
instance transforms buffer into three separate GPU buffers (for three LOD levels), 
filtering by instance center positions being within frustum of current view and forking 
by distance from the eye to the instances into 3 LOD piles.


How to structure ?
~~~~~~~~~~~~~~~~~~~~~~

* LODCull needs to be an optional constituent of the instanced oglrap-/Renderer 
  depending on instance transform counts exceeding a minimum as configured in oglrap-/Scene



Testing InstLODCull
----------------------

::

    op --j1707 --gltf 3 --tracer --instcull


