# === func-gen- : cuda/gfxcardstatus fgp cuda/gfxcardstatus.bash fgn gfxcardstatus fgh cuda
gfxcardstatus-src(){      echo cuda/gfxcardstatus.bash ; }
gfxcardstatus-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(gfxcardstatus-src)} ; }
gfxcardstatus-vi(){       vi $(gfxcardstatus-source) ; }
gfxcardstatus-env(){      elocal- ; }
gfxcardstatus-usage(){ cat << EOU

gfxCardStatus
===============

* http://gfx.io/

gfxCardStatus is an unobtrusive menu bar app for OS X that allows MacBook Pro
users to see which apps are affecting their battery life by using the more
power-hungry graphics.


forcing use of integrated graphics ?
---------------------------------------

* https://devtalk.nvidia.com/default/topic/528725/forcing-retina-macbook-pro-to-use-intel-graphics-for-desktop-freeing-memory-on-cuda-device-/
* http://gfx.io/switching.html#integrated-only-mode-limitations
* https://devtalk.nvidia.com/default/topic/519429/dev-driver-5-0-for-mac/?offset=15#3691304

how it works
--------------

Revolves around IOKit framework IOConnectCallScalarMethod call with **kSetMuxState**  selector in GSMux.m

/usr/local/env/cuda/gfxCardStatus/Classes/GSMux.m::

     42 
     43 typedef enum {
     44     muxDisableFeature    = 0, // set only
     45     muxEnableFeature    = 1, // set only
     46 
     47     muxFeatureInfo        = 0, // get: returns a uint64_t with bits set according to FeatureInfos, 1=enabled
     48     muxFeatureInfo2        = 1, // get: same as MuxFeatureInfo
     49 
     50     muxForceSwitch        = 2, // set: force Graphics Switch regardless of switching mode
     51     // get: always returns 0xdeadbeef
     52 
     53     muxPowerGPU            = 3, // set: power down a gpu, pretty useless since you can't power down the igp and the dedicated gpu is powered down automatically
     54     // get: maybe returns powered on graphics cards, 0x8 = integrated, 0x88 = discrete (or probably both, since integrated never gets powered down?)
     55 
     56     muxGpuSelect        = 4, // set/get: Dynamic Switching on/off with [2] = 0/1 (the same as if you click the checkbox in systemsettings.app)
     57 
     58     // TODO: Test what happens on older mbps when switchpolicy = 0
     59     // Changes if you're able to switch in systemsettings.app without logout
     60     muxSwitchPolicy        = 5, // set: 0 = dynamic switching, 2 = no dynamic switching, exactly like older mbp switching, 3 = no dynamic stuck, others unsupported
     61     // get: possibly inverted?
     62 
     63     muxUnknown            = 6, // get: always 0xdeadbeef
     64 
     65     muxGraphicsCard        = 7, // get: returns active graphics card
     66     muxUnknown2            = 8, // get: sometimes 0xffffffff, TODO: figure out what that means
     67 
     68 } muxState;
     69 

::

    109 static BOOL setMuxState(io_connect_t connect, muxState state, uint64_t arg)
    110 {
    111     kern_return_t kernResult;
    112     uint64_t scalarI_64[3] = { 1 /* always? */, (uint64_t) state, arg };
    113 
    114     kernResult = IOConnectCallScalarMethod(connect,      // an io_connect_t returned from IOServiceOpen().
    115                                            kSetMuxState, // selector of the function to be called via the user client.
    116                                            scalarI_64,   // array of scalar (64-bit) input values.
    117                                            3,            // the number of scalar input values.
    118                                            NULL,         // array of scalar (64-bit) output values.
    119                                            0);           // pointer to the number of scalar output values.
    120 
    121     if (kernResult == KERN_SUCCESS)
    122         GTMLoggerDebug(@"setMuxState was successful.");
    123     else
    124         GTMLoggerDebug(@"setMuxState returned 0x%08x.", kernResult);
    125 
    126     return kernResult == KERN_SUCCESS;
    127 }
    128 



EOU
}
gfxcardstatus-dir(){ echo $(local-base)/env/cuda/gfxCardStatus ; }
gfxcardstatus-cd(){  cd $(gfxcardstatus-dir); }
gfxcardstatus-mate(){ mate $(gfxcardstatus-dir) ; }
gfxcardstatus-get(){
   local dir=$(dirname $(gfxcardstatus-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d "gfxCardStatus" ] && git clone https://github.com/codykrieger/gfxCardStatus


}
