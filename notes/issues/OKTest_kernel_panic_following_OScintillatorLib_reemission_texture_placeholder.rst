OKTest
========

In response to this issue in CerenkovMinimal (which has no scintillators)::

    2018-08-14 14:24:04.248 INFO  [16171207] [OContext::close@241] OContext::close numEntryPoint 2
    2018-08-14 14:24:04.249 INFO  [16171207] [OContext::close@245] OContext::close setEntryPointCount done.
    2018-08-14 14:24:04.463 INFO  [16171207] [OContext::close@251] OContext::close m_cfg->apply() done.
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Variable not found (Details: Function "RTresult _rtContextValidate(RTcontext)" caught exception: Variable "Unresolved reference to variable reemission_texture from _Z8generatev_cp5" not found in scope)
    Process 81474 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff74570b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff74570b6e <+10>: jae    0x7fff74570b78            ; <+20>
        0x7fff74570b70 <+12>: movq   %rax, %rdi
        0x7fff74570b73 <+15>: jmp    0x7fff74567b00            ; cerror_nocancel
        0x7fff74570b78 <+20>: retq   
    Target 0: (CerenkovMinimal) stopped.
    (lldb) 


Added handling for a placeholder texture to OScintillatorLib.  
Wished to check scintillator buffer dimensions, so ran OKTest ... 
but got an unually quick kernel panic, seconds after launching OKTest


::

    Anonymous UUID:       32BCAB7F-2AEA-A951-3785-013ECFB913EA

    Tue Aug 14 14:49:14 2018

    *** Panic Report ***
    panic(cpu 4 caller 0xffffff8014689754): "thread_invoke: preemption_level 1, possible cause: blocking while holding a spinlock, or within interrupt context"@/BuildRoot/Library/Caches/com.apple.xbs/Sources/xnu/xnu-4570.51.1/osfmk/kern/sched_prim.c:2231
    Backtrace (CPU 4), Frame : Return Address
    0xffffff81ff933610 : 0xffffff801466e166 
    0xffffff81ff933660 : 0xffffff8014796714 
    0xffffff81ff9336a0 : 0xffffff8014788a00 
    0xffffff81ff933720 : 0xffffff8014620180 
    0xffffff81ff933740 : 0xffffff801466dbdc 
    0xffffff81ff933870 : 0xffffff801466d99c 
    0xffffff81ff9338d0 : 0xffffff8014689754 
    0xffffff81ff933950 : 0xffffff80146885df 
    0xffffff81ff9339a0 : 0xffffff8014781fd6 
    0xffffff81ff933a00 : 0xffffff801461eaad 
    0xffffff81ff933a20 : 0xffffff80146f3f91 
    0xffffff81ff933b00 : 0xffffff80146ba723 
    0xffffff81ff933c30 : 0xffffff801464fc16 
    0xffffff81ff933c60 : 0xffffff8014650803 
    0xffffff81ff933cb0 : 0xffffff8014673df2 
    0xffffff81ff933cf0 : 0xffffff7f94f97ced 
    0xffffff81ff933d30 : 0xffffff7f94fef319 
    0xffffff81ff933d50 : 0xffffff7f950a788e 
    0xffffff81ff933da0 : 0xffffff7f950ecd7c 
    0xffffff81ff933dc0 : 0xffffff7f95e16242 
    0xffffff81ff933e10 : 0xffffff7f94ff650c 
    0xffffff81ff933ed0 : 0xffffff8014c9a005 
    0xffffff81ff933f30 : 0xffffff8014c98772 
    0xffffff81ff933f70 : 0xffffff8014c97dac 
    0xffffff81ff933fa0 : 0xffffff801461f4f7 
          Kernel Extensions in backtrace:
             com.nvidia.web.NVDAResmanWeb(10.3.1)[732647D4-EFC4-3E75-9618-B96D61BE214C]@0xffffff7f94f90000->0xffffff7f95608fff
                dependency: com.apple.iokit.IOPCIFamily(2.9)[1850E7DA-E707-3027-A3AA-637C80B57219]@0xffffff7f94e94000
                dependency: com.apple.iokit.IONDRVSupport(519.15)[B419F958-11B8-3F7D-A31B-A72166B6E234]@0xffffff7f94f75000
                dependency: com.apple.iokit.IOGraphicsFamily(519.15)[D5F2A20D-CAB0-33B2-91B9-E8755DFC34CB]@0xffffff7f94f1f000
                dependency: com.apple.AppleGraphicsDeviceControl(3.18.48)[89491182-0B41-3BC3-B16F-D5043425D66F]@0xffffff7f94f85000
             com.nvidia.web.NVDAGK100HalWeb(10.3.1)[BC0C27F0-12AF-36CA-AC52-ACD84F718B30]@0xffffff7f95d07000->0xffffff7f95e64fff
                dependency: com.nvidia.web.NVDAResmanWeb(10.3.1)[732647D4-EFC4-3E75-9618-B96D61BE214C]@0xffffff7f94f90000
                dependency: com.apple.iokit.IOPCIFamily(2.9)[1850E7DA-E707-3027-A3AA-637C80B57219]@0xffffff7f94e94000

    BSD process name corresponding to current thread: kernel_task

    Mac OS version:
    17E199

    Kernel version:
    Darwin Kernel Version 17.5.0: Mon Mar  5 22:24:32 PST 2018; root:xnu-4570.51.1~1/RELEASE_X86_64
    Kernel UUID: 1B55340B-0B14-3026-8A47-1E139DB63DA3
    Kernel slide:     0x0000000014400000
    Kernel text base: 0xffffff8014600000
    __HIB  text base: 0xffffff8014500000
    System model name: MacBookPro11,3 (Mac-2BD1B31983FE1663)

    System uptime in nanoseconds: 1178683328588779
    last loaded kext at 1171984041203970: com.apple.driver.usb.cdc	5.0.0 (addr 0xffffff7f98349000, size 28672)
    last unloaded kext at 1172251108648188: com.apple.driver.usb.cdc	5.0.0 (addr 0xffffff7f98349000, size 28672)
    loaded kexts:
    com.nvidia.CUDA	1.1.0
    com.nvidia.web.GeForceWeb	10.3.1
    com.nvidia.web.NVDAGK100HalWeb	10.3.1
    com.nvidia.web.NVDAResmanWeb	10.3.1
    com.apple.filesystems.msdosfs	1.10
    com.apple.driver.AppleHWSensor	1.9.5d0
    com.apple.driver.AudioAUUC	1.70
    com.apple.driver.AGPM	110.23.33
    com.apple.driver.ApplePlatformEnabler	2.7.0d0
    com.apple.driver.X86PlatformShim	1.0.0
    com.apple.filesystems.autofs	3.0
    com.apple.driver.AppleHDA	281.51
    com.apple.driver.AppleGraphicsDevicePolicy	3.18.48
    com.apple.AGDCPluginDisplayMetrics	3.18.48
    com.apple.driver.AppleUpstreamUserClient	3.6.5
    com.apple.driver.AppleHV	1
    com.apple.iokit.IOUserEthernet	1.0.1
    com.apple.iokit.IOBluetoothSerialManager	6.0.5f3
    com.apple.driver.pmtelemetry	1
    com.apple.driver.AppleIntelHD5000Graphics	10.3.2
    com.apple.Dont_Steal_Mac_OS_X	7.0.0
    com.apple.driver.eficheck	1
    com.apple.driver.AGDCBacklightControl	3.18.48
    com.apple.driver.AppleLPC	3.1
    com.apple.driver.AppleMuxControl	3.18.48
    com.apple.driver.AppleCameraInterface	6.01.2
    com.apple.driver.AppleThunderboltIP	3.1.1
    com.apple.driver.AppleSMCLMU	211
    com.apple.driver.AppleIntelFramebufferAzul	10.3.2
    com.apple.driver.AppleOSXWatchdog	1
    com.apple.driver.AppleIntelSlowAdaptiveClocking	4.0.0
    com.apple.driver.AppleMCCSControl	1.5.4
    com.apple.driver.AppleUSBCardReader	439.50.6
    com.apple.driver.AppleUSBTCButtons	254
    com.apple.driver.AppleUSBTCKeyboard	254
    com.apple.filesystems.hfs.kext	407.50.6
    com.apple.AppleFSCompression.AppleFSCompressionTypeDataless	1.0.0d1
    com.apple.BootCache	40
    com.apple.AppleFSCompression.AppleFSCompressionTypeZlib	1.0.0
    com.apple.filesystems.apfs	748.51.0
    com.apple.driver.AppleAHCIPort	329.50.2
    com.apple.driver.AirPort.BrcmNIC	1240.29.1a7
    com.apple.driver.AppleSmartBatteryManager	161.0.0
    com.apple.driver.AppleACPIButtons	6.1
    com.apple.driver.AppleRTC	2.0
    com.apple.driver.AppleHPET	1.8
    com.apple.driver.AppleSMBIOS	2.1
    com.apple.driver.AppleACPIEC	6.1
    com.apple.driver.AppleAPIC	1.7
    com.apple.nke.applicationfirewall	183
    com.apple.security.TMSafetyNet	8
    com.apple.security.quarantine	3
    com.apple.kext.triggers	1.0
    com.apple.driver.DspFuncLib	281.51
    com.apple.kext.OSvKernDSPLib	526
    com.apple.iokit.IOAVBFamily	675.6
    com.apple.plugin.IOgPTPPlugin	675.12
    com.apple.iokit.IOEthernetAVBController	1.1.0
    com.apple.driver.AppleSSE	1.0
    com.apple.iokit.IOSerialFamily	11
    com.apple.AppleGPUWrangler	3.18.48
    com.apple.driver.X86PlatformPlugin	1.0.0
    com.apple.driver.IOPlatformPluginFamily	6.0.0d8
    com.apple.driver.AppleGraphicsControl	3.18.48
    com.apple.AppleGraphicsDeviceControl	3.18.48
    com.apple.iokit.IOAcceleratorFamily2	378.18.1
    com.apple.iokit.IOSurface	211.12
    com.apple.iokit.IOSlowAdaptiveClockingFamily	1.0.0
    com.apple.driver.AppleHDAController	281.51
    com.apple.iokit.IOHDAFamily	281.51
    com.apple.iokit.IOAudioFamily	206.5
    com.apple.vecLib.kext	1.2.0
    com.apple.driver.AppleBacklightExpert	1.1.0
    com.apple.iokit.IONDRVSupport	519.15
    com.apple.driver.AppleSMBusController	1.0.18d1
    com.apple.iokit.IOGraphicsFamily	519.15
    com.apple.iokit.BroadcomBluetoothHostControllerUSBTransport	6.0.5f3
    com.apple.iokit.IOBluetoothHostControllerUSBTransport	6.0.5f3
    com.apple.iokit.IOBluetoothHostControllerTransport	6.0.5f3
    com.apple.iokit.IOBluetoothFamily	6.0.5f3
    com.apple.driver.usb.AppleUSBHub	1.2
    com.apple.driver.AppleUSBMultitouch	261
    com.apple.driver.usb.IOUSBHostHIDDevice	1.2
    com.apple.driver.usb.networking	5.0.0
    com.apple.driver.usb.AppleUSBHostCompositeDevice	1.2
    com.apple.filesystems.hfs.encodings.kext	1
    com.apple.iokit.IOAHCIBlockStorage	301.40.2
    com.apple.iokit.IOAHCIFamily	288
    com.apple.driver.AppleThunderboltDPInAdapter	5.5.3
    com.apple.driver.AppleThunderboltDPAdapterFamily	5.5.3
    com.apple.driver.AppleThunderboltPCIDownAdapter	2.1.3
    com.apple.driver.AppleThunderboltNHI	4.7.2
    com.apple.iokit.IOThunderboltFamily	6.7.8
    com.apple.iokit.IO80211Family	1200.12.2
    com.apple.driver.mDNSOffloadUserClient	1.0.1b8
    com.apple.driver.corecapture	1.0.4
    com.apple.driver.usb.AppleUSBHostPacketFilter	1.0
    com.apple.iokit.IOUSBFamily	900.4.1
    com.apple.driver.usb.AppleUSBXHCIPCI	1.2
    com.apple.driver.usb.AppleUSBXHCI	1.2
    com.apple.driver.AppleUSBHostMergeProperties	1.2
    com.apple.driver.AppleEFINVRAM	2.1
    com.apple.driver.AppleEFIRuntime	2.1
    com.apple.iokit.IOHIDFamily	2.0.0
    com.apple.iokit.IOSMBusFamily	1.1
    com.apple.security.sandbox	300.0
    com.apple.kext.AppleMatch	1.0.0d1
    com.apple.driver.DiskImages	480.50.10
    com.apple.driver.AppleFDEKeyStore	28.30
    com.apple.driver.AppleEffaceableStorage	1.0
    com.apple.driver.AppleKeyStore	2
    com.apple.driver.AppleUSBTDM	439.50.6
    com.apple.driver.AppleMobileFileIntegrity	1.0.5
    com.apple.iokit.IOUSBMassStorageDriver	140.50.3
    com.apple.iokit.IOSCSIBlockCommandsDevice	404.30.2
    com.apple.iokit.IOSCSIArchitectureModelFamily	404.30.2
    com.apple.iokit.IOStorageFamily	2.1
    com.apple.driver.AppleCredentialManager	1.0
    com.apple.driver.KernelRelayHost	1
    com.apple.iokit.IOUSBHostFamily	1.2
    com.apple.driver.usb.AppleUSBCommon	1.0
    com.apple.driver.AppleBusPowerController	1.0
    com.apple.driver.AppleSEPManager	1.0.1
    com.apple.driver.IOSlaveProcessor	1
    com.apple.iokit.IOReportFamily	31
    com.apple.iokit.IOTimeSyncFamily	675.12
    com.apple.iokit.IONetworkingFamily	3.4
    com.apple.driver.AppleACPIPlatform	6.1
    com.apple.driver.AppleSMC	3.1.9
    com.apple.iokit.IOPCIFamily	2.9
    com.apple.iokit.IOACPIFamily	1.4
    com.apple.kec.pthread	1
    com.apple.kec.Libm	1
    com.apple.kec.corecrypto	1.0

    EOF
    Model: MacBookPro11,3, BootROM MBP112.0145.B00, 4 processors, Intel Core i7, 2.6 GHz, 16 GB, SMC 2.19f12
    Graphics: Intel Iris Pro, Intel Iris Pro, Built-In
    Graphics: NVIDIA GeForce GT 750M, NVIDIA GeForce GT 750M, PCIe
    Memory Module: BANK 0/DIMM0, 8 GB, DDR3, 1600 MHz, 0x02FE, -
    Memory Module: BANK 1/DIMM0, 8 GB, DDR3, 1600 MHz, 0x02FE, -
    AirPort: spairport_wireless_card_type_airport_extreme (0x14E4, 0x134), Broadcom BCM43xx 1.0 (7.77.37.29.1a7)
    Bluetooth: Version 6.0.5f3, 3 services, 27 devices, 1 incoming serial ports
    Network Service: Wi-Fi, AirPort, en0
    Serial ATA Device: APPLE SSD SM1024F, 1 TB
    USB Device: USB 3.0 Bus
    USB Device: Ultra Fit
    USB Device: Apple Internal Keyboard / Trackpad
    USB Device: BRCM20702 Hub
    USB Device: Bluetooth USB Host Controller
    Thunderbolt Bus: MacBook Pro, Apple Inc., 17.1
