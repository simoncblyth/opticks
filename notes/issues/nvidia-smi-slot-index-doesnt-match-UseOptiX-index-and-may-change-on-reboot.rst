nvidia-smi-slot-index-doesnt-match-UseOptiX-index-and-may-change-on-reboot
===============================================================================

NB this has consequences for interop.  When using OpenGL interop its simplest to 
do everything using the GPU that is connected to the display. So need to arrange that 
the GPU thats driving the display is the only one made visible with the "--cvd" 
argument.


At this reboot TITAN RTX has ordinal 0 according to nvidia-smi::

    [blyth@localhost UseOptiX]$ nvidia-smi
    Fri May 10 20:14:47 2019       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 418.56       Driver Version: 418.56       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN RTX           Off  | 00000000:73:00.0  On |                  N/A |
    | 41%   38C    P8    16W / 280W |    352MiB / 24189MiB |      1%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  TITAN V             Off  | 00000000:A6:00.0 Off |                  N/A |
    | 36%   50C    P8    29W / 250W |      0MiB / 12036MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0     13838      G   /usr/bin/X                                   141MiB |
    |    0     15905      G   /usr/bin/gnome-shell                         201MiB |
    |    0     18478      G   /usr/lib64/firefox/firefox                     3MiB |
    |    0     18974      G   /usr/lib64/firefox/firefox                     3MiB |
    +-----------------------------------------------------------------------------+
    [blyth@localhost UseOptiX]$ 



But the OptiX enumeration doesnt match, it stays at the more familiar TITAN V in slot 0::

    [blyth@localhost UseOptiX]$ UseOptiX
    OptiX version 60000 major.minor.micro 6.0.0   Number of devices = 2 

     Device 0:                        TITAN V  Compute Support: 7 0  Total Memory: 12621381632 bytes 
     Device 1:                      TITAN RTX  Compute Support: 7 5  Total Memory: 25364987904 bytes 


    [blyth@localhost UseOptiX]$ UseOptiX --cvd 0
    setting envvar internally : CUDA_VISIBLE_DEVICES=0
    OptiX version 60000 major.minor.micro 6.0.0   Number of devices = 1 

     Device 0:                        TITAN V  Compute Support: 7 0  Total Memory: 12621381632 bytes 


    [blyth@localhost UseOptiX]$ UseOptiX --cvd 1
    setting envvar internally : CUDA_VISIBLE_DEVICES=1
    OptiX version 60000 major.minor.micro 6.0.0   Number of devices = 1 

     Device 0:                      TITAN RTX  Compute Support: 7 5  Total Memory: 25364987904 bytes 



    [blyth@localhost UseOptiX]$ UseOptiX --cvd 0,1
    setting envvar internally : CUDA_VISIBLE_DEVICES=0,1
    OptiX version 60000 major.minor.micro 6.0.0   Number of devices = 2 

     Device 0:                        TITAN V  Compute Support: 7 0  Total Memory: 12621381632 bytes 
     Device 1:                      TITAN RTX  Compute Support: 7 5  Total Memory: 25364987904 bytes 

    [blyth@localhost UseOptiX]$ UseOptiX --cvd 1,0
    setting envvar internally : CUDA_VISIBLE_DEVICES=1,0
    OptiX version 60000 major.minor.micro 6.0.0   Number of devices = 2 

     Device 0:                      TITAN RTX  Compute Support: 7 5  Total Memory: 25364987904 bytes 
     Device 1:                        TITAN V  Compute Support: 7 0  Total Memory: 12621381632 bytes 




