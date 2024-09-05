X11_forwarding_between_same_desk_workstations
==============================================

One desk : one laptop + two*( workstation + monitor + keyboard),  three GPUs (4 including laptop):

* P : "Precision" with TITAN V, TITAN RTX
* A : "Ada" with RTX 5000 Ada Generation



WORKS : ssh P->A, display A, GPU from A, console on P "DISPLAY=:1"
---------------------------------------------------------------------

* ACTUALLY USEFUL : MEANS CAN USE ONE KEYBOARD 


* launch using keyboard connected to P, 
* see console on P
* see graphics on monitor connected to A
* enter key controls using keyboad connected to A

* inverse also working (P<=>A) 


P::

    [blyth@localhost ~]$ grep -A4  "host A" ~/.ssh/config
    host A
        user blyth
        hostname 192.168.185.246
        #ForwardX11 yes
        #ForwardX11Trusted yes


    [blyth@localhost ~]$ ssh A
    Web console: https://localhost:9090/ or ... 
    Last login: Thu Sep  5 10:51:21 2024 from ...

    [blyth@localhost ~]$ DISPLAY=:1 ~/o/cx.sh 





NOT RELIABLY : ssh P->A, display P, GPU from A, console on P "DISPLAY=:1"
-----------------------------------------------------------------------------

* "BY MISTAKE" THIS MANAGED TO WORK ONCE : BUT WAS SLOW
* UNCLEAR WHY WOULD WANT TO DO THIS : AT LEAST IN CURRENT CONTEXT  


P::

    [blyth@localhost ~]$ ssh A
    ...
    [blyth@localhost ~]$ DISPLAY=:0 ~/o/cx.sh 

    2024-09-05 11:02:08.761 INFO  [292201] [CSGOptiX::initPIDXYZ@703]  params->pidxyz (4294967295,4294967295,4294967295) 
    SGLFW::Error_callback: X11: Failed to open display :0


Try again with changed ~/.ssh/config on P::

    [blyth@localhost ~]$ grep -A4 host\ A ~/.ssh/config
    host A
        user blyth
        hostname 192.168.185.246
        ForwardX11 yes
        ForwardX11Trusted yes




