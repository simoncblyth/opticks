RTX-stack
============


Issue
--------

Changing RTX stack size seems not to be doing anything to performance.

maxTraceDepth
maxCallableProgramDepth




::


    OKTest --target 62590 --pfx scan-pf-0 --cat cvd_1_rtx_1_1M --generateoverride 1000000 --compute --save --production --savehit --dbghitmask TO,BT,RE,SC,SA --multievent 10 --xanalytic --rngmax 3 --cvd 1 --rtx 1 --maxTraceDepth 2 --maxCallableProgramDepth 2 

    // --usageReportLevel 3



::

    ip profile.py --cat cvd_1_rtx_1_1M --pfx scan-pf-0 --tag 0



DYB Launch timings as wind down the stack::

    In [2]: pr.q
    Out[2]: array([0.0204, 0.0182, 0.0208, 0.0191, 0.0179, 0.02  , 0.0193, 0.018 , 0.0178, 0.0196])

    In [1]: pr.q
    Out[1]: array([0.0221, 0.0173, 0.0181, 0.0196, 0.0208, 0.018 , 0.0184, 0.0183, 0.0172, 0.0178])

    In [1]: pr.q
    Out[1]: array([0.0221, 0.0182, 0.021 , 0.0191, 0.0192, 0.0184, 0.0206, 0.0178, 0.0179, 0.0188])    

    In [2]: np.average(pr.q)
    Out[2]: 0.019305899999744726



::

    OKTest --target 62590 --pfx scan-pf-0 --cat cvd_1_rtx_1_1M --generateoverride 1000000 --compute --save --production --savehit --dbghitmask TO,BT,RE,SC,SA --multievent 10 --xanalytic --rngmax 3 --cvd 1 --rtx 1 --maxTraceDepth 10 --maxCallableProgramDepth 5 



::

    In [1]: pr.q
    Out[1]: array([0.0191, 0.0187, 0.0184, 0.0177, 0.0162, 0.0183, 0.0173, 0.0182, 0.018 , 0.0171])



    In [1]: pr.q
    Out[1]: array([0.0212, 0.0178, 0.0204, 0.0181, 0.0181, 0.0208, 0.0195, 0.0179, 0.0179, 0.0189])     --maxTraceDepth 20 --maxCallableProgramDepth 20 


    In [1]: pr.q
    Out[1]: array([0.0201, 0.0179, 0.0187, 0.0183, 0.018 , 0.0197, 0.0204, 0.0178, 0.0208, 0.0202])        --maxTraceDepth 2 --maxCallableProgramDepth 2    




