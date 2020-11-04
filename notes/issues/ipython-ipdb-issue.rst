ipython issue
================



::

   ht.py /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1
   ox.py /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1



Issue 
--------

Using python rather than ipython leads to::

    epsilon:g4ok blyth$ ht.py /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1
    Traceback (most recent call last):
      File "/Users/blyth/opticks/ana/ht.py", line 68, in <module>
        from opticks.ana.histype import HisType
      File "/Users/blyth/opticks/ana/histype.py", line 72, in <module>
        from opticks.ana.seq import SeqType, SeqTable, SeqAna
      File "/Users/blyth/opticks/ana/seq.py", line 56, in <module>
        from IPython.core.debugger import Pdb as MyPdb
      File "/Users/blyth/miniconda3/lib/python3.7/site-packages/IPython/__init__.py", line 56, in <module>
        from .terminal.embed import embed
      File "/Users/blyth/miniconda3/lib/python3.7/site-packages/IPython/terminal/embed.py", line 17, in <module>
        from IPython.terminal.ipapp import load_default_config
      File "/Users/blyth/miniconda3/lib/python3.7/site-packages/IPython/terminal/ipapp.py", line 28, in <module>
        from IPython.core.magics import (
      File "/Users/blyth/miniconda3/lib/python3.7/site-packages/IPython/core/magics/__init__.py", line 21, in <module>
        from .execution import ExecutionMagics
      File "/Users/blyth/miniconda3/lib/python3.7/site-packages/IPython/core/magics/execution.py", line 24, in <module>
        import cProfile as profile
      File "/Users/blyth/miniconda3/lib/python3.7/cProfile.py", line 22, in <module>
        run.__doc__ = _pyprofile.run.__doc__
    AttributeError: module 'profile' has no attribute 'run'
    epsilon:g4ok blyth$ 


Have also observed this to cause a failure of IntegrationTests.tboolean.box::

    370 Traceback (most recent call last):
    371   File "/home/blyth/local/opticks/py/opticks/ana/tboolean.py", line 37, in <module>
    372     from opticks.ana.ab   import AB
    373   File "/home/blyth/opticks/ana/ab.py", line 34, in <module>
    374     from opticks.ana.histype import HisType
    375   File "/home/blyth/opticks/ana/histype.py", line 72, in <module>
    376     from opticks.ana.seq import SeqType, SeqTable, SeqAna
    377   File "/home/blyth/opticks/ana/seq.py", line 62, in <module>
    378     from IPython.core.debugger import Pdb as MyPdb
    379   File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/IPython/__init__.py", line 56, in <module>
    380     from .terminal.embed import embed
    381   File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/IPython/terminal/embed.py", line 17, in <module>
    382     from IPython.terminal.ipapp import load_default_config
    383   File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/IPython/terminal/ipapp.py", line 28, in <module>
    384     from IPython.core.magics import (
    385   File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/IPython/core/magics/__init__.py", line 21, in <module>
    386     from .execution import ExecutionMagics
    387   File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/IPython/core/magics/execution.py", line 24, in <module>
    388     import cProfile as profile
    389   File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/cProfile.py", line 22, in <module>
    390     run.__doc__ = _pyprofile.run.__doc__
    391 AttributeError: module 'profile' has no attribute 'run'
    392 2020-11-05 00:37:28.874 INFO  [281260] [SSys::run@100] /home/blyth/local/env/tools/conda/miniconda3/bin/python3 /home/blyth/local/opticks/py/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --cat tboo    lean-box --pfx tboolean-box --src torch --show   rc_raw : 256 rc : 1
    393 2020-11-05 00:37:28.874 ERROR [281260] [SSys::run@107] FAILED with  cmd /home/blyth/local/env/tools/conda/miniconda3/bin/python3 /home/blyth/local/opticks/py/opticks/ana/tboolean.py --tagoffset 0 -    -tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show   RC 1
    394 
    395 2020-11-05 00:37:28.874 INFO  [281260] [OpticksAna::run@129]  anakey tboolean cmdline /home/blyth/local/env/tools/conda/miniconda3/bin/python3 /home/blyth/local/opticks/py/opticks/ana/tboolean.py -    -tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show   interactivity 0 rc 1 rcmsg OpticksAna::run non-zero RC from ana script
    396 2020-11-05 00:37:28.874 FATAL [281260] [Opticks::dumpRC@243]  rc 1 rcmsg : OpticksAna::run non-zero RC from ana script
    397 2020-11-05 00:37:28.889 FATAL [281260] [Opticks::dumpRC@243]  rc 1 rcmsg : OpticksAna::run non-zero RC from ana script
    398 2020-11-05 00:37:28.889 INFO  [281260] [main@32]  RC 1


Despite having renamed profile.py some time ago the .pyc can still be laying around
causing the problem, so must also clean up the python cache files::

    cd ~/opticks/ana
    rm *.pyc
    rm -rf __pycache__ 

Seems the tests are using installed versions of the python::

    [blyth@localhost ana]$ rm -rf __pycache__
    [blyth@localhost ana]$ rm profile.py
    [blyth@localhost ana]$ pwd
    /home/blyth/local/opticks/py/opticks/ana



Avoiding the issue by using ipython makes for complicated commandline
-----------------------------------------------------------------------

::

   ipython $(which ht.py) -i -- /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1
   ipython $(which ox.py) -i -- /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1


Workaround : avoid ipdb unless invoked with ipython
--------------------------------------------------------

::

     59 if sys.argv[0].find("ipython") > -1:
     60     try:
     61         from IPython.core.debugger import Pdb as MyPdb
     62     except ImportError:
     63         class MyPdb(object):
     64             def set_trace(self):
     65                 log.error("IPython is required for ipdb.set_trace() " )
     66             pass
     67         pass
     68     pass
     69     ipdb = MyPdb()
     70 else:
     71     ipdb = None
     72 pass



Googling suggests might be due to a "profile.py"
----------------------------------------------------

::

    epsilon:ana blyth$ git mv profile.py profile_.py 
    epsilon:ana blyth$ rm -rf __pycache__
    

Seems to fix it without the only ipython switch on.



