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



