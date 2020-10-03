integration_tests_depending_on_tboolean_py_in_PATH
======================================================


::

    2020-10-03 01:59:25.515 INFO  [152633] [OpticksAna::run@92]  anakey tboolean enabled Y
    2020-10-03 01:59:25.515 INFO  [152633] [OpticksAna::run@101]  cmdline tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show  

    sh: tboolean.py: command not found
    2020-10-03 01:59:25.559 INFO  [152633] [SSys::run@100] tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show   rc_raw : 32512 rc : 127
    2020-10-03 01:59:25.559 ERROR [152633] [SSys::run@107] FAILED with  cmd tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show   RC 127

    2020-10-03 01:59:25.559 INFO  [152633] [OpticksAna::run@113]  anakey tboolean cmdline tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show   interactivity 0 rc 127 rcmsg OpticksAna::run non-zero RC from ana script
    2020-10-03 01:59:25.559 FATAL [152633] [Opticks::dumpRC@228]  rc 127 rcmsg : OpticksAna::run non-zero RC from ana script
    2020-10-03 01:59:25.573 INFO  [152633] [Opticks::saveParameters@1198]  postpropagate save parameters.json into TagZeroDir /tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/0
    2020-10-03 01:59:25.574 FATAL [152633] [Opticks::dumpRC@228]  rc 127 rcmsg : OpticksAna::run non-zero RC from ana script
    2020-10-03 01:59:25.574 INFO  [152633] [main@32]  RC 127
    2020-10-03 01:59:25.694 INFO  [152633] [CG4::cleanup@473] [
    2020-10-03 01:59:25.694 INFO  [152633] [CG4::cleanup@475] ]
    === o-main : /home/blyth/local/opticks/lib/OKG4Test --okg4test --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero --generateoverride 10000 --envkey --rendermode +global,+axis --geocenter --stack 2180 --eye 1,0,0 --up 0,0,1 --test --testconfig mode=PyCsgInBox_analytic=1_name=tboolean-box_csgpath=/home/blyth/local/opticks/tmp/tboolean-box_outerfirst=1_autocontainer=Rock//perfectAbsorbSurface/Vacuum_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0_autoseqmap=TO:0,SR:1,SA:0 --torch --torchconfig type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.0_radius=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500 --torchdbg --tag 1 --anakey tboolean --args --save ======= PWD /home/blyth/local/opticks/build/integration/tests RC 127 Sat Oct 3 01:59:25 CST 2020
    echo o-postline : dummy
    o-postline : dummy
    /home/blyth/local/opticks/bin/o.sh : RC : 127
    tboolean-- RC 127
    === tboolean-lv : tboolean-box RC 127
    ====== /home/blyth/local/opticks/bin/tboolean.sh --generateoverride 10000 ====== PWD /home/blyth/local/opticks/build/integration/tests ============ RC 127 =======


    50% tests passed, 1 tests failed out of 2

    Total Test time (real) =  12.90 sec

    The following tests FAILED:
          2 - IntegrationTests.tboolean.box (Failed)
    Errors while running CTest
    Sat Oct  3 01:59:25 CST 2020
    [blyth@localhost integration]$ which tboolean.py
    /usr/bin/which: no tboolean.py in (/home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/bin:/home/...


::

    ${OPTICKS_PYTHON:-python} $OPTICKS_PREFIX/py/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show 

    blyth@localhost ~]$ ${OPTICKS_PYTHON:-python} $OPTICKS_PREFIX/py/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show 
    Traceback (most recent call last):
      File "/home/blyth/local/opticks/py/opticks/ana/tboolean.py", line 32, in <module>
        import os, sys, logging, numpy as np
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/logging/__init__.py", line 26, in <module>
        import sys, os, time, io, traceback, warnings, weakref, collections.abc
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/traceback.py", line 5, in <module>
        import linecache
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/linecache.py", line 11, in <module>
        import tokenize
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/tokenize.py", line 33, in <module>
        import re
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/re.py", line 122, in <module>
        import enum
      File "/home/blyth/local/opticks/py/opticks/ana/enum.py", line 29, in <module>
        import os, re, logging, argparse
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/argparse.py", line 90, in <module>
        from gettext import gettext as _, ngettext
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/gettext.py", line 49, in <module>
        import locale
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/locale.py", line 180, in <module>
        _percent_re = re.compile(r'%(?:\((?P<key>.*?)\))?'
    AttributeError: module 're' has no attribute 'compile'
    [blyth@localhost ~]$ 


Old enum.py in installation dir::

    [blyth@localhost ~]$ rm /home/blyth/local/opticks/py/opticks/ana/enum.py 


MyPdb not working with ipython3::

    [blyth@localhost ~]$ ${OPTICKS_PYTHON:-python} $OPTICKS_PREFIX/py/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show 
    Traceback (most recent call last):
      File "/home/blyth/local/opticks/py/opticks/ana/tboolean.py", line 37, in <module>
        from opticks.ana.ab   import AB
      File "/home/blyth/opticks/ana/ab.py", line 34, in <module>
        from opticks.ana.histype import HisType
      File "/home/blyth/opticks/ana/histype.py", line 68, in <module>
        from opticks.ana.seq import SeqType, SeqTable, SeqAna
      File "/home/blyth/opticks/ana/seq.py", line 30, in <module>
        from IPython.core.debugger import Pdb as MyPdb
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/IPython/__init__.py", line 56, in <module>
        from .terminal.embed import embed
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/IPython/terminal/embed.py", line 17, in <module>
        from IPython.terminal.ipapp import load_default_config
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/IPython/terminal/ipapp.py", line 28, in <module>
        from IPython.core.magics import (
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/IPython/core/magics/__init__.py", line 21, in <module>
        from .execution import ExecutionMagics
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/IPython/core/magics/execution.py", line 24, in <module>
        import cProfile as profile
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/cProfile.py", line 10, in <module>
        import profile as _pyprofile
      File "/home/blyth/local/opticks/py/opticks/ana/profile.py", line 87, in <module>
        ipdb = MyPdb()
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/IPython/core/debugger.py", line 237, in __init__
        self.shell = TerminalInteractiveShell.instance()
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/traitlets/config/configurable.py", line 412, in instance
        inst = cls(*args, **kwargs)
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/IPython/terminal/interactiveshell.py", line 525, in __init__
        super(TerminalInteractiveShell, self).__init__(*args, **kwargs)
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 687, in __init__
        self.init_magics()
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/IPython/terminal/interactiveshell.py", line 508, in init_magics
        super(TerminalInteractiveShell, self).init_magics()
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 2249, in init_magics
        user_magics=m.UserMagics(self))
    AttributeError: module 'IPython.core.magics' has no attribute 'UserMagics'
    [blyth@localhost ~]$ 



::

    [2020-10-03 03:28:32,511] p286094 {__init__            :qdv.py    :132} INFO     - mx None 
    [2020-10-03 03:28:32,511] p286094 {__init__            :ab.py     :126} INFO     - ]
    [2020-10-03 03:28:32,511] p286094 {compare             :ab.py     :499} INFO     - ]
    Traceback (most recent call last):
      File "/home/blyth/local/opticks/py/opticks/ana/tboolean.py", line 46, in <module>
        ab = AB(ok)
      File "/home/blyth/opticks/ana/ab.py", line 337, in __init__
        self.init_point()
      File "/home/blyth/opticks/ana/ab.py", line 515, in init_point
        self.point = self.make_point()
      File "/home/blyth/opticks/ana/ab.py", line 586, in make_point
        rls = self.reclabs(0,None)
      File "/home/blyth/opticks/ana/ab.py", line 1056, in reclabs
        l.extend(Ctx.reclabs_(clab))
      File "/home/blyth/opticks/ana/ctx.py", line 83, in reclabs_
        for ir in range(nsqs):rls[ir,ir] = "[" + rls[ir,ir] + "]"
    TypeError: can only concatenate str (not "numpy.bytes_") to str
    [blyth@localhost ~]$ 
    [blyth@localhost ~]$ o


