#!/usr/bin/env python
"""
py3check.py
============

Running this finds all .py files recursively 
within the invoking directory and attempts to parse 
the code as python3, without running the scripts/modules.

::

   cd ~/opticks
   py3check.py 

   INFO:__main__:tot:346 tot_py3:216 tot_py2:130 frac_py3: 0.62 

"""
import sys, os, ast, fnmatch, logging
log = logging.getLogger(__name__)

def ast_parse(path):
    """
    * :google:`check python3 compat without running the script`
    * https://stackoverflow.com/questions/40886456/how-to-detect-if-code-is-python-3-compatible
    """
    code = open(path, 'rb').read()
    try:
        return ast.parse(code)
    except SyntaxError as exc:
        return False

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    assert sys.version_info.major == 3

    tot=0
    tot_py3=0  
    tot_py2=0  
  
    for dirpath, dirnames, names in os.walk("."):
        for name in fnmatch.filter(names, "*.py"):
            path = os.path.join(dirpath, name) 
            py3 = ast_parse(path)
            tot += 1 
            if py3:
                tot_py3 += 1 
            else:
                tot_py2 += 1 
            pass
            st = "Y" if py3 else "N"
            if st == "N":
                print("[%s] %s " %  (st, path))
        pass
    pass

    frac_py3 = float(tot_py3)/float(tot)
    log.info("tot:%d tot_py3:%d tot_py2:%d frac_py3:%5.2f " % (tot,tot_py3,tot_py2,frac_py3))


