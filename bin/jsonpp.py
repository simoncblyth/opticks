#!/usr/bin/env python
"""
jsonpp.py 
=========

Pretty print json received on stdin::

    cat some.json | ~/o/bin/jsonpp.py 

"""
import sys, json
if __name__ == '__main__':
    js = json.loads(sys.stdin.read())
    print(json.dumps(js, indent=4))
    key = "cfmeta"
    if key in js:
         print(js[key])
    pass

pass


