#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#


import logging
log = logging.getLogger(__name__) 

# py2->py3  unicode->str str->bytes
try:
    unicode
    ansi_ = lambda msg, codes:unicode("\x1b[%sm%s\x1b[0m" % (";".join(map(str, codes)), msg))
except NameError:
    ansi_ = lambda msg, codes:str("\x1b[%sm%s\x1b[0m" % (";".join(map(str, codes)), msg))
pass

code = {
    "white_on_red_bright":(41,37,1),
    "blink_white_on_red_bright":(5,41,37,1),
    "blink_white_on_red":(5,41,37),
    "blink_red":(5,31,),
    "red":(31,),
    "yellow":(33,),
    "green":(32,),
    "pink":(35,),
    "normal":(0,),
    "underline":(4,),
    "bold":(1,),
    "reverse":(7,),
    "blink":(5,),
}  

red_ = lambda msg:ansi_(msg, code["red"])
underline_ = lambda msg:ansi_(msg, code["underline"])
bold_ = lambda msg:ansi_(msg, code["bold"])
reverse_ = lambda msg:ansi_(msg, code["reverse"])
blink_ = lambda msg:ansi_(msg, code["blink"])

fatal_ = lambda msg:ansi_(msg, code["white_on_red_bright"])
error_ = lambda msg:ansi_(msg, code["red"])
warning_ = lambda msg:ansi_(msg, code["yellow"])
info_ = lambda msg:ansi_(msg, code["green"])
debug_ = lambda msg:ansi_(msg, code["pink"])


enum2func = {
     logging.FATAL:fatal_,
     logging.ERROR:error_,
     logging.WARNING:warning_,
     logging.INFO:info_,
     logging.DEBUG:debug_,
}

def emit_ansi(fn):
    """ 
    Based on https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    """
    def new(*args):
        levelno = args[1].levelno
        args[1].msg = enum2func[levelno](args[1].msg)
        return fn(*args)
    return new

def init_logging(level="info", color=True, cflog=False):
    """
    """
    if color:
        logging.StreamHandler.emit = emit_ansi(logging.StreamHandler.emit)  
    pass
    #fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)-8s - %(message)s'
    #fmt = '[%(asctime)s] p%(process)s {%(filename)-10s:%(lineno)d} %(levelname)-8s - %(message)s'

    if cflog:
        fmt = '[{%(funcName)-20s:%(filename)-10s:%(lineno)d} %(levelname)-8s - %(message)s'
    else:
        fmt = '[%(asctime)s] p%(process)s {%(funcName)-20s:%(filename)-10s:%(lineno)d} %(levelname)-8s - %(message)s'
    pass
    logging.basicConfig(level=getattr(logging,level.upper()), format=fmt)
pass


if __name__ == '__main__':
    init_logging(color=True, level="debug")

    names = "debug info warning error critical fatal".split()
    for name in names: 
        uname = name.upper()
        func = getattr(log, name)
        level = getattr(logging, uname)
        msg = "%20s : %20s : %d " % ( name, uname, level )
        func( msg ) 
    pass   



