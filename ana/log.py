#!/usr/bin/env python

import logging
log = logging.getLogger(__name__) 

ansi_ = lambda msg, codes:unicode("\x1b[%sm%s\x1b[0m" % (";".join(map(str, codes)), msg))

code = {
    "blink_white_on_red_bright":(5,41,37,1),
    "blink_white_on_red":(5,41,37),
    "blinkred":(5,31,),
    "red":(31,),
    "yellow":(33,),
    "green":(32,),
    "pink":(35,),
    "normal":(0,),
}  

enum2code = {
     logging.FATAL:code["blink_white_on_red_bright"],
     logging.ERROR:code["red"],
     logging.WARNING:code["yellow"],
     logging.INFO:code["green"],
     logging.DEBUG:code["pink"],
}

def emit_ansi(fn):
    """ 
    Based on https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    """
    def new(*args):
        levelno = args[1].levelno
        args[1].msg = ansi_(args[1].msg, enum2code[levelno])
        return fn(*args)
    return new

def init_logging(level="info", color=True):
    """
    """
    if color:
        logging.StreamHandler.emit = emit_ansi(logging.StreamHandler.emit)  
    pass
    #fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)-8s - %(message)s'
    fmt = '[%(asctime)s] p%(process)s {%(filename)-10s:%(lineno)d} %(levelname)-8s - %(message)s'
    logging.basicConfig(level=getattr(logging,level.upper()), format=fmt)
pass


if __name__ == '__main__':
    init_logging(color=True, level="debug")

    names = "debug info warning warn error critical fatal".split()
    for name in names: 
        uname = name.upper()
        func = getattr(log, name)
        level = getattr(logging, uname)
        msg = "%20s : %20s : %d " % ( name, uname, level )
        func( msg ) 
    pass   



