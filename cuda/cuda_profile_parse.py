#!/usr/bin/env python
"""
::

    (chroma_env)delta:chroma_camera blyth$ ./cuda_profile_parse.py cuda_profile_0.log
    WARNING:__main__:failed to parse : # CUDA_PROFILE_LOG_VERSION 2.0 
    WARNING:__main__:failed to parse : # CUDA_DEVICE 0 GeForce GT 750M 
    WARNING:__main__:failed to parse : # CUDA_CONTEXT 1 
    WARNING:__main__:failed to parse : method,gputime,cputime,occupancy 

    memcpyDtoH           : {'gputime': 201.504, 'cputime': 5260556.83} 
    write_size           : {'gputime': 6.208, 'cputime': 37.704, 'occupancy': 0.048} 
    fill                 : {'gputime': 50.048, 'cputime': 20.733, 'occupancy': 2.0} 
    render               : {'gputime': 5259416.5, 'cputime': 234.175, 'occupancy': 0.5} 
    memcpyHtoD           : {'gputime':   22289.11999999997, 'cputime': 23602.95499999999} 
    (chroma_env)delta:chroma_camera blyth$ 


#. memcpyDtoH consumes the same 'cputime' as render takes 'gputime' with the 
   vast majority of that at the last sample

::

    (chroma_env)delta:chroma_camera blyth$ tail -5 cuda_profile_0.log
    method=[ memcpyHtoD ] gputime=[ 590.560 ] cputime=[ 449.675 ] 
    method=[ fill ] gputime=[ 24.544 ] cputime=[ 13.470 ] occupancy=[ 1.000 ] 
    method=[ fill ] gputime=[ 25.504 ] cputime=[ 7.263 ] occupancy=[ 1.000 ] 
    method=[ render ] gputime=[ 5259416.500 ] cputime=[ 234.175 ] occupancy=[ 0.500 ] 
    method=[ memcpyDtoH ] gputime=[ 194.016 ] cputime=[ 5260492.000 ] 


"""

import sys, os, re, logging
log = logging.getLogger(__name__)

class Parser(object):
    prefix = "method=["
    ptn = re.compile("(\S*)=\[\s(\S*)\s\]")
    def __init__(self):
        self.total = {}   

    def __repr__(self):
        return "\n".join(["%-20s : %s " % ( m,self.total[m] ) for m in self.total.keys()])

    def parse(self, txt):
        """
        Convert lines like::
                     
            method=[ write_size ] gputime=[ 2.528 ] cputime=[ 17.956 ] occupancy=[ 0.016 ] 
            method=[ memcpyDtoH ] gputime=[ 2.496 ] cputime=[ 26.433 ] 
            method=[ write_size ] gputime=[ 1.856 ] cputime=[ 9.821 ] occupancy=[ 0.016 ] 
            method=[ memcpyDtoH ] gputime=[ 2.496 ] cputime=[ 20.545 ] 

        Into dicts::

            {'gputime': 2.528, 'occupancy': 0.016, 'method': 'write_size', 'cputime': 17.956}
            {'gputime': 2.496, 'method': 'memcpyDtoH', 'cputime': 26.433}
            {'gputime': 1.856, 'occupancy': 0.016, 'method': 'write_size', 'cputime': 9.821}
            {'gputime': 2.496, 'method': 'memcpyDtoH', 'cputime': 20.545}
            {'gputime': 1.824, 'occupancy': 0.016, 'method': 'write_size', 'cputime': 9.927}
            {'gputime': 2.496, 'method': 'memcpyDtoH', 'cputime': 17.852}
            {'gputime': 1.44, 'method': 'memcpyHtoD', 'cputime': 4.299}
            
        """
        if txt[0:len(self.prefix)] != self.prefix:
            return None
        d = {}
        for m in re.findall(self.ptn,txt):
            k = m[0]
            try:
                v = float(m[1])
            except ValueError:
                v = m[1]
            d[k] = v
        return d

    def accumulate(self, d):
        method = d.pop('method')
        if not method in self.total:
            self.total[method] = {}

        for k,v in d.items():
            if not k in self.total[method]:
                self.total[method][k] = 0.  
            self.total[method][k] += v 

    def __call__(self,path):
        for line in open(path,"r").readlines():
            txt = line[:-1]
            d = self.parse(txt)
            if d is None:
                log.debug("failed to parse : %s " % txt )
            else:
                yield d 
                self.accumulate(d)

def main():
    logging.basicConfig(level=logging.INFO)
    p = Parser()
    p(sys.argv[1])
    print p

if __name__ == '__main__':
    main()


