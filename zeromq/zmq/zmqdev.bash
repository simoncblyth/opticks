# === func-gen- : zeromq/zmq/zmq fgp zeromq/zmq/zmq.bash fgn zmq fgh zeromq/zmq
zmqdev-src(){      echo zeromq/zmq/zmq.bash ; }
zmqdev-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(zmqdev-src)} ; }
zmqdev-vi(){       vi $(zmqdev-source) ; }
zmqdev-env(){      olocal- ; }
zmqdev-usage(){ cat << EOU

ZMQ : Low Level C API 
=======================

This is for low level ZMQ C API usage/examples.

See also:

*zeromq-*
    getting/building/installing  etc.. 

*czmqdev-*
    higher level C binding

*pyzmqdev-*
    python binding






TODO
-----

#. configure zmq_broker to run continuously beneath supervisord 


ZMQ Broker as used by G4DAEChroma
-----------------------------------

The G4DAEChroma ZMQ broker provides two ports with
TCPIP urls provided by envvars:

* ZMQ_BROKER_URL_FRONTEND
* ZMQ_BROKER_URL_BACKEND

Clients with data to process send it to the frontend. 
Examples of clients:

* csa.sh (nuwa.py script with DetSimChroma/G4DAEChroma configured)
* env/zeromq/pyzmq/npysend.sh (simple test script that reads photon or step (cerenkov/scintillation)
  files as sends them to the broker) 

Workers which are able to do the processing connect to the backend, 
and receive their data from the broker. Examples of workers:

* env/geant4/geometry/collada/g4daeview/g4daechroma.sh
* g4daeview.sh


SSH Tunneling/port forwarding 
----------------------------------

*zmqdev-tunnel-open*

     configures and opens in background an ssh tunnel and
     modifies TCPIP url according

*zmqdev-tunnel-cmd laddr raddr*

     ssh -fN -p 22 -L laddr:raddr

     -f backgrounds ssh
     -N no remote command   
     -p 22   port to use
     -L setup port forwarding


client/broker/worker topology
-----------------------------

Broker sits in the middle between client(s) and worker(s).
The client REQuest goes to the ROUTER socket and worker REPly returns to DEALER socket. 
The broker asynchronously monitors the ROUTER/DEALER sockets internally, 
and ensures that replys are sent back to the client that requested it.
ROUTER/DEALER sockets are non-blocking variants of the blocking REQ/REP sockets. 

NB unlike simple client/server REQ/REP between two nodes, the workers
need to **connect** (not **bind**) to their sockets. 

#. client **connects** to frontend, configured by envvar such as:

   * FRONTEND=tcp://ip.addr.of.broker:5001

#. worker **connects** to backend, configured by envvar:

   * BACKEND=tcp://ip.addr.of.broker:5002

#. broker running on node ip.addr.of.broker **binds** locally 
   to frontend and backend sockets, configured by envvars such as:

   * FRONTEND=tcp://*:5001  (the ROUTER socket)
   * BACKEND=tcp://*:5002   (the DEALER socket) 
   

The big advantage is that clients and workers need to know nothing 
about each other, meaning that clients and workers can come and go with no
need for reconfiguration. 

Only the "stable" broker IP address and relevant frontend/backend port 
needs be known by clients/workers.

dodgy outgoing network workaround
-----------------------------------

belle7 refuses to operate as client or worker talking to remote broker on ports 5001, 5002

* it seems that outgoing connections on "unusual" ports are blocked for belle7
* workaround is to open a tcp connection from remote client and get belle7 to 
  reply within that open connection  

Topology that works:

#. put broker on belle7::

   [blyth@belle7 ~]$ zmqdev-;zmqdev-broker
 
#. worker and client elsewhere (or on belle7 too):: 

   [blyth@cms02 ~]$ czmqdev-; ZMQ_BROKER_HOST=$(local-tag2ip N) czmqdev-worker
   delta:~ blyth$ zmqdev-; ZMQ_BROKER_HOST=$(local-tag2ip N) zmqdev-client


Connecting nuwa.py/Geant4 and g4daeview.py/Chroma
---------------------------------------------------

Configure nuwa.py/Geant4 as client sending REQ with ChromaPhotonList 
objects to the broker with *CSA_CLIENT_CONFIG* envvar::

     68 csa-nuwarun(){
     69 
     70    zmqdev-
     71    export CSA_CLIENT_CONFIG=$(zmqdev-broker-url)   
     72    nuwa.py -n 1 -m "fmcpmuon --chroma"
     73 
     74 }

Configure g4daeview.py/Chroma as worker receiving REP





EOU
}
zmqdev-dir(){ echo $(opticks-home)/zeromq/zmq ; }
zmqdev-bindir(){ echo $(local-base)/env/bin ; }

zmqdev-cd(){  cd $(zmqdev-dir); }
zmqdev-scd(){  cd $(zmqdev-dir); }
zmqdev-mate(){ mate $(zmqdev-dir) ; }

zmqdev-bin(){ echo $(zmqdev-bindir)/$1 ; }
zmqdev-cc(){
   local name=$1
   local bin=$(zmqdev-bin $name)
   mkdir -p $(dirname $bin)
   echo $msg compiling $bin 
   zeromq-
   cc $name.c -o $bin \
         -I$(zeromq-prefix)/include \
         -L$(zeromq-prefix)/lib -lzmq \
         -Wl,-rpath,$(zeromq-prefix)/lib
}

zmqdev-cc-build(){
  echo $msg building 
  zmqdev-scd
  local line
  ls -1 *.c | while read line ; do 
     local name=${line/.c}
     zmqdev-cc $name
  done
}


zmqdev-broker-info(){  cat << EOI

   zmqdev-broker-url          : $(zmqdev-broker-url)
   zmqdev-broker-url-frontend : $(zmqdev-broker-url-frontend)
   zmqdev-broker-url-backend  : $(zmqdev-broker-url-backend)

   zmqdev-broker-host         : $(zmqdev-broker-host)

EOI
}
zmqdev-frontend-port(){ echo 5001 ; }
zmqdev-backend-port(){  echo 5002 ; }
zmqdev-broker-tag(){ echo ${ZMQ_BROKER_TAG:-SELF} ; }
zmqdev-broker-host(){ local-tag2ip $(zmqdev-broker-tag) ; }

zmqdev-broker-url(){ zmqdev-broker-url-frontend ; }
zmqdev-broker-url-frontend(){ echo tcp://$(zmqdev-broker-host):$(zmqdev-frontend-port) ;}
zmqdev-broker-url-backend(){  echo tcp://$(zmqdev-broker-host):$(zmqdev-backend-port) ;}

zmqdev-broker-export(){
   export ZMQ_BROKER_URL_FRONTEND=$(zmqdev-broker-url-frontend)
   export ZMQ_BROKER_URL_BACKEND=$(zmqdev-broker-url-backend)
}

zmqdev-broker(){ FRONTEND=tcp://*:$(zmqdev-frontend-port) BACKEND=tcp://*:$(zmqdev-backend-port) $(zmqdev-bin zmq_broker) ; }
zmqdev-client(){ FRONTEND=$(zmqdev-broker-url-frontend) $(zmqdev-bin zmq_client) ; }
zmqdev-worker(){  BACKEND=$(zmqdev-broker-url-backend)  $(zmqdev-bin zmq_worker) ; }



zmqdev-tunnel-cmd(){
   local laddr=$1
   local raddr=$2
   local cmd="ssh -N -p 22 -L ${laddr}:${raddr} "  
   # -N no remote command
   # -f go to background
   echo $cmd
}
zmqdev-tunnel-raddr(){
  local var=${1:-ZMQ_BROKER_URL_BACKEND}
  local url=${!var}    
  local raddr=${url/tcp:\/\/}
  echo $raddr 
}
zmqdev-tunnel-laddr(){
  local laddr="127.0.0.1:$(available_port.py)" 
  echo $laddr
}

zmqdev-tunnel-open-backend(){  zmqdev-tunnel-open ZMQ_BROKER_URL_BACKEND ${1:-G5} ; }
zmqdev-tunnel-open-frontend(){ zmqdev-tunnel-open ZMQ_BROKER_URL_FRONTEND ${1:-G5} ; }
zmqdev-tunnel-open(){
  local msg="=== $FUNCNAME :" 
  local var=${1:-ZMQ_BROKER_URL_BACKEND} 
  local node=${2}

  [ -z "${!var}" ] && echo $msg WARNING : opening an ssh tunnel required envvar $var && return 0 

  local orig=${!var}

  local laddr=$(zmqdev-tunnel-laddr)
  local raddr=$(zmqdev-tunnel-raddr $var)
  local cmd="$(zmqdev-tunnel-cmd $laddr $raddr) $node"

  export $var=tcp://$laddr
  echo $msg var $var node $node cmd $cmd 

  local pid
  $cmd &
  pid=$!  # grab pid of the background process spawned by prior cmd

  export ZMQ_TUNNEL_PID=$pid 
  echo $msg ZMQ_TUNNEL_PID $ZMQ_TUNNEL_PID
  echo $msg modified $var from $orig to ${!var} to route via tunnel
}

zmqdev-tunnel-close(){
  local msg="=== $FUNCNAME :" 
  [ -z "$ZMQ_TUNNEL_PID" ] && echo $msg ZMQ_TUNNEL_PID is not defined && return 

  #ps aux $ZMQ_TUNNEL_PID
  echo $msg killing $ZMQ_TUNNEL_PID
  kill $ZMQ_TUNNEL_PID
}

zmqdev-tunnel-node-parse-cmdline(){
  local zmqtunnelnode=""
  local cmdline="$1" 
  if [ "${cmdline/--zmqtunnelnode}" != "${cmdline}" ]; then
     for arg in $cmdline ; do
         case $arg in
              --zmqtunnelnode=*)  zmqtunnelnode=${1#*=} ;;
         esac
     done
  fi
  echo $zmqtunnelnode
}






