G4Opticks-multievent-running-must-reset-to-avoid-genstep-accumulation
========================================================================

Issue reported by Hans
-------------------------

Concerning Opticks I found that each time in when I call getHits in the
EndOfEventAction I get the accumulated number of photon Hits not  the number
of hits in the event as I would have expected and there doesn't seem a way in
the API to reset the Array. So it would be nice to be able to reset or make
returning just the hits per event the default.::


    G4Opticks* ok = G4Opticks::GetOpticks();
    G4int eventid = event->GetEventID();
    int num_hits = ok->propagateOpticalPhotons(eventid);
    //std::vector<PhotonHit*> hitsVector;
    std::vector<G4VHit*> hitsVector;
    NPY<float>* hits = ok->getHits();



Reproduce this using G4OKTest by controlling the numbers of photons
---------------------------------------------------------------------

::

    OpticksEvent=INFO G4OKTest --torchtarget 3153


Symptom : number of photons keeps accumulating from event to event, eg with 100 photons in first and 200 in second::

    2020-11-01 11:33:43.664 ERROR [9040424] [G4OKTest::collectGensteps@226]  num_photons 200
    2020-11-01 11:33:43.665 INFO  [9040424] [OpticksEvent::resize@1108]  num_photons 300 num_records 3000 maxrec 10 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/-2
    2020-11-01 11:33:43.665 INFO  [9040424] [OpticksEvent::resize@1108]  num_photons 300 num_records 3000 maxrec 10 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2


Issue is not event resetting, but rather genstep resetting. Must remember to call G4Opticks::reset() to reset genstep collection.




