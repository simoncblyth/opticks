ckm-viz-noshow-photon-first-steps
=======================================

::

    [blyth@localhost CerenkovMinimal]$ t ckm-viz
    ckm-viz is a function
    ckm-viz () 
    { 
        OPTICKS_KEY=$(ckm-key) $(ckm-dbg) OKTest --natural --envkey --xanalytic
    }



kludge workaround is to shift time from zero
------------------------------------------------

I recall something similar before where photons starting at time zero
are not rendered visibly : perhaps this is the same. Before, 
I simply shifted time from zero. 

This still works::


    calhost CerenkovMinimal]$ hg diff PrimaryGeneratorAction.cc
    diff -r 6e8585dcd56f examples/Geant4/CerenkovMinimal/PrimaryGeneratorAction.cc
    --- a/examples/Geant4/CerenkovMinimal/PrimaryGeneratorAction.cc Mon May 27 21:04:07 2019 +0800
    +++ b/examples/Geant4/CerenkovMinimal/PrimaryGeneratorAction.cc Mon May 27 21:43:38 2019 +0800
    @@ -21,7 +21,7 @@
         G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
         G4ParticleDefinition* particle = particleTable->FindParticle("e+");
         fParticleGun->SetParticleDefinition(particle);
    -    fParticleGun->SetParticleTime(0.0*CLHEP::ns);
    +    fParticleGun->SetParticleTime(0.1*CLHEP::ns);
         fParticleGun->SetParticlePosition(G4ThreeVector(0.0*CLHEP::cm,0.0*CLHEP::cm,0.0*CLHEP::cm));
         fParticleGun->SetParticleMomentumDirection(G4ThreeVector(1.,0.,0.));
         fParticleGun->SetParticleEnergy(0.8*MeV);   // few photons at ~0.7*MeV loads from ~ 0.8*MeV
    [blyth@localhost CerenkovMinimal]$ 




Permitting zero times as valid, seems to fix with no ill effects so far
-------------------------------------------------------------------------

glrap/gl/rec/geom.glsl::

     33 void main ()
     34 {
     35     uint seqhis = sel[0].x ;
     36     uint seqmat = sel[0].y ;
     37     if( RecSelect.x > 0 && RecSelect.x != seqhis )  return ;
     38     if( RecSelect.y > 0 && RecSelect.y != seqmat )  return ;
     39 
     40     uint photon_id = gl_PrimitiveIDIn/MAXREC ;
     41     if( PickPhoton.x > 0 && PickPhoton.y > 0 && PickPhoton.x != photon_id )  return ;
     42 
     43 
     44     vec4 p0 = gl_in[0].gl_Position  ;
     45     vec4 p1 = gl_in[1].gl_Position  ;
     46     float tc = Param.w / TimeDomain.y ;
     47 
     48     uint valid  = (uint(p0.w > 0.)  << 0) + (uint(p1.w > 0.) << 1) + (uint(p1.w > p0.w) << 2) ;
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         time exactly zero not regarded as valid   

     49     uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (uint(Pick.x == 0 || photon_id % Pick.x == 0) << 2) ;
     50     uint vselect = valid & select ;
     51 
     52 #incl fcolor.h
     53 
     54     if(vselect == 0x7) // both valid and straddling tc
     55     {
     56         vec3 pt = mix( vec3(p0), vec3(p1), (tc - p0.w)/(p1.w - p0.w) );
     57         gl_Position = ISNormModelViewProjection * vec4( pt, 1.0 ) ;
     58 
     59         if(NrmParam.z == 1)
     60         {
     61             float depth = ((gl_Position.z / gl_Position.w) + 1.0) * 0.5;
     62             if(depth < ScanParam.x || depth > ScanParam.y ) return ;
     63         }
     64 
     65 
     66         EmitVertex();
     67         EndPrimitive();
     68     }
     69     else if( valid == 0x7 && select == 0x5 )     // both valid and prior to tc
     70     {
     71         vec3 pt = vec3(p1) ;
     72         gl_Position = ISNormModelViewProjection * vec4( pt, 1.0 ) ;
     73 
     74         if(NrmParam.z == 1)
     75         {
     76             float depth = ((gl_Position.z / gl_Position.w) + 1.0) * 0.5;
     77             if(depth < ScanParam.x || depth > ScanParam.y ) return ;
     78         }
     79 
     80 
     81         EmitVertex();
     82         EndPrimitive();
     83     }
     84 
     85 }

