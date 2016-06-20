clear functions;

% Point to the ArrayFire environment
afpath = getenv('AF_PATH');
includepath = [afpath '\include'];
libpath = [afpath '\lib'];

% Specify the library for compilation
%aflib = 'afcpu.lib';
aflib = 'afcuda.lib';
%aflib = 'afopencl.lib';

% Compile ArrayFire dependent sources
mex(['-I',includepath],['-L',libpath],['-l',aflib],'src\mexSolveDCA.cpp');
mex src/im2colrand.c
%mex src/mexEncodeTIP.c src/cabacEnc.c src/cabacLUT.c
%mex src/mexDecodeTIP.c src/cabacDec.c src/cabacLUT.c
