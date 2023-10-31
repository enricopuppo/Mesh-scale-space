ShapeCorrespGA_x86.exe $id1$ $id2$ $N$ $mode$ $doAdaptiveSampling$ $autoRemoveOutliers$

//first run in robust map computation mode via
ShapeCorrespGA_x86 1 2 100 1 1 1

//then run in desired map computation mode via
ShapeCorrespGA_x86 1 2 100 2 1 1

Notice that N=100 and doAdaptiveSampling=1=true and autoRemoveOutliers=1=true are discarded in the first run.

Outliers are detected automatically using the nice d_iso (isometric distortion) values produced by the genetic algorithm (or the adaptive sampling). So I recommend removing them, as it is part of the automatic process. In this case the map will be of size (N-n)x(N-n) where n << N (on average n=2 for N=100).

Timing: 1.off vs. 2.off with N=100 samples take about 4 seconds (sampling) + 8 seconds (GA) + 7 seconds (AS) on my 8GB 3.4GHz Windows PC.

Always used the same genetic parameters in my experiments with various types of meshes, e.g., POP_SIZE=N*10, MUTATION_RATE=XOVER_RATE=0.85. So, I do not take them as command line arguments. You may however locate these (and perhaps other) variables in my code and update them and re-compile the code, which is based fully on the standard C library, e.g., no external library linkage required for the compilation. It should compile easily on Unix and Mac as well.

Same code is used by two projects, 32-bit ShapeCorrespGA_x86 and 64-bit ShapeCorrespGA_x64. In this folder I left both the 32-bit and 64-bit executables, whose copies can be found in the respective Release subfolders.

Output format:
#Matches D_iso D_grd MaxGeoDistOnSource MaxGeoDistOnTarget
s_i	matching target index
s_{i+1}	matching target index
.
.

D_grd makes sense only if source vertex i matches with target vertex i for all i. In other words, ground-truth correspondence is available as i-i mathes.




Another example run on huge real-world scan data of ~190K vertices
//first run in robust map computation mode via
ShapeCorrespGA_x86 91 99 100 1 0 1

//then run in desired map computation mode via
ShapeCorrespGA_x86 91 99 100 2 0 1

Notice that the robust map itself is already sufficient as landmark corrrespondences of a generic registration algorithm. Going to a sparse map of size N=100 is kind of arbitrary.

Robust map (of automatic size 6) takes 6 seconds for the initial sampling plus 0.14 seconds for map computation.

Sparse map (of size N=100) takes 96 seconds for the initial sampling plus 20 seconds for map computation. No adaptive sampling in the command above but you could enable it if desired.

************
Please cite:
Y. Sahillioglu, A Genetic Isometric Shape Correspondence Algorithm with Adaptive Sampling, 2017.
************



--ysf
