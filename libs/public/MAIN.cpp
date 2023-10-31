//Author: Yusuf Sahillioglu
//Aim: Implementation of my paper: A Genetic Isometric Shape Correspondence Algorithm with Adaptive Sampling
//this code establishes optimal correspondence b/w 2 isometric shapes using genetic algorithm (and then improves via adaptive sampling)

//#include <time.h> for timing only; disabling for full portability

#include "GA.h"

int main(int argc, char ** argv)
{
	//without srand the program would have generated the same number each time we ran it because the generator would have been seeded with the same default value each time; now it'll be seeded w/ sys time and generate different numbers
//	srand( (unsigned) time(NULL) );

	bool bruteForceSolution = false, //make it true to see the slow (intractable for CHROM_LEN>10) but optimal solution as a baseline comparator 
		 computeDenseMap = false, //set this to true to compute the dense map by interpolation of the GA-based coarse map (a primitive scheme that has plenty of room for improvement)
		 removeOutliers = true, //due to coarse sampling some few samples may be forced to be matched with irrelevant ones; remove such outliers if this is true
		 doAdaptiveSampling = true; //set it to true to apply my adaptive sampling on mesh2 to improve the correspondences found by the GA
		 //insertExistingMaps = false; //insert pre-computed maps to the initial population (to improve population); always false 'cos not works as explained in my findings.txt
#ifdef USE_GRD_DISTORTION //hypothetical mode where i use the predefined ground-truth correspondences in fitness evaluation; this is just to see the best result possible in terms of grd-truth; note that GA minimizes isometric distortion, which hopefully minimizes this one
	//removeOutliers = false; //see the ground-truth result with no removal
#endif

	//overwrite defaulta above by getting 6 cmd line arguments
	if (argc != 7)
	{
		cout << "\nwhere're my 6 arguments?\n" <<
			    "ShapeCorrespGA_x64.exe $id1$ $id2$ $N$ $mode$ $doAdaptiveSampling$ $autoRemoveOutliers$\n\n" << 
				"$id1$: integer id of the off-formatted source mesh in \\input folder\n\n" <<
				"$id1$: integer id of the off-formatted target mesh in \\input folder\n\n" <<
				"$N$: number of samples to be matched (used in mode=2 only)\n\n" <<
				"$mode$: 1 to compute initial robust map M_0 via genetic algo (GA), 2 to compute the map M_1 of size N via GA\n\n" <<
				"$doAdaptiveSampling$: 1 to enable adaptive sampling after GA (used in mode=2 only), otherwise disable it\n\n" <<
				"$autoRemoveOutliers$: 1 to automatically detect and remove outlier matches from the final map, otherwise keep the outliers\n\n";
		exit(0);
	}

	int id1 = atoi(argv[1]), id2 = atoi(argv[2]); //[0] is the name of the program
	Mesh* mesh1 = new Mesh(false, id1), * mesh2 = new Mesh(true, id2);
	char fName[250], fName2[250];
	
	sprintf(fName, "input\\%d.off", id1);
	sprintf(fName2, "input\\%d.off", id2);
	
	mesh1->loadOff(fName);
	mesh2->loadOff(fName2);

	//find solution
//	double start_time = get_time();
	bruteForceSolution = false;
	int N = atoi(argv[3]), mode = (atoi(argv[4]) == 1 ? COMPUTE_ROBUST_LIST : GA_WITH_ROBUST_LIST);
	removeOutliers = (atoi(argv[6]) == 1);
	//mode = GA_WITH_ROBUST_LIST;//GA_WITHOUT_ROBUST_LIST;
	//if (mode == COMPUTE_ROBUST_LIST) doAdaptiveSampling = false; //never do adaptive sampling in this mode [no adaptiveSampling makes sense here too]
	(new GA(mesh1, mesh2, mode, N))->go(bruteForceSolution, removeOutliers); //new genetic algorithm paper
	//(new GA(mesh1, mesh2, mode))->go2(); //new genetic algorithm paper [generally slight improvement on diso (.032-->.031) but not so important; dgrd not improves at all]
#ifdef VERBOSE
	//cout << "========> " << get_time()-start_time << " secs for GA init + GA itself computations\n\n"; //excludes load/export times (as they are not computations)
#endif
	if (atoi(argv[5]) == 1 && mode != COMPUTE_ROBUST_LIST) //if (doAdaptiveSampling)
	{
//		start_time = get_time();
		/*bool keyboardSampling = false; used for debugging/idea developing; not needed anymore
		mesh1->adaptiveSampling(mesh2, keyboardSampling); only 1 ordering (e.g. worst to best) can be tried unlike the GA's multiple orderings
		mesh1->adaptiveSampling2(mesh2); sucks 'cos new startVertexes all give the same sampling; i'll tie this to genetic algorithm*/
		mesh1->adaptiveSampling3(mesh2, removeOutliers); //coordinate descent algorithm to minimize the adaptive sampling objective function
#ifdef VERBOSE
//		cout << "========> " << get_time()-start_time << " secs for adaptiveSampling computations\n\n";
#endif
	}

	if (computeDenseMap)
	{
//		start_time = get_time();
		//interpolate the cool coarse map in corresp[] (resultToFile() created corresp[]) into a full dense map and fprint it
		mesh1->interpolateCoarseMap(mesh2);
#ifdef VERBOSE
//		cout << "========> " << get_time()-start_time << " secs for interpolation computations\n\n";
#endif
	}

	return 0;
}
