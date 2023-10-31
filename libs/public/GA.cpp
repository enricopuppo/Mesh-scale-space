#include "GA.h"

GA::GA(Mesh* m1, Mesh* m2, int mode, int N)
{
//	double start_time = get_time();
	
	//genetic algorithm initializations
	MODE = mode;	
	if (MODE == COMPUTE_ROBUST_LIST)
	{
		CHROM_LEN = 10;
		POP_SIZE = 200;
		INIT_POP_METHOD = RND_INIT; //always use random init for this mode
		robustListInUse = false; //there is no robustList to use; i'll instead compute it in this mode
		SWAPPER = RANDOM_SWAP;//RANK_BASED_SWAP;//RANDOM_SWAP; //RANK_BASED_SWAP
	}
	else if (MODE == GA_WITH_ROBUST_LIST)
	{
		CHROM_LEN = N;//100;
//CHROM_LEN = 14;//15
		POP_SIZE = N*10;//1000;
		INIT_POP_METHOD = GV_BASED_INIT; //AGD_BASED_INIT
		robustListInUse = true; //use a robust traversal list in genetic algo's getIsometricDistortionRL() calls; robust list obtained in COMPUTE_ROBUST_LIST mode in this project but any robust list (of sufficient size to prevent sym flips) is ok
		SWAPPER = GV_BASED_SWAP;

/*CHROM_LEN = 100;
POP_SIZE = 2000;//1000; 2000 is as accurate as 1000 so never use it and save big time
INIT_POP_METHOD = GV_BASED_INIT;
robustListInUse = true; //use a robust traversal list in genetic algo's getIsometricDistortionRL() calls; robust list obtained in COMPUTE_ROBUST_LIST mode in this project but any robust list (of sufficient size to prevent sym flips) is ok
SWAPPER = GV_BASED_SWAP;//GV_BASED_SWAP;//RANK_BASED_SWAP;*/
	}
	else if (MODE == GA_WITHOUT_ROBUST_LIST)
	{
		CHROM_LEN = N;//100;
		POP_SIZE = N*10;//2000;//1000;
		INIT_POP_METHOD = AGD_BASED_INIT; //cannot do GV_BASED_INIT here 'cos there's no GV
		robustListInUse = false;
		SWAPPER = RANK_BASED_SWAP; //cannot do GV_BASED_SWAP here 'cos there's no GV
	}
	else if (MODE == GA_WITH_ROBUST_LIST_HIGHRESO)
	{
		CHROM_LEN = 1000;
		POP_SIZE = 20000;
		INIT_POP_METHOD = GV_BASED_INIT; //AGD_BASED_INIT
		robustListInUse = true;
		SWAPPER = GV_BASED_SWAP;
	}
	else
	{
		cout << "select a valid execution mode\n";
		exit(0);
	}
#ifdef VERBOSE
	cout << "good-badpart sizes " << GOODPART_SIZE << "-" << BADPART_SIZE << " now set properly\n"; //thanks to POP_SIZE inits above
#else
	cout << "sampling..";
#endif
	nMuts = nXovs = 0;
	population = new int*[POP_SIZE];
	initChromosome = new int[CHROM_LEN];
	populationFitnesses = new float[POP_SIZE];
/*if (MODE == COMPUTE_ROBUST_LIST)
{
#define X
}cannot do this 'cos then X is always defined regardless of if condition, e.g., even if MODE == GA_WITH_ROBUST_LIST*/

	//mesh initializations
	mesh1 = m1;
	mesh2 = m2;
	if (robustListInUse)
	{
#ifdef LMDS_INIT
		mesh1->fillRobustTraversalList("output\\robustList\\0-6intersect--em-refined.dat"); //40 samples
#else
		//mesh1->fillRobustTraversalList("output\\robustList\\0-6intersect--em-refined7.dat"); //7 samples
		char fName[250]; //sprintf(fName, "output\\robustList\\%d-%d.dat", m1->id, m2->id);
		sprintf(fName, "input\\robustMaps\\%d-%d.dat", m1->id, m2->id);
		mesh1->fillRobustTraversalList(fName);
#endif
		mesh2->robustList = mesh1->robustList;
		if (CHROM_LEN >= 1000)
		{
			mesh1->computeSamples(CHROM_LEN); //samples from fillRobustTraversalList() added directly and then Euclidean dists used to fill the rest (typically CHROM_LEN >> 100 here)
			mesh2->computeSamples(CHROM_LEN);	
		}
		else
		{
			mesh1->computeEvenlySpacedSamples(CHROM_LEN); //typically CHROM_LEN = 100 here
			mesh2->computeEvenlySpacedSamples(CHROM_LEN);
		}
	}
	else
	{
		if (CHROM_LEN >= 1000)
		{
			cout << "Many dijkstras inefficient when dealing with " << CHROM_LEN << " samples; reduce samples or make robustListInUse = true\n";
			exit(0);
		}
		if (MODE == COMPUTE_ROBUST_LIST)
		{
			mesh1->computeEvenlySpacedSamples(CHROM_LEN, true); //typically CHROM_LEN = 10 here but thanks to earlyBreak it will compute 5-6 samples
			CHROM_LEN = (int) mesh1->samples.size();
			mesh2->computeEvenlySpacedSamples(CHROM_LEN); //to guarantee equal-size samples in mesh1 & mesh2 use the new CHROM_LEN here
		}
		else
		{
			mesh1->computeEvenlySpacedSamples(CHROM_LEN); //typically CHROM_LEN = 100 here
			mesh2->computeEvenlySpacedSamples(CHROM_LEN);
		}
	}
	for (int i = 0; i < POP_SIZE; i++)
		population[i] = new int[CHROM_LEN];
	initChromosome = new int[CHROM_LEN];
	for (int i = 0; i < CHROM_LEN; i++)
		initChromosome[i] = i;

	//do agd computations even if INIT_POP_METHOD == RND_INT 'cos swapMutation may require agdOrder values
	mesh1->computeAGD(robustListInUse); //also sorts vertices w.r.t AGD values
	mesh2->computeAGD(robustListInUse); //also sorts vertices w.r.t AGD values
	//similarly windowSize will be used in swapMutation (even if INIT_POP_METHOD == RND_INT may use it)
	windowSize = CHROM_LEN / 10; //CHROM_LEN=100 -> windowSize=10, 1000 -> 100, and so on; for CHROM_LEN=100, 40vs48, 37vs47 ranks (see rankings.png) are still good matches; so use windowSize=10)
	windowSize = (windowSize == 0 ? 3 : windowSize);
	if (MODE != COMPUTE_ROBUST_LIST && //no robustList to compute the gv descriptors
		MODE != GA_WITHOUT_ROBUST_LIST) //this mode never employs GV
	{
		//geodesic vectors will be useful in GV_BASED_INIT and/or swapMutation
		mesh1->computeGV();
		mesh2->computeGV();
	}
	for (int i = 0; i < (int) mesh1->robustList.size(); i += 2)
		mesh1->verts[ mesh1->robustList[i] ]->robust = mesh2->verts[ mesh1->robustList[i+1] ]->robust = true;
	
	computeInitialMatchCandidates();

	if (N_XOVS <= 0)
	{
		cout << "even if SELECT_BEST_OF_N_XOVS is undefined (hence N_XOVS seems to be useless), keep N_XOVS positive at all times\n";
		exit(0);
	}

	//initializations for evolvePopulation*() are done once here (hence no multiple allocs and deletes in those functions)
	newPopulation = new int*[BADPART_SIZE];
	childChromo = new int[CHROM_LEN];
	bestChildChromo = new int[CHROM_LEN];
	tmpPopulation = new int*[POP_SIZE];
	for (int c = 0; c < POP_SIZE; c++)
		tmpPopulation[c] = new int[CHROM_LEN];
	for (int i = 0; i < BADPART_SIZE; i++)
		newPopulation[i] = new int[CHROM_LEN];
	//sort the population[][] w.r.t. fitness of each member; sorted result is written back to population[][]
	sortedIdxs = new int[POP_SIZE];
	fitnessVals = new float[POP_SIZE];
	tmpFitnesses = new float[POP_SIZE];

#ifdef VERBOSE
//	cout << "========> " << get_time()-start_time << " secs for GA initialization (sampling part is dominant here)\n\n"; //POPSIZE=1000,CHROMLEN=100, 4 secs (very fast)
#else
	cout << "done!\n";
#endif
}

void GA::computeInitialMatchCandidates()
{
	//fills the initMatchCandids[] to be used in initPopulation()

	if (INIT_POP_METHOD == AGD_BASED_INIT)
	{
		forceRobustMatchesInInit = true;//false
		float candidTotal = 0.0f;
		for (int j = 0; j < CHROM_LEN; j++)
		{
			mesh1->verts[ mesh1->samples[j] ]->initMatchCandids.clear(); //in case this function is called 2+ times (go2() case)
			//make the robustList samples always select their good matchIdx in initPopulation() by making their initMatchCandids.size=1
			if (forceRobustMatchesInInit && mesh1->verts[ mesh1->samples[j] ]->robust)
			{
				int k;
				for (k = 0; k < CHROM_LEN; k++)
					if (mesh2->samples[k] == mesh1->verts[ mesh1->samples[j] ]->matchIdx)
					{
						mesh1->verts[ mesh1->samples[j] ]->initMatchCandids.push_back(k); //sample j (mesh1) and k (mesh2) will be in initial correspondence
						candidTotal += (int) mesh1->verts[ mesh1->samples[j] ]->initMatchCandids.size();
						break;
					}
				if (k == CHROM_LEN) //break; not executed
				{
					//cout << "AGD_BASED_INIT: no match found; impossible\n"; no it's possible only if fillRobustTraversalList() uses a file computed by adaptiveSampling (mesh2 samples adaptively changed so not compatible w/ FPS samples here)
					//exit(0);
					//; still initMatchCandids empty so not do continue; and fill initMatchCandids below
				}
				else
					continue;
			}

			//initial match candidates for this jth sample must be in windowSize vicinity in mesh2 w.r.t. agdOrder values
			int start = mesh1->verts[ mesh1->samples[j] ]->agdOrder - windowSize/2,
				end = mesh1->verts[ mesh1->samples[j] ]->agdOrder + windowSize/2;
			if (start < 0)
				start = 0;
			if (end >= CHROM_LEN)
				end = CHROM_LEN-1;
			for (int k = 0; k < CHROM_LEN; k++)
				if (mesh2->verts[ mesh2->samples[k] ]->agdOrder >= start && mesh2->verts[ mesh2->samples[ k ] ]->agdOrder <= end)
					mesh1->verts[ mesh1->samples[j] ]->initMatchCandids.push_back(k);
			candidTotal += (int) mesh1->verts[ mesh1->samples[j] ]->initMatchCandids.size();
		}
#ifdef VERBOSE
		cout << "AGD_BASED_INIT: avg # candids per sample = " << candidTotal / CHROM_LEN << endl;
#endif
		mesh1->windowSize = mesh2->windowSize = windowSize; //meshes need this info for getRankDistortion()
	}
	else if (INIT_POP_METHOD == GV_BASED_INIT)
	{
		forceRobustMatchesInInit = true;//false
		float candidTotal = 0.0f;
		for (int j = 0; j < CHROM_LEN; j++)
		{
			mesh1->verts[ mesh1->samples[j] ]->initMatchCandids.clear(); //in case this function is called 2+ times (go2() case)
			//make the robustList samples always select their good matchIdx in initPopulation() by making their initMatchCandids.size=1
			if (forceRobustMatchesInInit && mesh1->verts[ mesh1->samples[j] ]->robust)
			{
				int k;
				for (k = 0; k < CHROM_LEN; k++)
					if (mesh2->samples[k] == mesh1->verts[ mesh1->samples[j] ]->matchIdx)
					{
						mesh1->verts[ mesh1->samples[j] ]->initMatchCandids.push_back(k); //sample j (mesh1) and k (mesh2) will be in initial correspondence
						candidTotal += (int) mesh1->verts[ mesh1->samples[j] ]->initMatchCandids.size();
						break;
					}
				if (k == CHROM_LEN) //break; not executed
				{
					//cout << "GV_BASED_INIT: no match found; impossible\n"; no it's possible only if fillRobustTraversalList() uses a file computed by adaptiveSampling (mesh2 samples adaptively changed so not compatible w/ FPS samples here)
					//exit(0);
					//; still initMatchCandids empty so not do continue; and fill initMatchCandids below
				}
				else
					continue;
			}

			//initial match candidates for this jth sample must have compatible corresponding rows in gv's; by compatible i allow distance difference upto 0.25 (1 is the max distance, e.g. hand to toe, so 0.25 allows errors upto toe to knee)
			float maxErr = 0.125f; //toe-to-knee normalized distance is 0.25 but since i'm taking the subtraction below i should use the half value
			do
			{
				int gvSize =  (int) mesh1->verts[ mesh1->samples[j] ]->gv.size(); //gvSize'll be same (and equal to robustList.size/2) for all samples
				if (gvSize == 0 && j == 0)
					cout << "\nWARNING: initial chromosomes will be random since GVs are empty\n\n";
				for (int k = 0; k < CHROM_LEN; k++)
				{
					bool goodCandid = true;
					for (int i = 0; i < gvSize; i++)
						if (fabs(mesh1->verts[ mesh1->samples[j] ]->gv[i] - mesh2->verts[ mesh2->samples[k] ]->gv[i]) > maxErr) //check the compatibility of the corresponding i'th rows in gv's
						{
							goodCandid = false;
							break;
						}
					if (goodCandid)
						mesh1->verts[ mesh1->samples[j] ]->initMatchCandids.push_back(k);
				}
				candidTotal += (int) mesh1->verts[ mesh1->samples[j] ]->initMatchCandids.size();
				maxErr += 0.05f; //in case initMatchCandids still empty (very unlikely) re-do the assignment w/ a more relaxed maxErr 
			} while (mesh1->verts[ mesh1->samples[j] ]->initMatchCandids.empty());
		}
#ifdef VERBOSE
		cout << "GV_BASED_INIT: avg # candids per sample = " << candidTotal / CHROM_LEN << endl;
#endif
		mesh1->windowSize = mesh2->windowSize = windowSize; //meshes need this info for getRankDistortion()
	}
	else if (INIT_POP_METHOD == RND_INIT)
		; //do nothing as initMatchCandids[]'ll never be used in this RND_INIT mode
	else
	{
		cout << "undefined INIT_POP_METHOD\n";
		exit(0);
	}
}
inline int randoi(int a, int b)
{
	//generate a random int between a (inclusive) and b > a (exclusive); that is [a,b)
	return (rand() % (b-a)) + a;
}
inline int randoi(int b)
{
	//generate a random int between 0 (inclusive) and b > 0 (exclusive); that is [0,b); saves 1 subtraction and 1 addition operation compared to generic randoi(int a, int b)
	return (rand() % b);
}
inline float randof()
{
	//generate a random real number between 0 and 1, inclusive [0,1]
	return (float) rand() / RAND_MAX;
}
inline float randof(float fMin, float fMax)
{
	//generate a random real number between fMin and fMax, inclusive [fMin,fMax]
	float f = (float) rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

void GA::initPopulation(bool insertExistingMaps)
{
	//initializes the population matrix

#ifdef VERBOSE
#ifndef MULTIPLE_REINIT
	cout << "initing population of size " << POP_SIZE << "..\n"; //save print space by skipping this
#endif
#endif

	if (GOODPART_SIZE < 2 || GOODPART_SIZE > POP_SIZE)
	{
		cout << "invalid GOODPART_SIZE " << GOODPART_SIZE << endl; //at least 2 good members are required; also it must not exceed all population size
		exit(0);
	}

#ifdef LMDS_INIT
	mesh1->initialLMDSMatching(mesh2); //this will update matchIdx values but no worries 'cos initMatchCandids already filled and the ultimate matchIdx values will be set in the end of the go()
	//initialize one special chromosome, say population[0][*], to the LMDS matching explicitly
	for (int j = 0; j < CHROM_LEN; j++)
	{
		population[0][j] = mesh1->verts[ mesh1->samples[j] ]->matchIdx; //matchIdx is normally an idx to verts[] but the called initialLMDSMatching() sets it as an idx to samples for this time only

int k=mesh1->verts[ mesh1->samples[j] ]->matchIdx;//cout << j << "\t\t" << k << endl;cout << mesh1->samples[ j ] << " - " << mesh2->samples[ k ] << "\t\t" << mesh1->verts[ mesh1->samples[j] ]->agdOrder << " ==== " << mesh2->verts[ mesh2->samples[ k ] ]->agdOrder << endl;

//cout << mesh1->samples[ j ] << " - " << mesh2->samples[ population[0][j] ] << "\t\t" << mesh1->verts[ mesh1->samples[j] ]->agdOrder << " == " << mesh2->verts[ mesh2->samples[ population[0][j] ] ]->agdOrder << endl << endl;
//if (j==5)exit(0);
	}//*/
#endif

	if (INIT_POP_METHOD == AGD_BASED_INIT || INIT_POP_METHOD == GV_BASED_INIT) //AGD-based or GV-based initialization below should include very good maps in the initial population despite its low size (POP_SIZE (300) vs. all-maps (100! = 9 x 10^157))
																			   //AGD-based not give very good maps: min ground-truth distortion is ~0.25; very good maps have ~0.05
																			   //GV-based gives good maps: min ground-truth distortion is ~0.1; very good maps have ~0.05; so use GV-based init
	{
#ifdef VERBOSE
#ifndef MULTIPLE_REINIT
		if (INIT_POP_METHOD == AGD_BASED_INIT)
			cout << "\nagdOrder-ranking-based initial population in use\n";
		else
			cout << "\ngeodesicvector-based initial population in use\n"; //save print space by skipping this
#endif
#endif
		vector< int > candidsCpy;
		int windowSizeEnlarged = windowSize * 10; //due to randomness below, i may not check all 8 candidates at the first 8 iterations (some iters will repeat the same random index); so give at least 8*10 chances and if still no match is found then i'll assign an arbitrary value to this chromosome
		float nRandomTotal = 0.0f;
		for (int i = 0; i < POP_SIZE; i++)
		{
			for (int s = 0; s < CHROM_LEN; s++) //CHROM_LEN is same as (int) mesh2->samples.size()
			{
				mesh2->verts[ mesh2->samples[s] ]->processed = false; //initialize this to prevent duplications in chromosomes of population; true means already used in this chromosome
				population[i][s] = -1;
			}
			//initialize the i'th row of population based on initMatchCandids
			for (int j = 0; j < CHROM_LEN; j++)
			{
				candidsCpy = mesh1->verts[ mesh1->samples[j] ]->initMatchCandids; //values in candidsCpy will be made -1 below and it won't affect the original initMatchCandids
				int nTry = 0, idx, nCandids = (int) candidsCpy.size(); //always equal to 1 for robust samples in robustList
				bool randomAssign = false;
				do
				{
					idx = randoi(nCandids);
					if (nTry++ > windowSizeEnlarged)
					{
						randomAssign = true;
						break;
					}
				} while (mesh2->verts[ mesh2->samples[ candidsCpy[idx] ] ]->processed);
//} while (candidsCpy[idx] == -1);
				if (! randomAssign)
				{
					population[i][j] = candidsCpy[idx];
					mesh2->verts[ mesh2->samples[ candidsCpy[idx] ] ]->processed = true; //upcoming overlapped j's cannot select this idx
				}
//candidsCpy[idx] = -1; //upcoming overlapped j's cannot select this idx
			}
			//assign random and unused mesh2 vertices to unmatched/uninitialized chromosome values, if any
			int nRandomAssigns = 0;
			for (int j = 0; j < CHROM_LEN; j++)
				if (population[i][j] == -1) //break happened above and left this entry as -1; give it an assignment
				{
					//go through all mesh2 samples and assign the first available one to this j
					for (int s = 0; s < CHROM_LEN; s++) 
						if (! mesh2->verts[ mesh2->samples[ s ] ]->processed)
						{
							population[i][j] = s;
							mesh2->verts[ mesh2->samples[ s ] ]->processed = true; //this sample is assigned so never use it for the upcoming iterations
							nRandomAssigns++;
							if (robustListInUse && forceRobustMatchesInInit && mesh1->verts[ mesh1->samples[j] ]->robust)
							{
								cout << "impossible for a robustList sample to use a random assignment " << (int) mesh1->verts[ mesh1->samples[j] ]->initMatchCandids.size() << endl; //'cos robustList sample must have initMatchCandids.size=1 and that candid is selected nonrandomly
								exit(0);
							}
							break;
						}
				}
			if (nRandomAssigns > 0)
			{
//				cout << i << "-" << nRandomAssigns << "\t"; //~5/250 random assingments occur on average; cout << nRandomAssigns << " random assignments occured for population " << i << endl;
				nRandomTotal += nRandomAssigns;
			}
		}
#ifdef VERBOSE
		cout << nRandomTotal/POP_SIZE << " / " << CHROM_LEN << " random assignments\n";// on average\n";
#endif
		/////////////////////// good maps being added to population ////////////////////////////
/*		if (INIT_POP_METHOD == AGD_BASED_INIT)
		{
 this turns out to be a bad match so not use it, i.e. grd=0.337 for 100 samples (good values are below 0.1)
			//initialize one special chromosome, say population[1][*], to the ~best matching explicitly, which is the one that assigns the i'th agdOrder of mesh1 to the i'th agdOrder of mesh2 without any randomness
			for (int j = 0; j < CHROM_LEN; j++)
			{
				for (int k = 0; k < CHROM_LEN; k++)
					if (mesh1->verts[ mesh1->samples[j] ]->agdOrder == mesh2->verts[ mesh2->samples[k] ]->agdOrder)
					{
						population[1][j] = k;

//cout << j << "\t\t" << k << endl;cout << mesh1->samples[ j ] << " - " << mesh2->samples[ k ] << "\t\t" << mesh1->verts[ mesh1->samples[j] ]->agdOrder << " == " << mesh2->verts[ mesh2->samples[ k ] ]->agdOrder << endl;
						break;
					}
//cout << mesh1->samples[ j ] << " - " << mesh2->samples[ population[1][j] ] << "\t\t" << mesh1->verts[ mesh1->samples[j] ]->agdOrder << " == " << mesh2->verts[ mesh2->samples[ population[1][j] ] ]->agdOrder << endl << endl;
//if (j==5)exit(0);
			}
		}//*/
		if (insertExistingMaps)
		{
			//fread, if exists, the NxN map created by GA algo w/ a poorer init (so thanks to this/these good GA map(s) in the initial population the current GA will perform even better)
			char fName[250]; sprintf(fName, "output\\populationMaps\\%d-%d.dat", mesh1->id, mesh2->id);
			FILE* pmPtr;
			if (pmPtr = fopen(fName, "r"))
			{
				int n, v1, v2, i = 0;
				while (fscanf(pmPtr, "%d\n", &n) != EOF) //go till the end of file
				{				
					for (int j = 0; j < CHROM_LEN; j++)
					{
						fscanf(pmPtr, "%d\t%d\n", &v1, &v2);
					
						//population[i][j] is a value in [0, CHROM_LEN); so i need to find the idx to v2
						int idx;
						for (idx = 0; idx < CHROM_LEN; idx++) 
							if (mesh2->samples[idx] == v2)
							{
								population[i][j] = idx;
								break;
							}
						if (idx == CHROM_LEN || n != CHROM_LEN || v1 != mesh1->samples[j]) //1st condition: break; not executed above (population[][] not filled)
						{
							cout << "incompatible map in file: " << fName << endl;
							exit(0);
						}
					}
					i++;
				}
#ifdef VERBOSE
				cout << i << " good previous GA maps fread into population's first " << i << " rows\n";
#endif
			}
		}

		//add MDS and EM maps to the next 2 entries of population, namely population[2][*] and population[3][*] (no greedy map 'cos it is not bijective)
//for a generic genetic algorithm do not add these maps of fixed sizes; they're size 100 and used to create robustList; now i may want to evolve chromosomes of larger sizes using robustList; so i cannot use these old 100-size maps
//besides i don't even use MDS/greedy/EM at all in my new simple initialization; so i should not have access to these old result files
/*		/////////////////// MDS ///////////////////////
		FILE* fPtr = fopen("output\\0-6mds.dat", "r");
		int mapSize, v1, v2;
		float b, c;
		if (fPtr)
		{
			fscanf(fPtr, "%d %f %f\n", &mapSize, &b, &c);
			if (mapSize != CHROM_LEN)
			{
				cout << "mapSize != CHROM_LEN will violate bijection; can be prevented by some heuristic and coding but anyway\n";
				exit(0);
			}
			for (int j = 0; j < mapSize; j++)
			{
				fscanf(fPtr, "%d\t%d\n", &v1, &v2);
				//find the samples[] idx of v1 to decide the location in population[2][?]
				int s1 = -1, s2 = -1;
				for (int s = 0; s < CHROM_LEN; s++)
					if (v1 == mesh1->samples[s])
					{
						s1 = s;
						break;
					}
				//find the samples[] idx of v2 to decide the rhs of population[2][s1]
				for (int s = 0; s < CHROM_LEN; s++)
					if (v2 == mesh2->samples[s])
					{
						s2 = s;
						break;
					}
				if (s1 == -1 || s2 == -1)
				{
					cout << "impossible!!!!\n"; //happens if robustListInUse=true 'cos in that 'case i use euclidean-based computeSamples() instead of the geodesic-based computeEvenlySpacedSamples() (MDS uses geodesic-based)
					exit(0);
				}
				population[2][s1] = s2;
			}
			fclose(fPtr);
		}
		/////////////////// MDS ends ///////////////////////*/
		/*/////////////////// EM ///////////////////////
		fPtr = fopen("output\\0-6emoutliersnotremoved.dat", "r");
		if (fPtr)
		{
			fscanf(fPtr, "%d %f %f\n", &mapSize, &b, &c);
			if (mapSize != CHROM_LEN)
			{
				cout << "mapSize != CHROM_LEN will violate bijection; can be prevented by some heuristic and coding but anyway\n";
				exit(0);
			}
			for (int j = 0; j < mapSize; j++)
			{
				fscanf(fPtr, "%d\t%d\n", &v1, &v2);
				//find the samples[] idx of v1 to decide the location in population[3][?]
				int s1 = -1, s2 = -1;
				for (int s = 0; s < CHROM_LEN; s++)
					if (v1 == mesh1->samples[s])
					{
						s1 = s;
						break;
					}
				//find the samples[] idx of v2 to decide the rhs of population[3][s1]
				for (int s = 0; s < CHROM_LEN; s++)
					if (v2 == mesh2->samples[s])
					{
						s2 = s;
						break;
					}
				if (s1 == -1 || s2 == -1)
				{
					cout << "impossible2!!!!\n";
					exit(0);
				}
				population[3][s1] = s2;
			}
			fclose(fPtr);
		}
		/////////////////// EM ends ///////////////////////*/
		/////////////////////// good maps being added to population ends ////////////////////////////
	}
	else if (INIT_POP_METHOD == RND_INIT) //random initial maps; easy but population is too bad that it may not evolve to a good result
	{
#ifdef VERBOSE
		cout << "\nrandom initial population in use\n";
#endif
		for (int i = 0; i < POP_SIZE; i++)
		{
			//initialize the i'th row of population to initChromosome
			for (int j = CHROM_LEN - 1; j >= 0; j--)
				population[i][j] = initChromosome[j];
			//shuffle the i'th row of population as follows
			for (int j = CHROM_LEN - 1; j > 0; j--)
			{
				int idx = randoi(j+1), //integer in [0, j+1) = [0,j] interval
				//swap idx w/ j'th element
				a = population[i][idx];
				population[i][idx] = population[i][j];
				population[i][j] = a;
			}
		}
	}
	else
	{
		cout << "undefined INIT_POP_METHOD\n";
		exit(0);
	}

	//////////////// see ground-truth distortions of the initial population (just for debugging) ////////////////
	vector< int > corresp;
	float minGrd = INF, maxGrd = -INF, totGrd = 0.0f;
	int minGrdI, maxGrdI;
	for (int i = 0; i < POP_SIZE; i++)
	{
		corresp.clear();
		for (int j = 0; j < CHROM_LEN; j++)
		{
			corresp.push_back(mesh1->samples[j]);
			corresp.push_back(mesh2->samples[ population[i][j] ]);
//if (i == 8 || i == 18) cout << i << "\t" << mesh1->samples[j] << "-" << mesh2->samples[ population[i][j] ] << endl; //check some population members
		}
		float grd = mesh1->getGroundDistortion(mesh2, corresp); //not using getGroundDistortionFull() 'cos it will allocate dSpanning for all samples (upto 1000) redundantly for the actual algo
		if (grd < minGrd)
		{
			minGrd = grd;
			minGrdI = i;
		}
		if (grd > maxGrd)
		{
			maxGrd = grd;
			maxGrdI = i;
		}
		totGrd += grd;
#ifdef VERBOSE
#ifdef LMDS_INIT
if (i == 0) cout << "LMDS map gives grd distortion (for robustList" << (int) mesh1->robustList.size()/2 << " only): " << grd << endl; //population[0][*] is specially set above
#endif
#endif
//if (i == 1) cout << "agdOrder[i]->mesh2.agdOrder[i] map (no initMatchCandids) gives grd distortion (for robustList only): " << grd << endl; //this turns out to be a bad match so not use it
//break;//see population[0][*], which is a special one set above
	}
	if (robustListInUse)
	{
		//if i called computeEvenlySpacedSamples(), i.e., CHROM_LEN<=100, then ground distortion above considered all samples; if not (= computeSamples() called), it considered only robustList samples
		int nGrd = 0; //tell the user how many samples are considered for grd distortion computation above
		for (int s = 0; s < CHROM_LEN; s++)
			if (mesh1->verts[ corresp[2*s] ]->dSpanningUnn)
				nGrd++;
#ifdef VERBOSE
		cout << "min/avg/max grd distortion (for " << nGrd << " samples) = " << minGrd << " (" << minGrdI << ") / " << totGrd/POP_SIZE << " / " << maxGrd << endl;
#endif
	}
#ifdef VERBOSE
	else
		cout << "min/avg/max grd distortion (for all samples) = " << minGrd << " (" << minGrdI << ") / " << totGrd/POP_SIZE << " / " << maxGrd << endl;
#endif
//fittestIdx = minGrdI; //enable 1st line in for-loop of go()
//exit(0);
	//////////////// see ground-truth distortions of the initial population (just for debugging) ends ////////////////*/
}

int tournamentNo;
void GA::go(bool bruteForceSolution, bool removeOutliers, bool insertExistingMaps, bool doAdaptiveSampling)
{
	if (bruteForceSolution)
	{
		if (CHROM_LEN <= 10)
			mesh1->bruteForceMap(mesh2, CHROM_LEN); //samples on which brute force algo will run are computed in GA() constructor
		else
		{
			cout << "intractable to solve the shape correspondence problem with " << CHROM_LEN << " samples using the brute force approach\n";
			exit(0);
		}
		return;
	}

//	double start_time = get_time();

	//fitness value is in [0,1] interval
	float bestFitnessValue, margin = 0.00001f; //0.0000001f; //similar upto n digits after dot where n = # zeros after dot here
	int bestFitnessIdx;

#ifdef MULTIPLE_REINIT
	float bestSoFar = -INF; //new population and new evolution will be tried until the bestSoFar is seen twice
	int nReinits = 0;
	bool sameInitialPopulationOnReinits = false; //true makes the same initialPopulationCopy to be used in each while-loop iter below; false uses a different initial populaiton at each start (more logical)
	int** initialPopulationCopy = NULL;
	if (sameInitialPopulationOnReinits)
	{
		initialPopulationCopy = new int*[POP_SIZE];
		for (int i = 0; i < POP_SIZE; i++)
			initialPopulationCopy[i] = new int[CHROM_LEN];
	}
	while (true)
	{
	//start the genetic evolution algorithm
	if (sameInitialPopulationOnReinits)
	{
		if (nReinits == 0)
		{
			initPopulation(insertExistingMaps);
			for (int i = 0; i < POP_SIZE; i++)
				for (int j = 0; j < CHROM_LEN; j++)
					initialPopulationCopy[i][j] = population[i][j];
		}
		else //load the same initial population to the population[] for the new while-loop iteration
			for (int i = 0; i < POP_SIZE; i++)
				for (int j = 0; j < CHROM_LEN; j++)
					population[i][j] = initialPopulationCopy[i][j];
	}
	else //a dufferent initial populaiton at each start (more logical)
		initPopulation(insertExistingMaps);
#else
	//start the genetic evolution algorithm
	initPopulation(insertExistingMaps);
#endif

	bestFitnessValue = -INF;
	bestFitnessIdx = -1;
	
#ifdef AUTO_EARLY_BREAK
	vector< float > fitnesses;
	int L = 100, //200;//2000; //early-break tournaments when the last L resulting fitnesses are the same (so no change)
		nSwaps0Tours = 0; //nSwapsTotal=0 for a sequence of tournaments is a good indicator of convergence
#endif

#ifdef VERBOSE
#ifndef MULTIPLE_REINIT //save print space by skipping this
#ifdef SELECT_BEST_OF_N_XOVS
	cout << "evolution starts (popSize = " << POP_SIZE << ", best of " << N_XOVS << " xovs, mutRate = " << MUTATION_RATE << ", xovRate = " << XOVER_RATE << ")..\n";
#else
	cout << "evolution starts (popSize = " << POP_SIZE << ", mutRate = " << MUTATION_RATE << ", xovRate = " << XOVER_RATE << ")..\n";
#endif
#endif
#endif
//FILE* fPtrFit = fopen("ga-fit.dat", "w"), * fPtrIso = fopen("ga-iso.dat", "w"), * fPtrGrd = fopen("ga-grd.dat", "w"); //for plots in paper; 1 line per tournament
	//start a tournament
	for (tournamentNo = 0; tournamentNo < MAX_NUM_TOUR; tournamentNo++)
	{
		nMuts = nXovs = nSwapsTotal = nSubstringsTotal = 0; substrSizesTotal = 0.0f;
		if (tournamentNo > 0) //don't evolve in the first iteration to see the fittest of the initial population			
			//evolvePopulation(bestFitnessIdx); //1sec per tournamentNo iter for 1000samples
			//evolvePopulation2(bestFitnessIdx); //2secs per tournamentNo iter for 1000samples
			evolvePopulation3(); //3secs per tournamentNo iter for 1000samples

		float fitness = getFittestMember(); //index of fittest member in this population is written to global variable fittestIdx; global populationFitnesses[] is also set as fitnesses of the current population is computed
//float grd3, fitness3 = evalSolution3(fittestIdx, grd3);fprintf(fPtrFit, "%f\n", fitness3); fprintf(fPtrIso, "%f\n", 1.0f - fitness3); fprintf(fPtrGrd, "%f\n", grd3);

#ifndef MULTIPLE_REINIT //save print space by skipping this
		if (tournamentNo % 100 == 0)
			if (tournamentNo > 0) //(nMuts != 0 && nXovs != 0)
				//cout << "[" << tournamentNo << "] After " << nMuts << " mutations (" << (float)nSwapsTotal/nMuts << ") and " << nXovs << " crossovers (" << ((float)nSubstringsTotal/(nXovs*N_XOVS)) << ", " << ((float)substrSizesTotal/(nXovs*N_XOVS)) << "), fittest member (" << fittestIdx << ") of this generation has fitness = " << fitness << endl;
				cout << "[" << tournamentNo << "] After " << nMuts << " mutations (" << (float)nSwapsTotal/nMuts << ") and " << nXovs << " crossovers (" << ((float)nSubstringsTotal/(nXovs)) << ", " << ((float)substrSizesTotal/(nXovs)) << "), fittest member (" << fittestIdx << ") of this generation has fitness = " << fitness << endl;
				//if SELECT_BEST_OF_N_XOVS undefined, then delete *N_XOVS just-above; *N_XOVS is there 'cos i overcount nSubstringsTotal N_XOVS times for each xover operation (nXovs); similarly for substrSizesTotal
			else
				cout << "[" << tournamentNo << "] After " << nMuts << " mutations and " << nXovs << " crossovers, fittest member (" << fittestIdx << ") of the initial population has fitness = " << fitness << endl;
#endif

/*		if (SWAPPER != RANDOM_SWAP && nSwapsTotal == 0 && tournamentNo > 0)
		{
			cout << "switching to RANDOM_SWAP at tournament " << tournamentNo << endl; //to add more genetic variety (current GV_BASED_SWAP started to produce 0 swaps so replace it with no-question-asked RANDOM_SWAP)
			SWAPPER = RANDOM_SWAP;
		}//# of swaps increased as expected but fitness has not improved at all; so skip this tactic and use it as an early-break condition*/
#ifdef AUTO_EARLY_BREAK
		if (nSwapsTotal == 0) //even 1 little alternating swap (very rare) may cause this to be always false hence no early break; so i tried if((float)nSwapsTotal/nMuts < 0.001f) condition but still observed no-early-break; so keep fitnesses[] too
		{
			if (MUTATION_RATE > 0.8 && ++nSwaps0Tours >= 10) //when i want to see the xovers only (MUTATION_RATE=0), this early break should be disabled (done w/ the 1st condition, which disables all other lower rates too)
			{
#ifdef VERBOSE
				cout << "[" << tournamentNo << "] fitness seems to be stabilized at " << fitness << " (no swaps in the last " << nSwaps0Tours << " tournaments); early-breaking\n";
#endif
				bestFitnessValue = fitness;
				bestFitnessIdx = fittestIdx;
				break;
			}
		}
		else
			nSwaps0Tours = 0; //reset this 'cos some swapping occured (swap means i still have bad genes so that i'm swapping; so no convergence)
//#ifdef USE_GRD_DISTORTION //using this test for USE_GRD_DISTORTION undefined mode is also very ok; i added this ifdef condition to save micro time in undefined (usual) mode; observed that nSwapsTotal may rarely never be 0 so keep this block enabled
		/////////// fitness[]-based test ///////////
		fitnesses.push_back(fitness);
		int nRepeats = 0;
		for (int i = (int) fitnesses.size()-1; i >= 0; i--)
			if (fitness == fitnesses[i])
				nRepeats++;
			else
				break;
		if (nRepeats >= L)
		{
#ifdef VERBOSE
			cout << "[" << tournamentNo << "] fitness seems to be stabilized at " << fitness << " (no change in the last " << L << " tournaments); early-breaking\n";
#endif
			bestFitnessValue = fitness;
			bestFitnessIdx = fittestIdx;
			break;
		}
		/////////// fitness[]-based test ends ///////////
//#endif
#endif
		//check whether the fittest population member is the solution
		if (fitness >= (1.0f - margin)) //margin for precision
		{
			bestFitnessValue = fitness;
			bestFitnessIdx = fittestIdx;
			break;
		}
		else //if solution not found continue evolving
			if (fitness > bestFitnessValue)
			{
				bestFitnessValue = fitness;
				bestFitnessIdx = fittestIdx;
			}
//break;//see the fittest (min isometric distortion) of the POP_SIZE initial maps; see an arbitrary one by setting bestFitnessIdx=arbitrary
	}
//fclose(fPtrFit);fclose(fPtrIso);fclose(fPtrGrd);

#ifdef MULTIPLE_REINIT //save print space by skipping this
	nReinits++;
	//if (bestFitnessValue > bestSoFar)
	if (bestFitnessValue > bestSoFar + margin) //thresholded version for precision errors
	{
#ifdef VERBOSE
		cout << "bestSoFar updated from " << bestSoFar << " to " << bestFitnessValue << endl;
#endif
		bestSoFar = bestFitnessValue;		
	}
	//else if (bestFitnessValue == bestSoFar) //bestSoFar is seen twice so finish the search
	else if (bestFitnessValue >= bestSoFar-margin && bestFitnessValue <= bestSoFar+margin) //thresholded version for precision errors
	{
#ifdef VERBOSE
		cout << bestFitnessValue << " is seen twice so it is likely to be the global maximum; finishing the search at reinit # " << nReinits << "\n"; 
#endif
		break;
	}
	}
#endif
	cout << "solution found at [" << tournamentNo << "]; fitness value: " << bestFitnessValue << "\n";

	//transfer the content of solution/fittest-member into Mesh.verts to see it on screen
	for (int s = 0; s < CHROM_LEN; s++)
	{
		mesh1->verts[ mesh1->samples[s] ]->matchIdx = mesh2->samples[ population[bestFitnessIdx][s] ];
		mesh2->verts[ mesh2->samples[ population[bestFitnessIdx][s] ] ]->matchIdx = mesh1->samples[s]; //dual operation
	}

	if (doAdaptiveSampling && MODE != COMPUTE_ROBUST_LIST)
	{
#ifdef VERBOSE
//		cout << "\n========> " << get_time()-start_time << " secs for GA process that finds the correspondence\n";
#endif
		//fprint the GA-optimal result based on original sampling
		char fName[250]; sprintf(fName, "output\\%d-%d before-as.dat", mesh1->id, mesh2->id);
		mesh1->resultToFile(fName, mesh2, removeOutliers, false);

		//do adaptive sampling based via GA (this GA is very simialr to my GA process in shape correspondence above)
		//start_time = get_time();
		adaptiveSampling2(); //adaptive sampling is done in Mesh.cpp now so this one not in use anymore
#ifdef VERBOSE
		//cout << "\n========> " << get_time()-start_time << " secs for adaptive sampling that improves the correspondence\n";
#endif
		sprintf(fName, "output\\%d-%d after-as.dat", mesh1->id, mesh2->id);
		mesh1->resultToFile(fName, mesh2, removeOutliers, false);//, true);
	}
	else
	{
		char fName[250];
		//sprintf(fName, "output\\populationMaps\\%d-%d.dat", mesh1->id, mesh2->id);
		//mesh1->appendToFile(fName, mesh2); //so that population is inited w/ this cool mapping later; insertExistingMaps=true not improving the result so dont append
		if (MODE == COMPUTE_ROBUST_LIST)
			sprintf(fName, "input\\robustMaps\\%d-%d.dat", mesh1->id, mesh2->id);
		else
			sprintf(fName, "output\\%d-%d ga.dat", mesh1->id, mesh2->id);
		mesh1->resultToFile(fName, mesh2, removeOutliers, MODE == COMPUTE_ROBUST_LIST);
	}
}
void GA::go2()
{
	//same as go() except this one does GA, removes outliers, does GA again on the new sample set (CHROM_LEN reduces)
	//double start_time = get_time();
	int nReinits = 0;
	while (nReinits++ < 2) //2 initializations, first w/ all samples, second with non-outlier samples
	{
		initPopulation(false);

		//fitness value is in [0,1] interval
		float bestFitnessValue = -INF, margin = 0.00001f; //0.0000001f;
		int bestFitnessIdx = -1;
	
#ifdef AUTO_EARLY_BREAK
		vector< float > fitnesses;
		int L = 100, //200;//2000; //early-break tournaments when the last L resulting fitnesses are the same (so no change)
			nSwaps0Tours = 0; //nSwapsTotal=0 for a sequence of tournaments is a good indicator of convergence
#endif

#ifdef SELECT_BEST_OF_N_XOVS
		cout << "evolution starts (popSize = " << POP_SIZE << ", maxIters = " << MAX_NUM_TOUR << ", best of " << N_XOVS << " xovs, mutRate = " << MUTATION_RATE << ", xovRate = " << XOVER_RATE << ")..\n";
#else
		cout << "evolution starts (popSize = " << POP_SIZE << ", maxIters = " << MAX_NUM_TOUR << ", mutRate = " << MUTATION_RATE << ", xovRate = " << XOVER_RATE << ")..\n";
#endif
		//start a tournament
		for (tournamentNo = 0; tournamentNo < MAX_NUM_TOUR; tournamentNo++)
		{
			nMuts = nXovs = nSwapsTotal = nSubstringsTotal = 0; substrSizesTotal = 0.0f;
			if (tournamentNo > 0) //don't evolve in the first iteration to see the fittest of the initial population			
				//evolvePopulation(bestFitnessIdx); //1sec per tournamentNo iter for 1000samples
				//evolvePopulation2(bestFitnessIdx); //2secs per tournamentNo iter for 1000samples
				evolvePopulation3(); //3secs per tournamentNo iter for 1000samples

			float fitness = getFittestMember(); //index of fittest member in this population is written to global variable fittestIdx; global populationFitnesses[] is also set as fitnesses of the current population is computed
			//mesh1->isometricDistortion = 1.0f - fitness; //assuming fitness=1-iso in use in evalSolution(), i recover iso as 1-fitness; thanks to this adaptive isometricDistortion value, i can now early-break getIsometricDistortion() calls [not in use]
//if(tournamentNo==0){bestFitnessIdx=fittestIdx;break;}//see one specific init from initPopulation() (set manually by updating fittestIdx there)
			if (tournamentNo % 100 == 0)
				if (tournamentNo > 0) //(nMuts != 0 && nXovs != 0)
					//cout << "[" << tournamentNo << "] After " << nMuts << " mutations (" << (float)nSwapsTotal/nMuts << ") and " << nXovs << " crossovers (" << ((float)nSubstringsTotal/(nXovs*N_XOVS)) << ", " << ((float)substrSizesTotal/(nXovs*N_XOVS)) << "), fittest member (" << fittestIdx << ") of this generation has fitness = " << fitness << endl;
					cout << "[" << tournamentNo << "] After " << nMuts << " mutations (" << (float)nSwapsTotal/nMuts << ") and " << nXovs << " crossovers (" << ((float)nSubstringsTotal/(nXovs)) << ", " << ((float)substrSizesTotal/(nXovs)) << "), fittest member (" << fittestIdx << ") of this generation has fitness = " << fitness << endl;
					//if SELECT_BEST_OF_N_XOVS undefined, then delete *N_XOVS just-above; *N_XOVS is there 'cos i overcount nSubstringsTotal N_XOVS times for each xover operation (nXovs); similarly for substrSizesTotal
				else
					cout << "[" << tournamentNo << "] After " << nMuts << " mutations and " << nXovs << " crossovers, fittest member (" << fittestIdx << ") of the initial population has fitness = " << fitness << endl;

	/*		if (SWAPPER != RANDOM_SWAP && nSwapsTotal == 0 && tournamentNo > 0)
			{
				cout << "switching to RANDOM_SWAP at tournament " << tournamentNo << endl; //to add more genetic variety (current GV_BASED_SWAP started to produce 0 swaps so replace it with no-question-asked RANDOM_SWAP)
				SWAPPER = RANDOM_SWAP;
			}//# of swaps increased as expected but fitness has not improved at all; so skip this tactic and use it as an early-break condition*/
#ifdef AUTO_EARLY_BREAK
			if (nSwapsTotal == 0) //even 1 little alternating swap (very rare) may cause this to be always false hence no early break; so i tried if((float)nSwapsTotal/nMuts < 0.001f) condition but still observed no-early-break; so keep fitnesses[] too
			{
				if (++nSwaps0Tours >= 10)
				{
#ifdef VERBOSE
					cout << "[" << tournamentNo << "] fitness seems to be stabilized at " << fitness << " (no swaps in the last " << nSwaps0Tours << " tournaments); early-breaking\n";
#endif
					bestFitnessValue = fitness;
					bestFitnessIdx = fittestIdx;
					break;
				}
			}
			else
				nSwaps0Tours = 0; //reset this 'cos some swapping occured (swap means i still have bad genes so that i'm swapping; so no convergence)
//#ifdef USE_GRD_DISTORTION //using this test for USE_GRD_DISTORTION undefined mode is also very ok; i added this ifdef condition to save micro time in undefined (usual) mode; observed that nSwapsTotal may rarely never be 0 so keep this block enabled
			/////////// fitness[]-based test ///////////
			fitnesses.push_back(fitness);
			int nRepeats = 0;
			for (int i = (int) fitnesses.size()-1; i >= 0; i--)
				if (fitness == fitnesses[i])
					nRepeats++;
				else
					break;
			if (nRepeats >= L)
			{
#ifdef VERBOSE
				cout << "[" << tournamentNo << "] fitness seems to be stabilized at " << fitness << " (no change in the last " << L << " tournaments); early-breaking\n";
#endif
				bestFitnessValue = fitness;
				bestFitnessIdx = fittestIdx;
				break;
			}
			/////////// fitness[]-based test ends ///////////
//#endif
#endif
			//check whether the fittest population member is the solution
			if (fitness >= (1.0f - margin)) //margin for precision
			{
				bestFitnessValue = fitness;
				bestFitnessIdx = fittestIdx;
				break;
			}
			else //if solution not found continue evolving
				if (fitness > bestFitnessValue)
				{
					bestFitnessValue = fitness;
					bestFitnessIdx = fittestIdx;
				}
		}//*/
		cout << "solution found! fitness value: " << bestFitnessValue << "\n\n";

		//transfer the content of solution/fittest-member into Mesh.verts to see it on screen
		for (int s = 0; s < CHROM_LEN; s++)
		{
			mesh1->verts[ mesh1->samples[s] ]->matchIdx = mesh2->samples[ population[bestFitnessIdx][s] ];
			mesh2->verts[ mesh2->samples[ population[bestFitnessIdx][s] ] ]->matchIdx = mesh1->samples[s]; //dual operation
		}
		if (nReinits == 1) //remove outliers
		{
			//adapted from resultToFile()
			int N = (int) mesh1->samples.size();
			vector< int > corresp;
			for (int i = 0; i < N; i++)
			{
				int v2 = mesh1->verts[ mesh1->samples[ i ] ]->matchIdx;
				//if (v2 == -1) or if (isDuplicated()) tests are redundant here so keep the code short by removing them
				corresp.push_back(mesh1->samples[ i ]);
				corresp.push_back(v2);
			}
#ifdef OUTLIER_FREE_DISTORTION
			float isometricDistortion = (! mesh1->robustList.empty() ? mesh1->getIsometricDistortionRL_outlierFree(mesh2, corresp) : mesh1->getIsometricDistortion_outlierFree(mesh2, corresp)); //diso values updated now
#else
			float isometricDistortion = (! mesh1->robustList.empty() ? mesh1->getIsometricDistortionRL(mesh2, corresp) : mesh1->getIsometricDistortion(mesh2, corresp)); //diso values updated now
#endif
			//genetic algo may rarely leave an outlier match that is X+ times worse than the average isometryCost; so, just make their matchIdx=-1 to keep the map 1-to-1 (not onto anymore (hence not bijection; just an injection) 'cos some mesh2.verts will be unmatched)
			bool outlierDetected = false;
			float nOutliers = 0, X = 3.0f;
#ifdef OUTLIER_FREE_DISTORTION
			X = 5.0f; //the same X value i used in *_outlierFree()
#endif
			for (int i = 0; i < (int) mesh1->samples.size(); i++)
			{
int m = (mesh1->verts[ mesh1->samples[i] ]->matchIdx != -1 ? mesh1->verts[ mesh1->samples[i] ]->matchIdx : 0);//cout << verts[ samples[i] ]->diso << "\t" << verts[ samples[i] ]->dSpanning[m] << endl;//last one is dgrd (not stored in v.dgrd yet)
				if (mesh1->verts[ mesh1->samples[i] ]->diso > X*isometricDistortion)
				{
#ifdef VERBOSE
cout << "outlierrrrrrrrrrr: " << mesh1->samples[i] << " -> " << mesh1->verts[ mesh1->samples[i] ]->matchIdx << " w/ diso = " << mesh1->verts[ mesh1->samples[i] ]->diso << ", avg isometricDistortion = " << isometricDistortion << "\t" << mesh1->verts[ mesh1->samples[i] ]->dSpanning[m] << endl;
#endif
					mesh1->verts[ mesh1->samples[i] ]->sample = mesh2->verts[ mesh1->verts[ mesh1->samples[i] ]->matchIdx ]->sample = false; //still in samples[] but not sample anymore; hence Painter.getMatchingLines/SpheresSep will skip this
					mesh1->verts[ mesh1->samples[i] ]->matchIdx = mesh2->verts[ mesh1->verts[ mesh1->samples[i] ]->matchIdx ]->matchIdx = -1;
					nOutliers++;
					outlierDetected = true;
					if (mesh1->verts[ mesh1->samples[i] ]->robust)
					{
						cout << "a robustList match should never be an outlier; sth wrong\n";
						exit(0);
					}
				}
			}
			if (outlierDetected)
			{
#ifdef VERBOSE
				cout << nOutliers << " outliers removed\told isometricDistortion = " << isometricDistortion << endl;
#endif
				//adapted from refillCorresp()
				corresp.clear();
				for (int i = 0; i < (int) mesh1->samples.size(); i++)
				{
					int bv1 = mesh1->samples[i], bv2 = mesh1->verts[bv1]->matchIdx;
					if (bv2 == -1)
						continue;
					corresp.push_back(bv1); //next safe correspondence goes to following 2 slots in corresp
					corresp.push_back(bv2);
				}
			}
			isometricDistortion = (! mesh1->robustList.empty() ? mesh1->getIsometricDistortionRL(mesh2, corresp) : mesh1->getIsometricDistortion(mesh2, corresp));
			float groundTruthDistortion = mesh1->getGroundDistortion(mesh2, corresp, true),
				  groundDistoFull = mesh1->getGroundDistortionFull(mesh2, corresp),
				  isoDistoNoRL = mesh1->getIsometricDistortion(mesh2, corresp);
			if (outlierDetected)
			{
#ifdef VERBOSE
				cout << "\n" << isometricDistortion << " & " << isoDistoNoRL << " &&&& " << groundTruthDistortion << " & " << groundDistoFull << " for isometric &&&& ground-truth distortions of the resulting map (before outlier-free operation)\n\n";
#endif
				//refill chromosomes w/ N - nOutliers samples
				CHROM_LEN = N - (int) nOutliers;
				for (int i = 0; i < N; i++)
					if (! mesh1->verts[ mesh1->samples[i] ]->sample) //not a sample anymore, i.e. an outlier
					{
//cout << "erasing " << mesh1->samples[i] << "\t" << (int) mesh1->samples.size() << endl;
						mesh1->samples.erase(mesh1->samples.begin() + i);
						i--; //elements shifted to left 1 unit; so redo this i'th iteration
					}
				for (int i = 0; i < N; i++) //same stuff for mesh2
					if (! mesh2->verts[ mesh2->samples[i] ]->sample) //not a sample anymore, i.e. an outlier
					{
//cout << "erasing " << mesh2->samples[i] << "\t" << (int) mesh2->samples.size() << endl;
						mesh2->samples.erase(mesh2->samples.begin() + i);
						i--; //elements shifted to left 1 unit; so redo this i'th iteration
					}
				//update initMatchCandids[] since CHROM_LEN changed
				computeInitialMatchCandidates();
			}
			else
			{
#ifdef VERBOSE
				cout << "\n" << isometricDistortion << " & " << isoDistoNoRL << " &&&& " << groundTruthDistortion << " & " << groundDistoFull << " for isometric &&&& ground-truth distortions of the resulting map\n";
#endif
				break; //no need for the second iteration as the samples are not updated (no outlier removal)
			}
		}
	} //end of while-loop

	//the content of solution/fittest-member already transferred to Mesh.verts.matchIdx values; fprint them w/ some statistics
	char fName[250]; sprintf(fName, "output\\%d-%d.dat", mesh1->id, mesh2->id);
	mesh1->resultToFile(fName, mesh2, true, false);
}

float GA::getFittestMember()
{
	//sets the global fittestIdx to the population[][] idx of the best member/solution at this time; also return the fitness value of that best member

//double start_time = get_time();
	fittestIdx = 0;
	float fittestValue = evalSolution(fittestIdx);
	populationFitnesses[fittestIdx] = fittestValue; //bookkeep these evaluated fitness values to avoid re-evaluation for the same things in the evolvePopulation()
	for (int i = 1; i < POP_SIZE; i++) //0 is used just-above so start from 1
	{
		//compare fitness of current member with fitness of fittest member so far
		float fitness = evalSolution(i);
		populationFitnesses[i] = fitness;
		if (fitness > fittestValue)
		{
			fittestIdx = i;
			fittestValue = fitness;			
		}
	}
//cout << get_time()-start_time << " secs for 1 getFittestMember() call\n";//POPSIZE=1000,CHROMLEN=100, 0.006 secs (very fast)

	return fittestValue;
}

float GA::evalSolution(int c)
{
	//evaluate the solution given by the c'th chromose, i.e. row c of the population (population[c][*])

	vector< int > corres; //consecutive 2 entries make 1 match of the whole corresp, e.g., corresp[0]-corresp[1], corresp[2]-corresp[3], where corresp[] is an idx to verts[] not samples[]
	for (int i = 0; i < CHROM_LEN; i++)
	{
		corres.push_back(mesh1->samples[ i ]);
		corres.push_back(mesh2->samples[ population[c][i] ]);
	}
#ifdef USE_GRD_DISTORTION
	return 1.0f - mesh1->getGroundDistortion(mesh2, corres);
#else
#ifdef OUTLIER_FREE_DISTORTION
	return 1.0f - (robustListInUse ? mesh1->getIsometricDistortionRL_outlierFree(mesh2, corres) : mesh1->getIsometricDistortion_outlierFree(mesh2, corres));// - mesh1->getRankDistortion(mesh2, corres);
#else
	return 1.0f - (robustListInUse ? mesh1->getIsometricDistortionRL(mesh2, corres) : mesh1->getIsometricDistortion(mesh2, corres));// - mesh1->getRankDistortion(mesh2, corres);
	//return 1.0f / (robustListInUse ? mesh1->getIsometricDistortionRL(mesh2, corres) : mesh1->getIsometricDistortion(mesh2, corres)); dont use this 'cos go() assumes 1-iso in use
	//return pow(EXP, 6 * (1 - (robustListInUse ? mesh1->getIsometricDistortionRL(mesh2, corres) : mesh1->getIsometricDistortion(mesh2, corres)))); dont use this 'cos go() assumes 1-iso in use
#endif	
#endif
}
float GA::evalSolution3(int c, float& grd)
{
	//same as evalSolution() except this evaluates the solution via getIsometricDistortion() and getGroundDistortion() for fprints

	vector< int > corres;
	for (int i = 0; i < CHROM_LEN; i++)
	{
		corres.push_back(mesh1->samples[ i ]);
		corres.push_back(mesh2->samples[ population[c][i] ]);
	}
	grd = mesh1->getGroundDistortion(mesh2, corres);
	return 1.0f - mesh1->getIsometricDistortion(mesh2, corres);
}

void GA::evolvePopulation(int bestFitnessIdx)
{
	//cross-over and mutate the current population (also keep some existing chromosomes intact/unchanged to enable inheritance of good things from ancestors) to create a new/evolved/better population[][]
	//this evolution selects 2 population members (winner, loser) at ramdom; then, sometimes, crossovers the 2 selection to overwrite the loser with the xovered one; then, sometimes, mutates random members of the population; it's also known as Microbial GA

	int	a = -1, b = -1, winner, loser;
	for (int i = 0; i < POP_SIZE; i++)
	{
		//select parents by pulling 2 population members at random (or using a better heurisctic: rouletteSelection() that gives higher probability of selection to fitter members)
		do
		{			
//#ifdef DO_ROULETTE_SELECTION this not make sense here 'cos loser will be overwritten and with roulette loser may be a good chromosome; so pick winner-loser totally randomly
//			a = rouletteSelection(); b = rouletteSelection();
//#else //do trivial random selection
			a = randoi(POP_SIZE); b = randoi(POP_SIZE); //[0, POP_SIZE) = [0,POP_SIZE-1] interval
//#endif
		} while (a == b);

		//have a fight and see who has best genes
		//if (populationFitnesses[a] > populationFitnesses[b]) //use bookkeeped populationFitnesses[] for faster execution (not accurate 'cos population[] is updated within this loop this loop)
		if (evalSolution(a) > evalSolution(b)) //makes sense 'cos population[] is updated, meaning that initial populationFitnesses[] values are not safe to use
		{
			winner = a;
			loser = b;
		}
		else
		{
			winner = b;
			loser = a;
		}
		//possibly do some crossover
		if (randof() < XOVER_RATE) //i != bestFitnessIdx && test unnecessary 'cos loser can never be the bestFitnessIdx here (beaten by at least winner) and only loser will get updated
			//orderOneCrossover(winner, loser); //global population[loser][*] updated after this, i.e. row loser is updated only
			anotherCrossover(winner, loser); //global population[loser][*] updated after this, i.e. row loser is updated only
	}

	//possibly mutate the new population a bit to add some new genetic material
	for (int i = 0 ; i < POP_SIZE; i++)
		if (i != bestFitnessIdx && randof() < MUTATION_RATE) //first condition to keep the bestFitnessIdx of this generation intact/unchanged
			swapMutation(i); //global population[i][*] updated after this, i.e. row i is updated only
}

void GA::evolvePopulation2(int bestFitnessIdx)
{
	//cross-over and mutate the current population (also keep some existing chromosomes intact/unchanged to enable inheritance of good things from ancestors) to create a new/evolved/better population[][]
	//this evolution selects 2 population members (fitters are more likely to be selected); then, sometimes, crossovers the 2 selection to create a new member in the next idx of newPopulation[]; updates population[]; then, sometimes, mutates random members of the updated population[]

//	int newPopulation[POP_SIZE][CHROM_LEN], childChromo[CHROM_LEN], bestChildChromo[CHROM_LEN]; crushes when POP_SIZE=1250, CHROM_LEN=1000; weird but true; better define them dynamically as below
	/* instead of creating and deleting them here, do it once in the initPopulation() to save micro time
	int** newPopulation = new int*[POP_SIZE], * childChromo = new int[CHROM_LEN], * bestChildChromo = new int[CHROM_LEN];
	for (int i = 0; i < POP_SIZE; i++)
		newPopulation[i] = new int[CHROM_LEN];*/
	int a = 1, b = -1, winner, loser;
	for (int i = 0; i < POP_SIZE; i++)
	{
		//select parents by pulling 2 population members at random (or using a better heurisctic: rouletteSelection() that gives higher probability of selection to fitter members)
		do
		{			
#ifdef DO_ROULETTE_SELECTION
			a = rouletteSelection(); b = rouletteSelection();
#else //do trivial random selection
			a = randoi(POP_SIZE); b = randoi(POP_SIZE); //[0, POP_SIZE) = [0,POP_SIZE-1] interval
#endif
		} while (a == b);
		
		//have a fight and see who has best genes
		if (populationFitnesses[a] > populationFitnesses[b]) //use bookkeeped populationFitnesses[] for faster execution (still accurate 'cos population[] not updated 'till i leave this for-loop)
		//if (evalSolution(a) > evalSolution(b)) makes no sense 'cos population[] not updated 'till i leave this for-loop, meaning that initial populationFitnesses[] values are safe to use
		{
			winner = a;
			loser = b;
		}
		else
		{
			winner = b;
			loser = a;
		}

		//possibly do some crossover
		if (randof() < XOVER_RATE)
		{
			float minDisto = INF;
			for (int xov = 0; xov < N_XOVS; xov++)
			{
				for (int j = 0; j < CHROM_LEN; j++)
					childChromo[j] = population[loser][j]; //needs to be inited to loser to make anotherCrossover2() work accurately; and needs to be in this loop to prevent duplicates in the resulting call-by-ref childChromo[]
				anotherCrossover2(winner, loser, childChromo); //call by reference on childChromo[]

#ifdef SELECT_BEST_OF_N_XOVS
				vector< int > corres; //consecutive 2 entries make 1 match of the whole corresp, e.g., corresp[0]-corresp[1], corresp[2]-corresp[3], where corresp[] is an idx to verts[] not samples[]
				for (int i = 0; i < CHROM_LEN; i++)
				{
					corres.push_back(mesh1->samples[ i ]);
					corres.push_back(mesh2->samples[ childChromo[i] ]);
				}
#ifdef OUTLIER_FREE_DISTORTION
				float disto = (robustListInUse ? mesh1->getIsometricDistortionRL_outlierFree(mesh2, corres) : mesh1->getIsometricDistortion_outlierFree(mesh2, corres));// + mesh1->getRankDistortion(mesh2, corres);
#else
				float disto = (robustListInUse ? mesh1->getIsometricDistortionRL(mesh2, corres) : mesh1->getIsometricDistortion(mesh2, corres));// + mesh1->getRankDistortion(mesh2, corres);
#endif
				if (disto < minDisto)
				{
					minDisto = disto;
					for (int j = 0; j < CHROM_LEN; j++)
						bestChildChromo[j] = childChromo[j];
				}
#else //don't try the remaining xov possibilities and break early if SELECT_BEST_OF_N_XOVS is not defined; efficiency vs. accuracy tradeoff; N=10 trials is fast anyway so I recommend no break;
				for (int j = 0; j < CHROM_LEN; j++)
					bestChildChromo[j] = childChromo[j];
				break;
#endif
			}
			//transfer the content of bestChildChromo to childChromo
			for (int j = 0; j < CHROM_LEN; j++)
				childChromo[j] = bestChildChromo[j];
#ifdef SELECT_BEST_OF_N_XOVS
			nXovs = nXovs - N_XOVS + 1; //nXovs is overcounted in this mode so get rid of the excess amount and add just 1
#endif
		}
		else
			for (int j = 0; j < CHROM_LEN; j++)
				childChromo[j] = population[winner][j]; //needs to be set to winner to make newPopulation[i][j] assignment below accurate
		//child obtained by xovering winner and loser is saved in the next idx of newPopulation
		for (int j = 0; j < CHROM_LEN; j++)
			newPopulation[i][j] = (i != bestFitnessIdx ? childChromo[j] : population[i][j]); //preserve the best/fittest member of the current population[] for the next newPopulation[]
	}

	//copy newPopulation into population
	//copy(&newPopulation[0][0], &newPopulation[0][0] + POP_SIZE*CHROM_LEN, &population[0][0]);
	for (int p = 0; p < POP_SIZE; p++)
		for (int c = 0; c < CHROM_LEN; c++)
			population[p][c] = newPopulation[p][c];
//for (int j = 0; j < CHROM_LEN; j++) cout << population[7][j] << "p\t"; cout << "\n\n"; //exit(0);

	//possibly mutate the new population a bit to add some new genetic material
	for (int i = 0 ; i < POP_SIZE; i++)
		if (i != bestFitnessIdx && randof() < MUTATION_RATE) //first condition to keep the bestFitnessIdx of this generation intact/unchanged
			swapMutation(i); //global population[i][*] updated after this, i.e. row i is updated only

/*	//memo recapture (arrays allocated with new[] must be deallocated with delete[])
	for (int i = 0; i < POP_SIZE; i++) delete [] newPopulation[i]; delete [] newPopulation;
	delete [] childChromo;
	delete [] bestChildChromo;*/
}

void insertionSortGA(float* A, int s, int* vIdxs) //cant use the same name in Mesh.cpp; body is the exact copy of Mesh.insertionSort()
{
	//does insertion sort to sort array A of size s in descending order; call-by-ref to A, hence array is sorted in the caller when this returns

	for (int p = 1; p < s; p++)
	{
		float tmp = A[p];
		int j, tmpIdx = vIdxs[p];
		for (j = p; j > 0 && tmp > A[j-1]; j--) //just replace > w/ < to make ascending order
		{
			A[j] = A[j-1];
			vIdxs[j] = vIdxs[j-1];
		}
		A[j] = tmp;
		vIdxs[j] = tmpIdx;
	}
}
void GA::evolvePopulation3()
{
	//cross-over and mutate the current population (also keep some existing chromosomes intact/unchanged to enable inheritance of good things from ancestors) to create a new/evolved/better population[][]
	//this evolution selects 2 population members from the goodpart (fitters are more likely to be selected); then, sometimes, crossovers the 2 selection to create a new member in the next idx of newPopulation[]; updates badpart of the population[]; then, sometimes, mutates random members of the badpart of the updated population[]

//double start_time = get_time();

	//crushes when POP_SIZE=1250, CHROM_LEN=1000; weird but true; better define them dynamically as below
	//int newPopulation[BADPART_SIZE][CHROM_LEN], //newPopulation will overwrite the badpart of the population[][], which is of size BADPART_SIZE
		//childChromo[CHROM_LEN], bestChildChromo[CHROM_LEN];
	/* instead of creating and deleting them here, do it once in the initPopulation() to save micro time
	int** newPopulation = new int*[BADPART_SIZE], * childChromo = new int[CHROM_LEN], * bestChildChromo = new int[CHROM_LEN], a = -1, b = -1, winner, loser;
	for (int i = 0; i < BADPART_SIZE; i++)
		newPopulation[i] = new int[CHROM_LEN];
	//int sortedIdxs[POP_SIZE]; float fitnessVals[POP_SIZE]; crushes when POP_SIZE=1250, CHROM_LEN=1000; weird but true; better define them dynamically as below
	int* sortedIdxs = new int[POP_SIZE];
	float* fitnessVals = new float[POP_SIZE]; */
	int a = -1, b = -1, winner, loser;
	//sort the population[][] w.r.t. fitness of each member; sorted result is written back to population[][]
	for (int c = 0; c < POP_SIZE; c++)
	{
		//save super time by using the already-computed fitness values in populationFitnesses[] instead of recomputing by evalSolution(c); perfectly accurate 'cos population[] not updated 'till the last getFittestMember() call
		fitnessVals[c] = populationFitnesses[c]; //evalSolution(c);
		sortedIdxs[c] = c;
	}
//cout << "fitness of the 1st member = " << fitnessVals[0] << "\n\n";
	insertionSortGA(fitnessVals, POP_SIZE, sortedIdxs); //call by ref to fitnessVals (redundant) and sortedIdxs (output to be used below)
	//since insertionSortGA did a descending sorting on fitness values, first GOODPART_SIZE entries of sortedIdxs[] point to the goodpart (= fit part) of population[][] and the remaining entries (POP_SIZE-GOODPART_SIZE) are the badpart
//sanity check: for (int i = 0; i < POP_SIZE; i++) for (int j = i-1; j >= 0; j--) if (sortedIdxs[i] == sortedIdxs[j]) { cout << "duplicate error! " << i << " " << sortedIdxs[i] << endl; exit(0); }
	//transfer sorted members into tmpPopulation and then from tmpPopulation to population[][]
	//int tmpPopulation[POP_SIZE][CHROM_LEN]; crushes when POP_SIZE=1250, CHROM_LEN=1000; weird but true; better define them dynamically as below
	for (int c = 0; c < POP_SIZE; c++)
	{
		for (int i = 0; i < CHROM_LEN; i++)
			tmpPopulation[c][i] = population[ sortedIdxs[c] ][i];
		tmpFitnesses[c] = populationFitnesses[ sortedIdxs[c] ];
	}
	//from tmpPopulation[] to population[][]; a simple population = tmpPopulation; assignment sucks 'cos delete [] tmpPopulation; below will crush my population on later iterations
	for (int c = 0; c < POP_SIZE; c++)
	{
		for (int i = 0; i < CHROM_LEN; i++)
			population[c][i] = tmpPopulation[c][i];
		//transfer tmpFitnesses[] to populationFitnesses[] too
		populationFitnesses[c] = tmpFitnesses[c];
	}	
//cout << "fitness of the new 1st member = " << fitnessVals[0] << "\n\n";
	
//cout << "v0.......";
	for (int i = 0; i < BADPART_SIZE; i++)
	{
		//possibly do some crossover
		if (randof() < XOVER_RATE)
		{
			//select parents by pulling 2 population members at random from the goodpart of the population, which is the first GOODPART_SIZE members from index 0 to GOODPART_SIZE-1
			do
			{			
				a = randoi(GOODPART_SIZE); b = randoi(GOODPART_SIZE); //[0, GOODPART_SIZE) = [0,GOODPART_SIZE-1] interval
			} while (a == b);
		
			//have a fight and see who has best genes
			if (populationFitnesses[a] > populationFitnesses[b]) //use bookkeeped populationFitnesses[] for faster execution (still accurate 'cos population[] not updated 'till i leave this for-loop); note that population is updated before this loop via tmpPopulation but same update is applied on populationFitnesses[] too :)
			//if (evalSolution(a) > evalSolution(b)) makes no sense 'cos population[] not updated 'till i leave this for-loop, meaning that initial populationFitnesses[] values are safe to use
			{
				winner = a;
				loser = b;
			}
			else
			{
				winner = b;
				loser = a;
			}
		
			float minDisto = INF;
			for (int xov = 0; xov < N_XOVS; xov++) //this will break after the 1st iter when SELECT_BEST_OF_N_XOVS undefined; so N_XOVS value not matter at that time
			{
				for (int j = 0; j < CHROM_LEN; j++)
					childChromo[j] = population[loser][j]; //needs to be inited to loser to make anotherCrossover2() work accurately; and needs to be in this loop to prevent duplicates in the resulting call-by-ref childChromo[]
				//anotherCrossover2(winner, loser, childChromo); //call by reference on childChromo[]
				anotherCrossover3(winner, loser, childChromo); //call by reference on childChromo[]
#ifdef SELECT_BEST_OF_N_XOVS
				vector< int > corres; //consecutive 2 entries make 1 match of the whole corresp, e.g., corresp[0]-corresp[1], corresp[2]-corresp[3], where corresp[] is an idx to verts[] not samples[]
				for (int i = 0; i < CHROM_LEN; i++)
				{
					corres.push_back(mesh1->samples[ i ]);
					corres.push_back(mesh2->samples[ childChromo[i] ]);
				}
				float disto;
#ifdef USE_GRD_DISTORTION
				disto = mesh1->getGroundDistortion(mesh2, corres);
#else
#ifdef OUTLIER_FREE_DISTORTION
				disto = (robustListInUse ? mesh1->getIsometricDistortionRL_outlierFree(mesh2, corres) : mesh1->getIsometricDistortion_outlierFree(mesh2, corres));// + mesh1->getRankDistortion(mesh2, corres);
#else
				disto = (robustListInUse ? mesh1->getIsometricDistortionRL(mesh2, corres) : mesh1->getIsometricDistortion(mesh2, corres));// + mesh1->getRankDistortion(mesh2, corres);
#endif
#endif
				if (disto < minDisto)
				{
					minDisto = disto;
					for (int j = 0; j < CHROM_LEN; j++)
						bestChildChromo[j] = childChromo[j];
				}
#else //don't try the remaining xov possibilities and break early if SELECT_BEST_OF_N_XOVS is not defined; efficiency vs. accuracy tradeoff; N=10 trials is fast anyway so I recommend no break;
				for (int j = 0; j < CHROM_LEN; j++)
					bestChildChromo[j] = childChromo[j];
				break;
#endif
			}
			//transfer the content of bestChildChromo to childChromo
			for (int j = 0; j < CHROM_LEN; j++)
				childChromo[j] = bestChildChromo[j];
#ifdef SELECT_BEST_OF_N_XOVS
			nXovs = nXovs - N_XOVS + 1; //nXovs is overcounted in this mode so get rid of the excess amount and add just 1
#endif
		}
		else
			for (int j = 0; j < CHROM_LEN; j++)
				childChromo[j] = population[i+GOODPART_SIZE][j]; //no xover so keep the current bad member in the newPopulation[][] using this childChromo[]
		//child obtained by xovering winner and loser (or obtained by not xovering) is saved in the next idx of newPopulation
		for (int j = 0; j < CHROM_LEN; j++)
			newPopulation[i][j] = childChromo[j];
	}
//cout << "v0000000000\n\n";

	//copy newPopulation into badpart of the population
	for (int p = 0; p < BADPART_SIZE; p++)
		for (int c = 0; c < CHROM_LEN; c++)
			population[p+GOODPART_SIZE][c] = newPopulation[p][c]; //population[GOODPART_SIZE], population[GOODPART_SIZE+1], population[GOODPART_SIZE+2], .. are the bad members of the population (placed after the goodpart; hence GOODPART_SIZE+something) and will be replaced by newPopulation[] computed above
//for (int j = 0; j < CHROM_LEN; j++) cout << population[7][j] << "p\t"; cout << "\n\n"; //exit(0);

//cout << "mm..";
	//possibly mutate the new population a bit to add some new genetic material
/*	for (int i = 0; i < BADPART_SIZE; i++) //select only the BADPART for mutation
		if (randof() < MUTATION_RATE)
			swapMutation(i+GOODPART_SIZE); //global population[i+GOODPART_SIZE][*] updated after this, i.e. row i+GOODPART_SIZE is updated only (hence mutates random members of the badpart of the updated population[])*/
	for (int i = 1; i < POP_SIZE; i++) //not select only the BADPART for mutation 'cos BADPART is now kindda fixed after xovers above; so select any chromosome for mutation (from POP_SIZE)
									   //start i=1 'cos i=0 is the fittest member of the population due to sorting above; it is impossible to change it; so the fitness value will be non-decreasing in each tournament
		if (randof() < MUTATION_RATE)
			swapMutation(i); //global population[i][*] updated after this, i.e. row i is updated only (hence mutates random members of the updated population[])
//cout << "mm\n\n";

/*	//memo recapture (arrays allocated with new[] must be deallocated with delete[])
	for (int i = 0; i < BADPART_SIZE; i++) delete [] newPopulation[i]; delete [] newPopulation;
	for (int i = 0; i < POP_SIZE; i++) delete [] tmpPopulation[i]; delete [] tmpPopulation;
	delete [] childChromo;
	delete [] bestChildChromo;
	delete [] sortedIdxs;
	delete [] fitnessVals;
	delete [] tmpFitnesses;*/
//cout << get_time()-start_time << " secs for 1 evolovePopulation3() call\n"; //POPSIZE=1000,CHROMLEN=100, 0.25 secs (slow) decreases to 0.15 secs towards the end (due to cache locality i guess)
																			//becomes 0.03 secs (8 times faster than above) if i do RANDOM_SWAP or RANK_BASED_SWAP (which sucks accuracy); so mutations costly; become 0.02 if i disable muts (MUTATION_RATE=0)
																			//becomes 0.22 secs (still slow) if i disable xovers (XOVER_RATE=0) (which sucks accuracy); so xovers not costly

																			//upon this observation i just decrease nMaxTries from 2*CHROM_LEN to CHROM_LEN/2 and obtained 0.09 secs (fast) decreases to 0.06 secs towards the end; no accuracy loss :)
																			//as a matter of fact, i checked nSwapsTotal value staying the same even with nMaxTries=10 (constant); it makes sense 'cos nSwapsTotal decrease gradually as i converge and i cannot swap no matter how hard i try; so dont try that hard and keep nMaxTries low :)
																			//with nMaxTries=10, i have 0.05 secs (very fast) decreases to 0.04 :)))))))))))
}

int GA::rouletteSelection()
{
	//returns index of the selected chromosome/solution/member of the current population; selection probability is proportional to the fitness of the chromosome, i.e., the fitter the more likely to be selected

	float totalSum = 0.0f;
	for (int x = 0; x < POP_SIZE; x++)
//#ifdef USE_INIT_FITNESSES
		totalSum +=	populationFitnesses[x]; //use bookkeeped populationFitnesses[] for faster execution (still accurate 'cos population[] not updated 'till i leave the for-loop of the caller function)
//#else
//		totalSum +=	evalSolution(x);
//#endif	

	float partialSum = 0.0f, rand = randof(0.0f, totalSum);
	rand -= 0.01f; //for precision errors decrease rand slightly which enables partialSum beating it eventually
	for (int x = POP_SIZE-1; x >= 0; x--)
	{
//#ifdef USE_INIT_FITNESSES
		partialSum += populationFitnesses[x];
//#else
//		partialSum += evalSolution(x);
//#endif
		if (partialSum >= rand)
			return x; 
	}

#ifdef VERBOSE
	cout << "should not reach here (rouletteSelection() precision error no big deal)\t" << rand << " <= " << totalSum << endl;
#endif
/*	for (int x = POP_SIZE-1; x >= 0; x--)
//#ifdef USE_INIT_FITNESSES
		cout << populationFitnesses[x] << "\t";
//#else
//		cout << evalSolution(x) << "\t";
//#endif
	exit(0);
	return -1;*/
	return 0; //instead of returning nonsense -1, act as if `if (partialSum >= rand)' part is eventually true, in which case x=0
}

inline bool gvCompatible(vector< float > a, vector< float > b, float maxErr, int gvSize)
{
	//returns true if vectors a and b are compatible, i.e., each corresponding entry differs by at most maxErr

	for (int i = 0; i < gvSize; i++)
		if (fabs(a[i] - b[i]) > maxErr) //check the compatibility of the corresponding i'th rows in gv's
			return false;
	return true;
}
void GA::swapMutation(int c)
{
	//performs mutation on chromosome c (population[c][*]) by swapping the 2 bad genes as long as there are such pairs (originally 2 genes are selected randomly but not in this project)

	//as tournaments proceed there'll be less and less mutations 'cos xovers minimizes bad genes in chromosomes

	if (SWAPPER == GV_BASED_SWAP)
	{
		//worst1/2 will point the top2 worst genes to be swapped; first pick the worst gene w.r.t. gv
		int worst1, worst2, nMaxTries = 10/*CHROM_LEN/2; CHROM_LEN*2;*/, end = CHROM_LEN - 1; //-1 'cos randoi lower bound is worst1+1
		//initial match candidates for this jth sample must have compatible corresponding rows in gv's; by compatible i allow distance difference upto 0.25 (1 is the max distance, e.g. hand to toe, so 0.25 allows errors upto toe to knee)
		float maxErr = 0.125f; //toe-to-knee normalized distance is 0.25 but since i'm taking the subtraction below i should use the half value
							   //when this is too low, e.g., 0.0625, i try all nMaxTries (and still not find a satisfactory worst2, hence nSwaps=0 always) which significantly slows down the execution
		int gvSize =  (int) mesh1->verts[ mesh1->samples[0] ]->gv.size(); //gvSize'll be same (and equal to robustList.size/2) for all samples
		if (gvSize == 0 && tournamentNo == 0)
			cout << "\nWARNING: swapMutation will be nonsense since GVs are empty\n\n";
#ifdef USE_GRD_DISTORTION
		vector< int > corres; //consecutive 2 entries make 1 match of the whole corresp, e.g., corresp[0]-corresp[1], corresp[2]-corresp[3], where corresp[] is an idx to verts[] not samples[]
		for (int i = 0; i < CHROM_LEN; i++)
		{
			corres.push_back(mesh1->samples[ i ]);
			corres.push_back(mesh2->samples[ population[c][i] ]);
		}
		float distortionLimit, disto, distortion, X = 1.1f;//2.0f is terrible as i do only ~1 swap after tournament 50; X=1.1f makes it better but using ranking is more efficient 'cos agdOrder values are already known; also more accurate as it is independent of X
		distortionLimit = X * mesh1->getGroundDistortion(mesh2, corres); //dgrd values are ready		
#endif
		//use diso values (not in use) or agdOrder rankings (in use) to make swaps
		for (int i = 0; i < end; i++)
		{
			int nTries = 0;
#ifdef USE_GRD_DISTORTION
			distortion = mesh2->verts[ mesh2->samples[ population[c][i] ] ]->dgrd;
			if (distortion > distortionLimit) //gv-based evaluation thinks mesh1.samples[i] and mesh2.samples[population[c][i]] is a bad pair
#else			
			if (! gvCompatible(mesh1->verts[ mesh1->samples[i] ]->gv, mesh2->verts[ mesh2->samples[ population[c][i] ] ]->gv, maxErr, gvSize)) //gv-based evaluation thinks mesh1.samples[i] and mesh2.samples[population[c][i]] is a bad pair
#endif
			{
				worst1 = i;
				bool tooManyTries = false;
				do
				{
					worst2 = randoi(worst1+1, CHROM_LEN); //+1 so no duplicate danger; also worst2 is from the right of worst1 (left of worst1 already fixed in prev iterations) so no duplicate danger
#ifdef USE_GRD_DISTORTION
					//disto = mesh2->verts[ mesh2->samples[ population[c][worst2] ] ]->dgrd; this disto by worst2 has no guarantees to be compatible with worst1
					disto = mesh2->verts[ mesh2->samples[ population[c][worst2] ] ]->dSpanning[ mesh1->samples[worst1] ]; //this disto by worst2 is guaranteed to be compatible with worst1 (after the while condition below)
#endif
					if (nTries++ == nMaxTries)
					{
						tooManyTries = true;
						break;
					}
				}
#ifdef USE_GRD_DISTORTION
				while (disto > distortionLimit); //to swap bad genes, worst2 must have had incompatible gv with worst1
#else
				while (! gvCompatible(mesh1->verts[ mesh1->samples[worst1] ]->gv, mesh2->verts[ mesh2->samples[ population[c][worst2] ] ]->gv, maxErr, gvSize)); //to swap bad genes, worst2 must have had incompatible gv with worst1
#endif
				if (! tooManyTries) //break event not used above so worst2 is valid
				{
//cout << tournamentNo << "\t" << c << "\tswapping " << worst1 << " - " << worst2 << "\t" << windowSize << endl;//distortion << " " << distortionLimit << endl;swapped=true;
					//swap array elements at those indices
					int tmp = population[c][worst1];
					population[c][worst1] = population[c][worst2];
					population[c][worst2] = tmp;
					nSwapsTotal++;
				}
			} //end of if
		} //end of i
	}
	else if (SWAPPER == RANK_BASED_SWAP)
	{
		//worst1/2 will point the top2 worst genes to be swapped; first pick the worst gene w.r.t. individual isometric distortion (diso) or ranking (in use 'cos computing diso in each swap is slightly costly; needs getIsometricDistortion(population[c][*]))
		int worst1, worst2, nMaxTries = 10/*CHROM_LEN/2; CHROM_LEN*2;*/, end = CHROM_LEN - 1; //-1 'cos randoi lower bound is worst1+1
#ifdef USE_GRD_DISTORTION
		vector< int > corres; //consecutive 2 entries make 1 match of the whole corresp, e.g., corresp[0]-corresp[1], corresp[2]-corresp[3], where corresp[] is an idx to verts[] not samples[]
		for (int i = 0; i < CHROM_LEN; i++)
		{
			corres.push_back(mesh1->samples[ i ]);
			corres.push_back(mesh2->samples[ population[c][i] ]);
		}
		float distortionLimit, disto, distortion, X = 1.1f;//2.0f is terrible as i do only ~1 swap after tournament 50; X=1.1f makes it better but using ranking is more efficient 'cos agdOrder values are already known; also more accurate as it is independent of X
		distortionLimit = X * mesh1->getGroundDistortion(mesh2, corres); //dgrd values are ready
#endif
		//use diso values (not in use) or agdOrder rankings (in use) to make swaps
		for (int i = 0; i < end; i++)
		{
			//cout << i << "-" << c << "--" << abs(mesh1->verts[ mesh1->samples[i] ]->agdOrder - mesh2->verts[ mesh2->samples[ population[c][i] ] ]->agdOrder) << "\t";
			int nTries = 0;
#ifdef USE_GRD_DISTORTION
			distortion = mesh2->verts[ mesh2->samples[ population[c][i] ] ]->dgrd;
			if (distortion > distortionLimit) //gv-based evaluation thinks mesh1.samples[i] and mesh2.samples[population[c][i]] is a bad pair
#else
			if (abs(mesh1->verts[ mesh1->samples[i] ]->agdOrder - mesh2->verts[ mesh2->samples[ population[c][i] ] ]->agdOrder) > windowSize) //rank-based evaluation thinks mesh1.samples[i] and mesh2.samples[population[c][i]] is a bad pair [//note that this is the same criteria as in goodSubstring() and while-condition below]
#endif
			{
				worst1 = i;
				bool tooManyTries = false;
				do
				{
					worst2 = randoi(worst1+1, CHROM_LEN); //+1 so no duplicate danger; also worst2 is from the right of worst1 (left of worst1 already fixed in prev iterations) so no duplicate danger
#ifdef USE_GRD_DISTORTION
					//disto = mesh2->verts[ mesh2->samples[ population[c][worst2] ] ]->dgrd; this disto by worst2 has no guarantees to be compatible with worst1
					disto = mesh2->verts[ mesh2->samples[ population[c][worst2] ] ]->dSpanning[ mesh1->samples[worst1] ]; //this disto by worst2 is guaranteed to be compatible with worst1 (after the while condition below)
#endif
					if (nTries++ == nMaxTries)
					{
						tooManyTries = true;
						break;
					}
				}
#ifdef USE_GRD_DISTORTION
				while (disto > distortionLimit); //to swap bad genes, worst2 must have had incompatible gv with worst1
#else
				while (abs(mesh1->verts[ mesh1->samples[worst1] ]->agdOrder - mesh2->verts[ mesh2->samples[ population[c][worst2] ] ]->agdOrder) > windowSize); //to swap bad genes, worst2 must have had an agdOrder that is very different from worst1.agdOrder; e.g. 34 vs. windowSize=5 makes here true (34>5) and hence re-loop
#endif
				if (! tooManyTries) //break event not used above so worst2 is valid
				{
//cout << tournamentNo << "\t" << c << "\tswapping " << worst1 << " - " << worst2 << "\t" << windowSize << endl;//distortion << " " << distortionLimit << endl;swapped=true;
					//swap array elements at those indices
					int tmp = population[c][worst1];
					population[c][worst1] = population[c][worst2];
					population[c][worst2] = tmp;
					nSwapsTotal++;
				}
			} //end of if
		} //end of i
	}
	else //RANDOM_SWAP
	{
//cout << "warning: the next tournament will get stuck in infinite loop 'cos agdOrder-ranking-based goodSubstring() will never find a good substring\n"; notInUse
		//random swap of genes
//		for (int i = 0; i < CHROM_LEN; i++)
//			mesh2->verts[ mesh2->samples[ population[c][i] ] ]->processed = false; //true means already used in swap operation (so don't use it again to prevent potential duplicates; no!!!!!! no duplicate risk w/o processed variable)
		int rand1, rand2;
		float localMutRate = 0.6f;//0.3f; //localMutRate percentage of genes of this chromose will be swapped
		for (int i = 0; i < CHROM_LEN; i++)
			if (randof() < localMutRate)// && ! mesh2->verts[ mesh2->samples[ population[c][i] ] ]->processed)
			{
				rand1 = i; //rand1.processed = false for sure
				do
				{
					rand2 = randoi(CHROM_LEN);
				} while (//mesh2->verts[ mesh2->samples[ population[c][rand2] ] ]->processed || //to prevent a potential duplicate (nonbijection), rand2 must not have been swapped/processed before; no!!!!!! no duplicate risk w/o processed variable
						 rand1 == rand2); //rand1 must be different from rand2 for a meaningful swap
				//swap array elements at those indices
				int tmp = population[c][rand1];
				population[c][rand1] = population[c][rand2];
				population[c][rand2] = tmp;
				//mesh2->verts[ mesh2->samples[ population[c][rand1] ] ]->processed = mesh2->verts[ mesh2->samples[ population[c][rand2] ] ]->processed = true; //rand1 and 2 are used in swap operation; not use them again
				nSwapsTotal++;
			}

/*//sanity check: is there a duplicate in population[c][*]?
for (int i = 0; i < CHROM_LEN; i++)
	for (int j = 0; j < CHROM_LEN; j++)
		if (i != j && population[c][i] == population[c][j])
		{
			cout << "duplicate detected at idx " << i << " " << population[c][i] << endl;
			exit(0);
		}//*/
	}
	nMuts++;
}

void GA::orderOneCrossover(int wi, int lo)
{
	//performs order-one crossover on chromosome c (population[c][*]) by first copying a random substring from winner wi to loser lo; then copy the numbers (that are not in new lo) from loCopy to new lo;
	//hence population[lo][*] is updated only and guaranteed to be duplicate free (a mandatory requirement to produce bijection maps in my application)

//for (int i = 0; i < CHROM_LEN; i++) cout << population[wi][i] << "w\t"; cout << "\n\n"; for (int i = 0; i < CHROM_LEN; i++) cout << population[lo][i] << "g\t"; cout << "\n\n";

	int* loCopy = new int[CHROM_LEN], start, end, subStrSize, nCopied = 0, dupCheck, shift = 0;
	for (int i = 0; i < CHROM_LEN; i++)
		loCopy[i] = population[lo][i];

	//substring start and end indices
	do
	{
		start = randoi(CHROM_LEN); end = randoi(CHROM_LEN); //[0, CHROM_LEN) = [0,CHROM_LEN-1] interval
	} while (start >= end); //start < end for sure after this point

	//copy substring from wi to lo
	for (int i = start; i <= end; i++)
		population[lo][i] = population[wi][i];
	subStrSize = end - start + 1;

	//copy the numbers (that are not in new lo) from loCopy to new lo
	while (nCopied < CHROM_LEN - subStrSize) //subStrSize already copied above; now copy the rest CHROM_LEN - subStrSize amount of numbers
	{
		dupCheck = loCopy[(end + 1 + shift++) % CHROM_LEN];
		//check whether this dupCheck appears in the substring already copied above
		bool exists = false;
		for (int i = start; i <= end; i++)
			if (population[lo][i] == dupCheck)
				exists = true;
		if (! exists) //no duplication; do the copying
			population[lo][(end + 1 + nCopied++) % CHROM_LEN] = dupCheck; //note that nCopied++
	}
	nXovs++;
	delete [] loCopy;
//cout << start << "-" << end << "\n\n"; for (int i = 0; i < CHROM_LEN; i++) cout << population[wi][i] << "w\t"; cout << "\n\n"; for (int i = 0; i < CHROM_LEN; i++) cout << population[lo][i] << "g\t"; cout << "\n\n"; exit(0);
}
void GA::anotherCrossover(int wi, int lo)
{
	//similar to orderOneCrossover() performs another crossover, which preserves loser better, i.e. after substring transfer, this method tries to preserve more of the loser than orderOneCrossover() does
	//this crossover is also known as Partially-Mapped Crossover (PMX)

	//for (int i = 0; i < CHROM_LEN; i++) cout << population[wi][i] << "w\t"; cout << "\n\n"; for (int i = 0; i < CHROM_LEN; i++) cout << population[lo][i] << "g\t"; cout << "\n\n";

	int start, end, subStrSize;

	//substring start and end indices
	do
	{
		start = randoi(CHROM_LEN); end = randoi(CHROM_LEN); //[0, CHROM_LEN) = [0,CHROM_LEN-1] interval
	} while (start >= end); //start < end for sure after this point

	subStrSize = end - start + 1;
	int* deletedSubstr = new int[subStrSize], * addedSubstr = new int[subStrSize];

	//copy substring from wi to lo
	for (int i = start; i <= end; i++)
	{
		deletedSubstr[i-start] = population[lo][i];
		addedSubstr[i-start] = population[wi][i];
		population[lo][i] = population[wi][i];		
	}

	for (int i = 0; i < subStrSize; i++)
		for (int j = 0; j < subStrSize; j++)
			if (addedSubstr[i] == deletedSubstr[j])
			{
				addedSubstr[i] = -1;
				deletedSubstr[j] = -1;
			}

	for (int i = 0; i < CHROM_LEN; i++)
		if (i < start || i > end)
		{
			for (int j = 0; j < subStrSize; j++)
				if (population[lo][i] == addedSubstr[j])
				{
					for (int k = 0; k < subStrSize; k++)
						if (deletedSubstr[k] != -1)
						{
							population[lo][i] = deletedSubstr[k];
							deletedSubstr[k] = -1;
							break;
						}
				}
		}

	delete [] deletedSubstr;
	delete [] addedSubstr;

	nXovs++;
	nSubstringsTotal++;
//cout << start << "-" << end << "\n\n"; for (int i = 0; i < CHROM_LEN; i++) cout << population[wi][i] << "w\t"; cout << "\n\n"; for (int i = 0; i < CHROM_LEN; i++) cout << population[lo][i] << "g\t"; cout << "\n\n"; exit(0);
}
void GA::anotherCrossover2(int wi, int lo, int* child)
{
	//this crossover is also known as Partially-Mapped Crossover (PMX); same as anotherCrossover() except this one never overwrites population[lo][*]; instead it writes to child[]

	int start, end, subStrSize, f = 1; //f >= -1 (-1 allows start=end which means subStrSize>=1; 0 allows subStrSize>=2, 1 allows subStrSize>=3, and so on)
	f = 3;//CHROM_LEN / 10; is auto-adaptive but a fixed 3 is meaningful for all sample sizes; it means substrings will be of size 5 or more

	//substring start and end indices
	do
	{
		start = randoi(CHROM_LEN); end = randoi(CHROM_LEN); //[0, CHROM_LEN) = [0,CHROM_LEN-1] interval
	//} while (start >= end); //start < end for sure after this point
	} while (start >= end-f); //start < end-f for sure after this point which disallows too short (size <= f+1) substrings

	subStrSize = end - start + 1;
	int* deletedSubstr = new int[subStrSize], * addedSubstr = new int[subStrSize];

	//copy substring from wi to lo
	for (int i = start; i <= end; i++)
	{
		deletedSubstr[i-start] = population[lo][i];
		addedSubstr[i-start] = population[wi][i];
		child[i] = population[wi][i];
	}

	for (int i = 0; i < subStrSize; i++)
		for (int j = 0; j < subStrSize; j++)
			if (addedSubstr[i] == deletedSubstr[j])
			{
				addedSubstr[i] = -1;
				deletedSubstr[j] = -1;
			}

	for (int i = 0; i < CHROM_LEN; i++)
		if (i < start || i > end)
		{
			for (int j = 0; j < subStrSize; j++)
				if (population[lo][i] == addedSubstr[j])
				{
					for (int k = 0; k < subStrSize; k++)
						if (deletedSubstr[k] != -1)
						{
							child[i] = deletedSubstr[k];
							deletedSubstr[k] = -1;
							break;
						}
				}
		}

	delete [] deletedSubstr;
	delete [] addedSubstr;
	
	nXovs++;
	nSubstringsTotal++;

/*	//sanity check: is there a duplicate in child[]?
	for (int i = 0; i < CHROM_LEN; i++)
		for (int j = 0; j < CHROM_LEN; j++)
			if (i != j && child[i] == child[j])
			{
				cout << "duplicate detected at idx " << i << " " << child[i] << endl;
				exit(0);
			}//*/
}

bool overlap(int a, int b, int c, int d)
{
	//returns true if intervals a-b and c-d overlap/intersect
#define BTW(L,X,U) ((L) <= (X) && (X) <= (U))
	return (BTW(c, b, d) || BTW(c, a, d)) || (a < c && d < b);
}

bool GA::goodSubstring(int s, int e, int wi)
{
	//returns true if substring from idx s to idx e is a good one in winner; a substr is good if it has compatible agdOrders/ranks with the corresponding mesh1 agdOrders

return true;//not sure if this function is crucial; my tests with 100-samples show that it is not crucial

	if (INIT_POP_METHOD == RND_INIT)// && //no precaution on agdOrder compatibility in RND_INIT; so this would have always returned false and kept anotherCrossover3() in infinite loop
		//tournamentNo < 500) //after enough tournaments agdOrders are compatible enough so i can use this function (skip the return true just-below after tournamentNo >= 500)
		return true;

	for (int i = s; i <= e; i++)
		if (abs(mesh1->verts[ mesh1->samples[i] ]->agdOrder - mesh2->verts[ mesh2->samples[ population[wi][i] ] ]->agdOrder) > windowSize) //rank-based evaluation thinks mesh1.samples[i] and mesh2.samples[population[c][i]] is a bad pair
																																		   //note that this is the same criteria as in swapMutation()
			return false;
	return true;
}
void GA::anotherCrossover3(int wi, int lo, int* child)
{
	//same as anotherCrossover2() except this one xovers multiple substrings instead of just 1; hence called multi-xover

	int b = (CHROM_LEN / 20 != 1 ? CHROM_LEN / 20 : 2), nSubstrings = randoi(1, b), //nSubstrings=1 gives me anotherCrossover2(); mod by 0 crashes so prevent b=1 case; (if b=0, randoi() returns 1 'cos 35 % 12 = 11 & 35 % -12 = 11 & 35 % -2 = 1 (-ve mods act like +ve mods))
		* starts = new int[nSubstrings], * ends = new int[nSubstrings], f = 1, //f >= -1 (-1 allows start=end which means subStrSize>=1; 0 allows subStrSize>=2, 1 allows subStrSize>=3, and so on)
																			   //f=CHROM_LEN / 10; is auto-adaptive but a fixed 3 is meaningful for all sample sizes; it means substrings will be of size 5 or more
		maxSubstrSize = CHROM_LEN / 5, //max length of a substring to be xovered; make it CHROM_LEN to cancel this effect
		nMaxTries = 1000; //try to find nonoverlapping substrings (and good substrings) at most this many times
//if (f == 1 && maxSubstrSize < 3) maxSubstrSize = 3; //f=1 allows subStrSize>=3 so maxSubstrSize < 3 is a contradiction and causes infinite loop below
	if (maxSubstrSize < f + 2) //generalized version of the argument just-above
		maxSubstrSize = f + 2;

	vector< int > deletedSubstr, addedSubstr, markedIdxs; //6 1 2 -1 3 10 9 16 -1 13 14 -1 will be created for deletedSubstr which means substrings 6 1 2, 3 10 9 16, 13 14 will be deleted from loser (-1 is the virtual separator that wont be in the vector<>)
	//decide start-end of each substring
	for (int i = 0; i < nSubstrings; i++)
	{
		int substrSize, nTries = 0, nOverlapTries = 0;
		do
		{
			starts[i] = randoi(CHROM_LEN); ends[i] = randoi(CHROM_LEN); //[0, CHROM_LEN) = [0,CHROM_LEN-1] interval
			substrSize = ends[i] - starts[i] + 1; //negative substrSize is naturally handled by the 1st while condition below
			if (nTries++ == nMaxTries)
			{
cout << "infinite-loop potential due to agdOrder-ranking-based goodSubstring() " << tournamentNo << " " << nSubstrings << endl;//swapMutation() optimizes that ranking (not explicitly) so it never inf-loops but it will inf-loop if i do RND_INIT which creates bad agdOrders in the beginning (before swapMutation() ~correct them))
				exit(0); //can't do break; 'cos current starts[i] pair ends[i] may not be valid, e.g., starts[i] > ends[i] nonsense
			}
		} //while (starts[i] >= ends[i]-f || substrSize > maxSubstrSize || ! goodSubstring(starts[i], ends[i], wi)); //start < end-f for sure after this point which disallows too short (size <= f+1) substrings
		    while (starts[i] >= ends[i]-f || substrSize > maxSubstrSize); //since goodSubstring() returns true immediately save time and not call it

		//prevent overlapping start-end intervals to keep things simple when doing duplicate handling on child[] below
		bool overlapping = false;
		for (int j = 0; j < i; j++)
			//starts[i]-ends[i] (a-b) vs. starts[j]-ends[j] (c-d)
			if (overlap(starts[i], ends[i], starts[j], ends[j]))
			{
				overlapping = true;
				break;
			}
		if (overlapping)
		{
			if (nOverlapTries++ == nMaxTries)
			{
#ifdef VERBOSE
cout << "non-overlapping substring in xover cannot be found within " << nOverlapTries-1 << " trials; continuing with " << i << " substrings\n";
#endif
				nSubstrings = i;
				break;
			}
			i--; //re-do the same i value in the upcoming iteration
			continue; //re-try starts/ends[i]
		}

		//append current start-end substr into deletedSubstr and addedSubstr
		for (int j = starts[i]; j <= ends[i]; j++)
		{
			deletedSubstr.push_back(population[lo][j]);
			addedSubstr.push_back(population[wi][j]);	
		}
	}
//for (int i = 0; i < nSubstrings; i++) cout << starts[i] << "--" << ends[i] << endl;//exit(0);

	int next = 0, size = (int) addedSubstr.size(); //same as deletedSubstr.size()

	//mark the repeated idxs in child[]
	for (int i = 0; i < size; i++) //addedSubstr
	{
		bool rep = true; //true means this char will be repeated in child[]
		for (int j = 0; j < size; j++) //deletedSubstr
			if (addedSubstr[i] == deletedSubstr[j]) //exists in both addedSubstr and deletedSubstr; hence no danger of repeating
			{
				rep = false;
				break;
			}
		if (rep)			
			for (int j = 0; j < CHROM_LEN; j++)
				if (child[j] == addedSubstr[i])
				{
					markedIdxs.push_back(j); //mark the duplicate-entry in child[] so that i can handle it later
					break;
				}
	}

	//overwrite the marked duplicates in child[] using the ones that exist only in the deletedSubstr, not in both addedSubstr and deletedSubstr
	for (int i = 0; i < size; i++) //deletedSubstr
	{
		int j;
		for (j = 0; j < size; j++) //addedSubstr
			if (deletedSubstr[i] == addedSubstr[j]) //exists in both addedSubstr and deletedSubstr
				break;
		if (j == size) //break; not used so exists only in the deletedSubstr
			child[ markedIdxs[next++] ] = deletedSubstr[i];
	}

	//copy substring from wi to child[] (recall child[] initially holds the loser lo)
	float substrSizesLocalTotal = 0.0f; //total for this call to anotherCrossover3()
	for (int i = 0; i < nSubstrings; i++)
	{
		for (int j = starts[i]; j <= ends[i]; j++)			
			child[j] = population[wi][j];
		substrSizesLocalTotal += (ends[i] - starts[i] + 1); //for statistical purposes
	}
	substrSizesTotal += (substrSizesLocalTotal/nSubstrings); //average substring size on this function call is added to the global substrSizesTotal

	delete [] starts;
	delete [] ends;
	
	nXovs++;
	nSubstringsTotal += nSubstrings;

/*for (int c = 0; c < CHROM_LEN; c++)
	cout << child[c] << "\t";
//exit(0);*/

/*	//sanity check: is there a duplicate in child[]?
	for (int i = 0; i < CHROM_LEN; i++)
		for (int j = 0; j < CHROM_LEN; j++)
			if (i != j && child[i] == child[j])
			{
				cout << "duplicate detected at idx " << i << " " << child[i] << endl;
				exit(0);
			}//*/
}

//#define USE_CLOSEST_ROBUST_MATCH //undefine this to use geodesics to ALL robust matches, i.e. the gv vector; define to use only the closest robust match in evalSolution2()
//#define MIN_RADIUS_IN_USE //define this to use min of all distances used in radius computation; undefine to use the average instead; this distance (min or avg) will be later used in fitness function
//#define BFS_BASED_PATCHES //define this to define/mark patches via O(V) BFS, which is faster but less accurate than O(VlogV) dijkstra-based computation; both BFS and dijkstra are shortcutted (but shortcutting not saving a lot)
void GA::adaptiveSampling2()
{
	//chromosomes represent the order of new sampling; do GA on them until the fitness is good (fitness = |new radiusN - old radiusN| + alpha * isometricDistortion where alpha=0 'cos radiusN & isometricDistortion are hard to combine)
	
	if (MODE == GA_WITHOUT_ROBUST_LIST)
	{
		cout << "no robust list so adaptiveSampling2 cannot run\n"; //even the USE_CLOSEST_ROBUST_MATCH mode requires robustList so cannot run
		return;
	}

#ifdef VERBOSE
	//below heavily adapted from Mesh.adaptiveSampling()
	cout << "adaptive sampling on mesh2 to improve correspondences ("
#ifdef BFS_BASED_PATCHES
		 << "patching by approx geodesics (BFS))..\n";
#else
		 << "patching by exact geodesics (Dijkstra))..\n";
#endif
#endif

	//average sampling radius of the FPS on mesh2 can be determined by averaging the closest-sample distances to each sample
	mesh2->radius = 0.0f;
	float radiusN = 0.0f;
	mesh2->minRadius = INF;
	int n = (int) mesh2->samples.size();
	if (n != CHROM_LEN)
	{
		cout << "cannot happen\n";
		exit(0);
	}	
	for (int i = 0; i < n; i++)
	{
		float minDist = INF, minDistN = INF;
		int closest = -1;
		for (int j = 0; j < n; j++)
			if (i != j && mesh2->verts[ mesh2->samples[i] ]->dSpanning[ mesh2->samples[j] ] < minDistN)
			{
				minDistN = mesh2->verts[ mesh2->samples[i] ]->dSpanning[ mesh2->samples[j] ];
				minDist = mesh2->verts[ mesh2->samples[i] ]->dSpanningUnn[ mesh2->samples[j] ];
				closest = mesh2->samples[j];
			}
		mesh2->radius += minDist;
		radiusN += minDistN;
		if (minDist < mesh2->minRadius)
		{
			mesh2->minRadius = minDist;
			mesh2->mrIdx = mesh2->samples[i];
			mesh2->mrIdx2 = closest;
		}
	}
	mesh2->radius /= n;
	radiusN /= n;
#ifdef VERBOSE
	cout << "FPS radius on mesh2: " << mesh2->radius << " " << radiusN << "\t(closest pair dist: " << mesh2->minRadius << " from " << mesh2->mrIdx << " to " << mesh2->mrIdx2 << ")\n";
#endif

	//each sample learns its patch by selecting the verts that are closer than radiusN
	float totalPatchSize = 0.0f;
	mesh2->closeEnoughUnn = mesh2->radius; //note that i can use big radius here (not radius/8) 'cos GA will eventualy hit a correct ordering with even a big radius
	for (int i = 0; i < n; i++)
	{
		mesh2->verts[ mesh2->samples[i] ]->patch.clear();//necessary only if adaptiveSampling2() will be called 2+ times; hence redundant here
		for (int v = 0; v < (int) mesh2->verts.size(); v++)
		{
			if (mesh2->verts[ mesh2->samples[i] ]->dSpanningUnn[ v ] <= mesh2->closeEnoughUnn)
				mesh2->verts[ mesh2->samples[i] ]->patch.push_back(v); //mesh2->samples[i] itself is also in the patch
			if (mesh2->verts[ mesh2->samples[i] ]->dSpanningUnn[ v ] <= 2.5f*mesh2->closeEnoughUnn)
				mesh2->verts[ mesh2->samples[i] ]->patch2.push_back(v); //to speed up dijkstraShortestPathsShortcut() i must insert only the ones that are 2*close to this generatorVert; keep them in patch2[]
		}
		totalPatchSize += (int) mesh2->verts[ mesh2->samples[i] ]->patch.size();
	}
#ifdef VERBOSE
	cout << "avg patch size = " << totalPatchSize / n << " vertices\n";
#endif

/*	//since radius-based closeEnoughUnn is an average value, for some patches there'll be extra samples in addition to the normal center sample
	for (int i = 0; i < n; i++)
	{
		int nSamples = 0;
		for (int j = 0; j < (int) mesh2->verts[ mesh2->samples[i] ]->patch.size(); j++)
			if (mesh2->verts[ mesh2->verts[ mesh2->samples[i] ]->patch[j] ]->sample)
				nSamples++;
		if (nSamples > 1)
			cout << nSamples << " samples within the patch of " << mesh2->samples[i] << " (" << (int) mesh2->verts[ mesh2->samples[i] ]->patch.size() << ")\n";
	} ~60 patches have 2+ samples (max 5 samples observed)*/
	
	///////////////////////////// GA stuff ////////////////////////////////////
	//process mesh2 samples based on population[][]
	//i may want to use a different-size population using this: recapture memory from population[][]; population = new int*[newPOP_SIZE]; for (int i = 0; i < newPOP_SIZE; i++) population[i] = new int[CHROM_LEN]; initChromosome should match newPOP_SIZE too
	//initChromosome already set to initChromosome[i] = i; in GA()	
	//random initial maps; easy but population is too bad that it may not evolve to a good result
	for (int i = 0; i < POP_SIZE; i++)
	{















//////////make initializaiton smarter for early/accurater convergence














		//initialize the i'th row of population to initChromosome
		for (int j = CHROM_LEN - 1; j >= 0; j--)
			population[i][j] = initChromosome[j];
		//shuffle the i'th row of population as follows
		for (int j = CHROM_LEN - 1; j > 0; j--)
		{
			int idx = randoi(j+1), //integer in [0, j+1) = [0,j] interval
			//swap idx w/ j'th element
			a = population[i][idx];
			population[i][idx] = population[i][j];
			population[i][j] = a;
		}
	}
	SWAPPER = RANDOM_SWAP; //must use RANDOM_SWAP in swapMutation()

	//start a tournament
#ifdef VERBOSE
	cout << "evolution starts for adaptive sampling (popSize = " << POP_SIZE << ", maxIters = " << MAX_NUM_TOUR << ", mutRate = " << MUTATION_RATE2 << ", xovRate = " << XOVER_RATE2 << ")..\n";
#endif
	float bestFitnessValue = -INF;
	int bestFitnessIdx = -1;	
#ifdef AUTO_EARLY_BREAK
	vector< float > fitnesses;
	int L = 40000;//early-break tournaments when the last L resulting fitnesses are the same (so no change)
#endif
	for (tournamentNo = 0; tournamentNo < MAX_NUM_TOUR; tournamentNo++)
	{
		nMuts = nXovs = nSwapsTotal = nSubstringsTotal = 0; substrSizesTotal = 0.0f;
		if (tournamentNo > 0) //don't evolve in the first iteration to see the fittest of the initial population
			evolvePopulationAS();
		float fitness = getFittestMember2(); //index of fittest member in this population is written to global variable fittestIdx; global populationFitnesses[] is also set as fitnesses of the current population is computed
		if (tournamentNo % 50 == 0)
			if (tournamentNo > 0) //(nMuts != 0 && nXovs != 0)
				//cout << "[" << tournamentNo << "] After " << nMuts << " mutations (" << (float)nSwapsTotal/nMuts << ") and " << nXovs << " crossovers (" << ((float)nSubstringsTotal/(nXovs*N_XOVS)) << ", " << ((float)substrSizesTotal/(nXovs*N_XOVS)) << "), fittest member (" << fittestIdx << ") of this generation has fitness = " << fitness << endl;
				cout << "[" << tournamentNo << "] After " << nMuts << " mutations (" << (float)nSwapsTotal/nMuts << ") and " << nXovs << " crossovers (" << ((float)nSubstringsTotal/(nXovs)) << ", " << ((float)substrSizesTotal/(nXovs)) << "), fittest member (" << fittestIdx << ") of this generation has fitness = " << fitness << endl;
				//if SELECT_BEST_OF_N_XOVS undefined, then delete *N_XOVS just-above; *N_XOVS is there 'cos i overcount nSubstringsTotal N_XOVS times for each xover operation (nXovs); similarly for substrSizesTotal
			else
				cout << "[" << tournamentNo << "] After " << nMuts << " mutations and " << nXovs << " crossovers, fittest member (" << fittestIdx << ") of the initial population has fitness = " << fitness << endl;
//if ((float)nSwapsTotal/nMuts == 0) exit(0);else cout << (float)nSwapsTotal/nMuts << " " << tournamentNo << endl;//for CHROMLEN=100, starts at nSwapsTotal/nMuts=~5, decrease ~.25 each tournament, exits at tourn=~30

#ifdef AUTO_EARLY_BREAK
		fitnesses.push_back(fitness);
		int nRepeats = 0;
		for (int i = (int) fitnesses.size()-1; i >= 0; i--)
			if (fitness == fitnesses[i])
				nRepeats++;
			else
				break;
		if (nRepeats >= L)
		{
#ifdef VERBOSE
			cout << "fitness seems to be stabilized at " << fitness << " (no change in the last " << L << " tournaments); early-breaking\n";
#endif
			bestFitnessValue = fitness;
			bestFitnessIdx = fittestIdx;
			break;
		}		
#endif

		//check whether the fittest population member is the solution
		if (fitness >= (1.0f - 0.0001f)) //0.0001f for precision
		{
			bestFitnessValue = fitness;
			bestFitnessIdx = fittestIdx;
			break;
		}
		else //if solution not found continue evolving
			if (fitness > bestFitnessValue)
			{
				bestFitnessValue = fitness;
				bestFitnessIdx = fittestIdx;
			}			
	}//*/
	cout << "solution found! fitness value: " << bestFitnessValue << "\n\n";

	//set all matchIdxs and sample stuff according to the bestFitnessIdx (and using true as the 2nd parameter) to see the result on screen
	evalSolution2(bestFitnessIdx, true);
	///////////////////////////// GA stuff ends ////////////////////////////////////
}

float GA::getFittestMember2()
{
	//sets the global fittestIdx to the population[][] idx of the best member/solution at this time; also return the fitness value of that best member

//double start_time = get_time();

	fittestIdx = 0;
	float fittestValue = evalSolution2(fittestIdx);
//cout << fittestValue << "e\t";
	populationFitnesses[fittestIdx] = fittestValue; //bookkeep these evaluated fitness values to avoid re-evaluation for the same things in the evolvePopulation()
	for (int i = 1; i < POP_SIZE; i++) //0 is used just-above so start from 1
	{
		//compare fitness of current member with fitness of fittest member so far
		float fitness = evalSolution2(i);
//cout << fitness << "f\t";
		populationFitnesses[i] = fitness;
		if (fitness > fittestValue)
		{
			fittestIdx = i;
			fittestValue = fitness;
		}
//if (i==7)exit(0);
	}
//cout << get_time()-start_time << " secs for 1 getFittestMember2() call\n"; //POPSIZE=1000,CHROMLEN=100, BFS_BASED_PATCHES defined 0.95 secs: 1.2, 1.0, .. 0.95 (converges at tournament 10); less accurate due to approx geodesics
																		   //:) BFS_BASED_PATCHES undefined 1.1 secs w/ dijkstraShortestPaths() and markNeighborhood( patch[] traversal ): 16.6, 3.6, 2.4, .., 1.1 [reaches BFS performance as more patch[]es made available for next tournaments] perfectly accurate
																		   //:) BFS_BASED_PATCHES undefined 0.97 secs w/ dijkstraShortestPathsShortcut() and markNeighborhood( patch[] traversal ): 2.1, 1.1, .., 0.97 (converges at tour 20) (verts[] traversal instead of patch[] traversal is always too slow)
																		   //BFS_BASED_PATCHES undefined 0.7 secs w/ dijkstraShortestPathsEuc() and markNeighborhood( patch[] traversal ): 0.8, 0.73, .., 0.7 (converges at tour 5); less accurate due to approx geodesics
	return fittestValue;
}

float GA::evalSolution2(int c, bool finalCall)
{
	//evaluate the solution given by the c'th chromose, i.e. row c of the population (population[c][*])

	//reload original samples to samples2 and let samples2 get updated below
	mesh2->samples2 = mesh2->samples; //make a copy of samples in samples2 to protect the original samples getting updated at every tournament (this assignment is safe, i.e. verified that samples2[5]=-9 not affect samples[5])

//if(c==0||c==1){for (int i = 0; i < CHROM_LEN; i++) cout << population[c][i] << "x\t";cout<<endl;}

/*int say=0;
if(c==0)
	for (int v=0; v<(int)mesh2->verts.size();v++)
		if (mesh2->verts[v]->sample)
			{cout << v << "i\t";say++;}
if(c==0)
{
	cout << say << "iiiiiiiiiii\n";

	for (int v=0; v<(int)mesh2->samples2.size();v++)
		cout << mesh2->samples2[v] << "q\t";
	cout << endl;
}*/

	for (int v = 0; v < (int) mesh2->verts.size(); v++)
	{
		mesh2->verts[v]->processed = mesh2->verts[v]->sample = false; //processed=true means unavailable (already inserted to samples2[] or made unavailable by being close to a final sample); also reset the sample values
		mesh2->verts[v]->dBFS = INF; //required only if BFS_BASED_PATCHES is defined; otherwise redundant to set it here; no! dijkstraShortestPathsBFS2() also uses it
//mesh2->verts[v]->heapNode = NULL; mesh2->verts[v]->dSpanned = INF; //required only if dijkstraShortestPathsShortcut() will be called; otherwise redundant to set them here [dijkstraShortestPathsShortcut() has a vert traversal now so always redundant here]
//mesh2->verts[v]->newInSamples2 = false; //for efficiency compute dSpanningUnn[] only once using this auxiliary variable [not in use]
	}

	bool robustListUnaffected = true; //set this to true to keep the robustList matches intact, i.e. unaffected by adaptive sampling
	if (robustListUnaffected)
		for (int v = 0; v < (int) mesh2->verts.size(); v++)
			if (mesh2->verts[v]->robust) //everyone sufficiently close to robust v (including v itself) is marked as unavailable (via processed=true) to preserve evenly-spaced sampling
				for (int i = 0; i < (int) mesh2->verts.size(); i++)
					if (mesh2->verts[v]->dSpanningUnn[i] <= mesh2->closeEnoughUnn)
						mesh2->verts[i]->processed = true;	
//mesh2->nChanges = 0;
	int gvSize = (int) mesh2->verts[ mesh2->samples[0] ]->gv.size(); //gvSize'll be same (and equal to robustList.size/2) for all samples
#ifndef USE_CLOSEST_ROBUST_MATCH
	if (gvSize == 0)
		cout << "\nWARNING: GVs are empty\n\n";
#endif
	for (int s = 0; s < CHROM_LEN; s++)
	{
//cout << s << endl;
		int ns = mesh2->samples2[ population[c][s] ]; //ns for next sample
		mesh2->verts[ns]->sample = true; //all verts are reset to sample=false above; so ns begins as a sample (may change below)
		//mesh2->verts[ns]->distToClosestSampleDetermined = true; //for efficiency compute dSpanningUnn[] only once using this auxiliary variable; continue;s below or not entering if (bestPatchVert != ns..) below keeps this true which is OK 'cos those ns's already know the dSpanningUnn to every other vertex
		//mesh2->verts[ns]->newInSamples2 = true; //ns is processed and regarded as newInSamples2 for the upcoming s-iterations ('cos it is in the finalized part of samples2 now); ns.newInSamples2 may be false below if i replace it w/ bestPatchVert (then bestPatchVert.newInSamples2 becomes true) [not in use]

		if (mesh2->verts[ns]->matchIdx == -1) //outlier mesh2.sample will not be processed further 'cos it has no match
		{
			mesh2->verts[ns]->sample = false; //outliers are not samples anymore (discard them in radius computations below; dijkstraShortestPathsBFS2() also requires sample=true/false knowledge)
			continue; //don't update outlier samples
		}

		if (robustListUnaffected && mesh2->verts[ns]->robust) //robust samples are fixed and cannot move if we are in robustListUnaffected mode
			continue; //don't update robust samples

		int bestPatchVert = -1;
#ifdef USE_CLOSEST_ROBUST_MATCH
		//replace ss w/ the vert in ss.patch which has the most compatible geodesic distance when compared with X = verts[ mesh2->verts[ss]->matchIdx ] to closestRobustIn1 geodesic distance
		int closestRobustIn1 = -1, closestRobustIn2 = -1;
		float X = INF;
		for (int i = 0; i < (int) mesh1->robustList.size(); i += 2) //mesh1->robustList = mesh2->robustList so use either of them
			if (mesh1->verts[ mesh1->robustList[i] ]->dSpanning[ mesh2->verts[ns]->matchIdx ] < X) //not matter if i use dSpanning[] or dSpanningUnn[] here
			{
				X = mesh1->verts[ mesh1->robustList[i] ]->dSpanning[ mesh2->verts[ns]->matchIdx ];
				closestRobustIn1 = mesh1->robustList[i];
				closestRobustIn2 = mesh1->robustList[i+1];
			}
if (closestRobustIn1 == -1) { cout << "no -1\n"; exit(0);}

		float minCost = INF; 
		for (int p = 0; p < (int) mesh2->verts[ns]->patch.size(); p++) //patch includes ss itself so ss may not be replaced at all (if it beats all other patch vertices)
		{
			int pv = mesh2->verts[ns]->patch[p];
			if (mesh2->verts[pv]->processed && pv != ns) //processed=true means unavailable to be used as a potential final sample; 2nd condition enables ns itself to be tested as candidate, which gives ns a change to remain as a sample
				continue; //thanks to 2nd condition, ns escapes this continue even if ns.processed=true; this gives me a change to not update the existing sample; seeing this took my 5 weeks :(

			//find geodesic distance of p to desired robustList mesh2.vert (closestRobustIn2)
			float Y = mesh2->verts[ closestRobustIn2 ]->dSpanning[pv]; //must use dSpanning[] if dSpanning[] used above; must use dSpanningUnn[] if dSpanningUnn[] used above
			if (fabs(X - Y) < minCost)
			{
				minCost = fabs(X - Y);
				bestPatchVert = pv;
			}
		}
#else
		//replace ss w/ the vert in ss.patch which has the most compatible gv w/ verts[ mesh2->verts[ss]->matchIdx ]->gv
		float minCost = INF;
		for (int p = 0; p < (int) mesh2->verts[ns]->patch.size(); p++) //patch includes ss itself so ss may not be replaced at all (if it beats all other patch vertices)
		{
			int pv = mesh2->verts[ns]->patch[p];
			if (mesh2->verts[pv]->processed && pv != ns) //processed=true means unavailable to be used as a potential final sample; 2nd condition enables ns itself to be tested as candidate, which gives ns a change to remain as a sample
				continue; //thanks to 2nd condition, ns escapes this continue even if ns.processed=true; this gives me a change to not update the existing sample; seeing this took my 5 weeks :(

			//find geodesic vector of p to robustList mesh2.vertices
			if (mesh2->verts[pv]->gv.empty()) //may be already filled due to a previous chromosome
				for (int i = 0; i < (int) mesh1->robustList.size(); i += 2) //mesh1->robustList = mesh2->robustList so use either of them
					mesh2->verts[pv]->gv.push_back(mesh2->verts[ mesh1->robustList[i+1] ]->dSpanning[pv]);

			float cost = 0.0f;
			for (int i = 0; i < gvSize; i++)
				cost += fabs(mesh1->verts[ mesh2->verts[ns]->matchIdx ]->gv[i] - mesh2->verts[pv]->gv[i]);
			if (cost < minCost)
			{
				minCost = cost;
				bestPatchVert = pv;
			}
		}
#endif
		if (bestPatchVert != ns && bestPatchVert != -1) //rarely bestPatchVert remains -1 since all the patch vertices were processed (made unavailable) by neighboring sample's dijkstra calls
		{
//mesh2->nChanges++;
//if(c==0)cout<<c << " " << s << "\t" << ns << "\t" << bestPatchVert << endl;//cout << "old sample " << ns << " shifted " << mesh2->verts[ns]->dSpanning[bestPatchVert] << " unit to the new sample " << bestPatchVert << endl;
//if(c==0||c==1)cout << population[c][s] << "X\t";
			//update variables accordingly
			mesh2->verts[ns]->sample = false; //not a sample anymore
			//mesh2->verts[ns]->newInSamples2 = false; //not in use

			if (finalCall) //dont alter matchIdx values unless this is the final call; for intermediate calls, setting matchIdx = -1 will make the upcoming calls wrongly continue; above (outlier part)
			{
				mesh2->verts[bestPatchVert]->matchIdx = mesh2->verts[ns]->matchIdx;
				mesh2->verts[ns]->matchIdx = -1;
				mesh1->verts[ mesh2->verts[bestPatchVert]->matchIdx ]->matchIdx = bestPatchVert;
			}
			mesh2->samples2[ population[c][s] ] = bestPatchVert; //population[c][s] was holding ns but now it is replaced by bestPatchVert
			mesh2->verts[bestPatchVert]->sample = true; //just became a sample
			
#ifdef BFS_BASED_PATCHES
			//to prevent vert traversal within CHROM_LEN loop (O(vn) complexity makes it 20 times slower than evalSolution()), use BFS-based marking; even the simplest dijkstraShortestPathsEuc() is unacceptable as it traverses all vertices
			mesh2->dijkstraShortestPathsBFS(bestPatchVert, mesh2->closeEnoughUnn); //unavailables done here (processed=true); note that processed=true above for robust verts uses exact-geodesic dSpanningUnn but this uses approx (not a problem but beware)
			//mesh2->verts[bestPatchVert]->newInSamples2 = true; //not in use
//bestPatchVert is in the patch of ns; so ns.dSpanningUnn[bestPatchVert] <= geodesic-closeEnoughUnn; dijkstraShortestPathsBFS() just-above uses BFS (approx geodesic) to reach closeEnoughUnn so it may return early before reaching ns; so ns.processed may remain false (no problem just a fact)
#else
			//mark everyone sufficiently close to new sample (including bestPatchVert itself) as unavailable to preserve evenly-spaced sampling; to do so first set the patch of bestPatchVert by filling dSpanningUnn[]
			//strategy # 1
			bool immediateReturn = mesh2->dijkstraShortestPaths(bestPatchVert, mesh2->closeEnoughUnn); //since evalSolution2() is called multiple times, bestPatchVert.dSpanningUnn might have been nonNULL which returns this call immediately (w/o setting processed); make sure to set processed values via markNeighborhood() in that case
			if (immediateReturn)
				mesh2->markNeighborhood(bestPatchVert, mesh2->closeEnoughUnn, true); //copy the function just-below to avoid call overhead (compiler quite probably does this inlineing but anyway)
//				for (int i = 0; i < (int) mesh2->verts[bestPatchVert]->patch.size(); i++)
//					mesh2->verts[ mesh2->verts[bestPatchVert]->patch[i] ]->processed = true;//maybe becaue of excessive mesh2-> dereferencing calling markNeighborhood() just-above is slightly faster*/

/*			//strategy # 2 (best)
			mesh2->dijkstraShortestPathsShortcut(bestPatchVert, mesh2->closeEnoughUnn, ns); //full dijkstra just-above is very time consuming; all i actually need is a small dijkstra around ns to find radius below*/

/*			//strategy # 3
			mesh2->dijkstraShortestPathsEuc(bestPatchVert, mesh2->closeEnoughUnn, ns); //set the patch of bestPatchVert by filling dSpanningUnn[] based on euclidean dists, which mostly approximates geodesic well since patches are sufficiently small (still probs may occur when hannd is touching chin); do it only for verts close to generator ns, i.e. ns.patch2, to save further time*/
#endif
		}
		else //current ns will still act as a sample in this chromosome; so set its patch to unavailable just like above
		{
#ifdef BFS_BASED_PATCHES
			mesh2->dijkstraShortestPathsBFS(ns, mesh2->closeEnoughUnn); //unavailables done here (processed=true)
#else
			//same as above but never use bestPatchVert here
			//strategy # 1 # 2 # 3 (common to all) [enable matching strategy in if-part]
			mesh2->markNeighborhood(ns, mesh2->closeEnoughUnn, true);//ns.dijkstraShortestPaths already called for sure (since ns is an existing sample before adaptiveSampling(); so skip dijkstraShortestPaths/Shortcut/Euc() call here
//			for (int i = 0; i < (int) mesh2->verts[ns]->patch.size(); i++)
//				mesh2->verts[ mesh2->verts[ns]->patch[i] ]->processed = true;//copy markNeighborhood() here to avoid call overhead (compiler quite probably does this inlineing but anyway)*/
#endif
		}
	} //end of s
//if(c==0||c==1)cout << "\n\n";

	//new radius based on the current sampling in samples2[]
	float radius2 = 0.0f, minRadius2 = INF;//, radiusN2 = 0.0f;
	int nDegenerates = 0, nAdds = 0, closest;
	for (int i = 0; i < CHROM_LEN; i++) //this radius computation is mandatory for dijkstraShortestPathsBFS() mode; this or just-above (comment-out) can be used for other modes
	{
		if (! mesh2->verts[ mesh2->samples2[i] ]->sample)
			continue; //discard outliers in radius computation
		float minDist = INF;
		for (int j = 0; j < CHROM_LEN; j++) //this radius computation is mandatory for dijkstraShortestPathsBFS() mode; this or just-above (comment-out) can be used for other modes
			if (i != j && mesh2->verts[ mesh2->samples2[j] ]->sample && mesh2->verts[ mesh2->samples2[i] ]->dSpanningUnn[ mesh2->samples2[j] ] < minDist) //2nd condition: discard outliers in radius computation
			{
				minDist = mesh2->verts[ mesh2->samples2[i] ]->dSpanningUnn[ mesh2->samples2[j] ];
				closest = mesh2->samples2[j];
			}
		if (minDist == INF) //dSpanningUnn[] remains INF for even the closest sample if dijkstraShortestPathsShortcut/Euc/BFS() could not assign a valid dSpannigUnn[] to it (shortcuted as it is nonclose)
		{
			//cout << "WARNING: skipping this sample which could not find a close sample to compute radius with\n";
			nDegenerates++; //~30 nDegenerates per chromosome (of size 100) in BFS_BASED_PATCHES defined mode
							//0 nDegenerates per chromosome (of size 100) in BFS_BASED_PATCHES undefined mode strategy # 1
							//~10 nDegenerates per chromosome (of size 100) in BFS_BASED_PATCHES undefined mode strategy # 2; reduces to 0 nDegenerates when i use 2.5f*closeEnoughUnn for patch2[] instead of 2.0f (makes sense to keep patch2[] larger)
							//0 nDegenerates per chromosome (of size 100) in BFS_BASED_PATCHES undefined mode strategy # 3; also see timings in getFittestMember2()
			continue;//*/

/*			//instead of continue; just-above i may find the geodesic from a to b explicitly via dijkstraShortestPathsBFS2()
			minDist = mesh2->dijkstraShortestPathsBFS2(mesh2->samples2[i], closest); //call by ref to closest so that it keeps the closest sample to samples2[i]*/
		}

		radius2 += minDist;
		nAdds++;
		//radiusN2 += minDistN;
#ifdef MIN_RADIUS_IN_USE
		if (minDist < minRadius2)
		{
			minRadius2 = minDist;
			mesh2->mrIdx = mesh2->samples2[i]; //index that gives the min radius
			mesh2->mrIdx2 = closest;
		}
#endif
	}
//cout << nDegenerates << "d\t";

	if (nAdds == 0)
	{
		cout << "too much degenerates/outliers; div by 0\n";
		exit(0);
	}
	radius2 /= nAdds;
	//radiusN2 /= nAdds;
	if (finalCall)
	{
		mesh2->samples = mesh2->samples2; //no more call to here so set the original samples equal to the optimal samples2
#ifdef VERBOSE
		cout << "\nnew FPS radius on mesh2: " << radius2;
#ifdef MIN_RADIUS_IN_USE
		cout << "\t(closest pair dist: " << minRadius2 << " from " << mesh2->mrIdx << " to " << mesh2->mrIdx2 << ")\n";//cout << radiusN2 << endl;
#else
		cout << endl;
#endif
#endif

		for (int i = 0; i < (int) mesh2->samples.size(); i++)
		{
			if (! mesh2->verts[ mesh2->samples[i] ]->sample)
				continue; //don't compute distance for outliers as they'll be omitted in resultToFile() anyway (via if(v2==-1)continue; at top)
			if (mesh2->verts[ mesh2->samples[i] ]->dSpanning)
				delete [] mesh2->verts[ mesh2->samples[i] ]->dSpanning;
			if (mesh2->verts[ mesh2->samples[i] ]->dSpanningUnn)
				delete [] mesh2->verts[ mesh2->samples[i] ]->dSpanningUnn;
			mesh2->verts[ mesh2->samples[i] ]->dSpanning = mesh2->verts[ mesh2->samples[i] ]->dSpanningUnn = NULL;
			mesh2->dijkstraShortestPaths(mesh2->samples[i]);
		}
	}
//if (c % 100 == 0)cout << mesh2->nChanges << "\t";// << radius2 << "\n\n";

#ifdef MIN_RADIUS_IN_USE
	float denom = (mesh2->minRadius > minRadius2 ? mesh2->minRadius : minRadius2);
	return 1.0f - fabs(mesh2->minRadius - minRadius2) / denom;
#else
	float denom = (mesh2->radius > radius2 ? mesh2->radius : radius2);
	return 1.0f - fabs(mesh2->radius - radius2) / denom;
#endif
}

void GA::evolvePopulationAS()
{
	//simplified version of evolvePopulation3() to improve population

//double start_time = get_time();

/*	instead of creating and deleting them here, do it once in the initPopulation() to save micro time
	int** newPopulation = new int*[BADPART_SIZE], * childChromo = new int[CHROM_LEN];
	for (int i = 0; i < BADPART_SIZE; i++)
		newPopulation[i] = new int[CHROM_LEN];
	//sort the population[][] w.r.t. fitness of each member; sorted result is written back to population[][]
	int* sortedIdxs = new int[POP_SIZE];
	float* fitnessVals = new float[POP_SIZE];*/
	int a = -1, b = -1, winner, loser;
	for (int c = 0; c < POP_SIZE; c++)
	{
		//save super time by using the already-computed fitness values in populationFitnesses[] instead of recomputing by evalSolution2(c); perfectly accurate 'cos population[] not updated 'till the last getFittestMember2() call
		fitnessVals[c] = populationFitnesses[c]; //evalSolution2(c);
		sortedIdxs[c] = c;
	}
	insertionSortGA(fitnessVals, POP_SIZE, sortedIdxs); //call by ref to fitnessVals (redundant) and sortedIdxs (output to be used below)
	//since insertionSortGA did a descending sorting on fitness values, first GOODPART_SIZE entries of sortedIdxs[] point to the goodpart (= fit part) of population[][] and the remaining entries (POP_SIZE-GOODPART_SIZE) are the badpart
	//transfer sorted members into tmpPopulation and then from tmpPopulation to population[][]
	for (int c = 0; c < POP_SIZE; c++)
	{
		for (int i = 0; i < CHROM_LEN; i++)
			tmpPopulation[c][i] = population[ sortedIdxs[c] ][i];
		tmpFitnesses[c] = populationFitnesses[ sortedIdxs[c] ];
	}
	//from tmpPopulation to population[][]
	for (int c = 0; c < POP_SIZE; c++)
	{
		for (int i = 0; i < CHROM_LEN; i++)
			population[c][i] = tmpPopulation[c][i];
		//transfer tmpFitnesses[] to populationFitnesses[] too
		populationFitnesses[c] = tmpFitnesses[c];
	}
	
	for (int i = 0; i < BADPART_SIZE; i++)
	{
		//possibly do some crossover
		if (randof() < XOVER_RATE2)
		{
			//select parents by pulling 2 population members at random from the goodpart of the population, which is the first GOODPART_SIZE members from index 0 to GOODPART_SIZE-1
			do
			{			
				a = randoi(GOODPART_SIZE); b = randoi(GOODPART_SIZE); //[0, GOODPART_SIZE) = [0,GOODPART_SIZE-1] interval
			} while (a == b);
		
			//have a fight and see who has best genes
			if (populationFitnesses[a] > populationFitnesses[b]) //use bookkeeped populationFitnesses[] for faster execution (still accurate 'cos population[] not updated 'till i leave this for-loop)
			//if (evalSolution2(a) > evalSolution2(b)) makes no sense 'cos population[] not updated 'till i leave this for-loop, meaning that initial populationFitnesses[] values are safe to use
			{
				winner = a;
				loser = b;
			}
			else
			{
				winner = b;
				loser = a;
			}
		
			for (int j = 0; j < CHROM_LEN; j++)
				childChromo[j] = population[loser][j]; //needs to be inited to loser to make anotherCrossover2() work accurately; and needs to be in this loop to prevent duplicates in the resulting call-by-ref childChromo[]
			anotherCrossover3(winner, loser, childChromo); //call by reference on childChromo[]
		}
		else
			for (int j = 0; j < CHROM_LEN; j++)
				childChromo[j] = population[i+GOODPART_SIZE][j]; //no xover so keep the current bad member in the newPopulation[][] using this childChromo[]
		//child obtained by xovering winner and loser (or obtained by not xovering) is saved in the next idx of newPopulation
		for (int j = 0; j < CHROM_LEN; j++)
			newPopulation[i][j] = childChromo[j];
	}

	//copy newPopulation into badpart of the population
	for (int p = 0; p < BADPART_SIZE; p++)
		for (int c = 0; c < CHROM_LEN; c++)
			population[p+GOODPART_SIZE][c] = newPopulation[p][c]; //population[GOODPART_SIZE], population[GOODPART_SIZE+1], population[GOODPART_SIZE+2], .. are the bad members of the population (placed after the goodpart; hence GOODPART_SIZE+something) and will be replaced by newPopulation[] computed above

	//possibly mutate the new population a bit to add some new genetic material
	for (int i = 1; i < POP_SIZE; i++) //not select only the BADPART for mutation 'cos BADPART is now kindda fixed after xovers above; so select any chromosome for mutation (from POP_SIZE)
									   //start i=1 'cos i=0 is the fittest member of the population due to sorting above; it is impossible to change it; so the fitness value will be non-decreasing in each tournament
		if (randof() < MUTATION_RATE2)
			swapMutation(i); //global population[i][*] updated after this, i.e. row i is updated only (hence mutates random members of the updated population[])

/*	//memo recapture (arrays allocated with new[] must be deallocated with delete[])
	for (int i = 0; i < BADPART_SIZE; i++) delete [] newPopulation[i]; delete [] newPopulation;
	for (int i = 0; i < POP_SIZE; i++) delete [] tmpPopulation[i]; delete [] tmpPopulation;
	delete [] childChromo;
	delete [] sortedIdxs;
	delete [] fitnessVals;*/
//cout << get_time()-start_time << " secs for 1 evolovePopulationAS() call\n"; //POPSIZE=1000,CHROMLEN=100, 0.005 secs (very fast)
}
