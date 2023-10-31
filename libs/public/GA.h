#ifndef _GA_H_
#define _GA_H_

#include "Mesh.h"

/*#include <windows.h> //for get_time()	//disabling for full portability of the public release
inline double get_time()
{
	LARGE_INTEGER t, frequency;
	QueryPerformanceCounter(&t);
	QueryPerformanceFrequency(&frequency);
	return (double)t.QuadPart/(double)frequency.QuadPart;
}//*/


//execution modes
#define COMPUTE_ROBUST_LIST				1
#define GA_WITH_ROBUST_LIST				2
#define GA_WITHOUT_ROBUST_LIST			3
#define GA_WITH_ROBUST_LIST_HIGHRESO	4

//INIT_POP_METHOD constants
#define AGD_BASED_INIT	1			//instead of random initial maps, create initial maps based on a descriptor value, average geodesic distance (AGD) in this case
#define GV_BASED_INIT	2			//instead of random initial maps, create initial maps based on a descriptor value, geodesic vector (GV) in this case; this is better than AGD [not in use]
#define	RND_INIT		3			//random initial maps

//swapMutation constants
#define RANDOM_SWAP		1	//swap random genes within the chromosomes (guarantees no-duplicates but does nothing for accuracy)
#define GV_BASED_SWAP	2	//swap genes w.r.t. geodesic vector compatibility; worst2 is guaranteed to be compatible with mesh1.worst1 [most accurate one]
#define RANK_BASED_SWAP	3	//swap genes w.r.t. agdORder ranking compatibility; worst2 is guaranteed to be compatible with mesh1.worst1 [when gv not available this performs better than RANDOM_SWAP]

//other constants
#define MUTATION_RATE	0.85f		//frequency of mutations happening
#define XOVER_RATE		0.85f		//frequency of cross-overs happening [this is the bottleneck in evolvePopulation3() when BADPART_SIZE is high, e.g., = 2000/2=1000; so keep the rate lower; no!!!!! with this i converge sooner; besides it makes the map good]
#define MAX_NUM_TOUR	5000		//max number of tournaments to be played during evolution
#define N_XOVS			5			//try N_XOVS substrings for SELECT_BEST_OF_N_XOVS case; valid only for SELECT_BEST_OF_N_XOVS case
//#define BADPART_SIZE	POP_SIZE/4	//this many members are tagged as bad members (to be potentially replaced); valid only for evolvePopulation3() mode [using POP_SIZE/2 makes 1000-sample 2000-popsize case very slow (5 hours); this is the bottleneck so keep it light]
#define BADPART_SIZE	POP_SIZE/2	//this many members are tagged as bad members (to be potentially replaced); valid only for evolvePopulation3() mode
#define GOODPART_SIZE	POP_SIZE-BADPART_SIZE //# good members; never change this equation; valid only for evolvePopulation3() mode
#define MUTATION_RATE2	0.85f		//for adaptiveSampling2() (above were for shape correspondence)
#define XOVER_RATE2		0.85f		//for adaptiveSampling2()

//other preprocessors
#define AUTO_EARLY_BREAK			//define this to early-break tournaments when the last L results are the same (so no change)
#define DO_ROULETTE_SELECTION		//an algorithm for cross-over pair selection; valid only for evolvePopulation2() mode
#define SELECT_BEST_OF_N_XOVS		//instead of selecting the first random substring, try N_XOVS substrings and evaluate each resulting child[]'s distortion, and select the best/mindistortion one
//#define USE_GRD_DISTORTION			//define this to use ground-truth distortion (instead of the isometric distortion) during the algorithm; this is for testing only 'cos in real-life we don't know the ground-truth; both GV_BASED_SWAP & RANK_BASED_SWAP act the same in this mode
//#define MULTIPLE_REINIT				//define this to re-initialize GA with new population and new evolution; also try re-evolution on the same fixed initial population (this is not enabled yet); reinits continue until the bestSoFar optimal value is seen twice

class GA
{
private:
	int MODE, //mode of execution
		INIT_POP_METHOD, //population initialization method
		CHROM_LEN, //one solution is of this length, i.e. one candidate mapping will have this many matches
		POP_SIZE, //population size, i.e. this many chromosomes will be used in evolution
		SWAPPER; //strategy in swapMutation()
	int** population; //each row of this matrix is 1 candidate solution, i.e. set of matched vertices in mesh2 corresponding to mesh1's sample0, sample1, sample2, .., samplen vertices
	int* initChromosome; //initial chromose will be used to generate the initial population to be evolved
	float* populationFitnesses, //store these evaluated fitness values to avoid re-computation of the same things in the evolvePopulation*()
		  substrSizesTotal; //debugging info
	int fittestIdx, //fittest member in the end of the evolve() iterations will be my optimal (or near-optimal) solution
		nMuts, nXovs, nSwapsTotal, nSubstringsTotal, //debugging info
		windowSize; //this vicinity of compatible ranks will be used as initial match candidates
	bool robustListInUse, forceRobustMatchesInInit; //robust matches in robustList must appear in all the initial population chromosomes
	Mesh* mesh1, * mesh2;
	int** newPopulation, ** tmpPopulation, * childChromo, * bestChildChromo, * sortedIdxs; float* fitnessVals, * tmpFitnesses; //initializations for evolvePopulation*()

	void computeInitialMatchCandidates();
	void initPopulation(bool insertExistingMaps);
	float getFittestMember();
	float evalSolution(int c);
	float evalSolution2(int c, bool finalCall = false);
	float evalSolution3(int c, float& grd);
	void evolvePopulation(int bestFitnessIdx);
	void evolvePopulation2(int bestFitnessIdx);
	void evolvePopulation3();
	void swapMutation(int c);
	void orderOneCrossover(int wi, int lo);
	void anotherCrossover(int wi, int lo);
	void anotherCrossover2(int wi, int lo, int* child);
	void anotherCrossover3(int wi, int lo, int* child);
	int rouletteSelection();
	bool goodSubstring(int s, int e, int wi);
	void adaptiveSampling2();
	float getFittestMember2();	
	void evolvePopulationAS();
public:
	GA(Mesh* m1, Mesh* m2, int mode, int N);
	void go(bool bruteForceSolution, bool removeOutliers, bool insertExistingMaps = false, bool doAdaptiveSampling = false);
	void go2();
};

#endif
