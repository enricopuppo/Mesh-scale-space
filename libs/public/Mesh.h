#ifndef _MESH_H_
#define _MESH_H_

#define _CRT_SECURE_NO_WARNINGS //i won't use fopen_s, sprintf_s, etc; so, don't warn me
#include <iostream>
#include <vector>
#include <set>
#include <queue>
using namespace std;

/*#include "Eigen/Dense" //dense matrices
#include "Eigen/Sparse" //sparse matrices
using namespace Eigen;*/

#include "FibHeap.h"

#define PI							3.141592653f
#define INF							(float) 1e38 //100000000.0f //largest float is 3.4x10^38; 1e+38 (or 1e38) also works (1e39 makes overflow; prints 1.#INF) but then treats as a double and gives warning when i compare/assign w/ a float; solved using (float) casting
#define EXP							2.718281828f //exponential e to be used in Gaussian kernel
#define VERBOSE //define this to print more information during the whole execution (development); undefine to print out the very very basics (release)

struct Triangle
{
	//tris[idx] is this tri
	int idx;

	//idx to verts forming this tri
	int v1i, v2i, v3i;

		//normal and unit normal
	float* normal, * unormal;

	//unsigned area
	float area;

	Triangle(int i, int i1, int i2, int i3) : idx(i), v1i(i1), v2i(i2), v3i(i3) {};

	void setNormal(float* v1, float* v2, float* v3)
	{
		normal = new float[3];
		unormal = new float[3];

		//normal is the A, B, C triple of Ax + By + Cz + D = 0; and unormal is the unitized version (direction)
		normal[0] = v1[1]*(v2[2]-v3[2]) + v2[1]*(v3[2]-v1[2]) + v3[1]*(v1[2]-v2[2]);
		normal[1] = v1[2]*(v2[0]-v3[0]) + v2[2]*(v3[0]-v1[0]) + v3[2]*(v1[0]-v2[0]);
		normal[2] = v1[0]*(v2[1]-v3[1]) + v2[0]*(v3[1]-v1[1]) + v3[0]*(v1[1]-v2[1]);
		//D = -v1[0]*(v2[1]*v3[2] - v3[1]*v2[2]) - v2[0]*(v3[1]*v1[2] - v1[1]*v3[2]) - v3[0]*(v1[1]*v2[2] - v2[1]*v1[2]);
		//D is needed only for distance from a point to tri computation; not needed in snake 3d

		//length of this vector
		float length = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);

		//note that, contractToV2 or middle may produce temporary 0-area (due to collinearity) tris
		if (length > 0)
		{
			unormal[0] = normal[0] / length;
			unormal[1] = normal[1] / length;
			unormal[2] = normal[2] / length; //frequent case
		}
		else
			unormal[0] = unormal[1] = unormal[2] = 0.0f; //rare case
	}

	void setArea(float* v1, float* v2, float* v3)
	{
		// If a triangle is defined by 3 points, say p, q and r, then
		// its area is 0.5 * length of ((p - r) cross (q - r)) [See Real-Time Rendering book, Appendix A]
/*		SbVec3f p_r = v2 - v1;
		SbVec3f q_r = v3 - v1;
		SbVec3f cross = p_r.cross(q_r);*/
		float* p_r = new float[3];
		for (int i = 0; i < 3; i++)
			p_r[i] = v2[i] - v1[i];

		float* q_r = new float[3];
		for (int i = 0; i < 3; i++)
			q_r[i] = v3[i] - v1[i];

		float* cross = new float[3];
		cross[0] = p_r[1]*q_r[2] - p_r[2]*q_r[1];
		cross[1] = p_r[2]*q_r[0] - p_r[0]*q_r[2];
		cross[2] = p_r[0]*q_r[1] - p_r[1]*q_r[0];

		//length of this vector
		float length = sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);

		area = 0.5f * length;

		//delete a, b, c; ignores b and c
		delete p_r;
		delete q_r;
		delete cross;
	}
};

struct Vertex
{
	//verts[idx] is this vert
	int idx,
		matchIdx, //idx of the matching vert on the other mesh
		nMatches, //defined for mesh2 verts only and when DO_EM_ALGO is defined; 1+ for mesh2.samples; it is the # of matches of this vert and updated only in greedyForBaseUpdate(), only place for 1-to-many matchings
		agdOrder, //ranking of this vert w.r.t. agd value
		newPosition; //adaptiveSampling3() will bring some samples into their new positions, which is merely another index (not an x-y-z point in 3d space)

	//spatial coordinates (on traditional Cartesian frame of xyz-axes)
	float* coords, * spectralK; //spectral coords for DO_EM_ALGO or LMDS_INIT defined

	//normal and unit normal and average geodesic distance value
	float* normal, * unormal, * color, * sphereColor, agd, ecc, diso, dgrd, drank; //individual isometric/groundtruth distortion of the match ending at this vert; ecc for eccentricity, which is the greatest geodesic b/w this vert and any other vert

	//idx to vert neighbors of this vert (use set for efficient duplicate entry prevention)
	typedef set< int, std::less< int > > vertNeighborsIS; vertNeighborsIS vertNeighborsSet;
	
	//idx to tri neighbs and to edges incident to this vertex
	vector< int > triNeighbors, edgeList, initMatchCandids, //idx to samples[] (in interval [0,samples.size)) to point out possible initial matches for this vertex
				  sampleNghbs, //inited only for samples and only when DO_EM_ALGO is defined; list of surrounding samples of this sample
				  patch, patch2; //verts close to this sample; defined for mesh2.samples only; patch2 is more relaxed/large by using 2*close as threshold (used only in dijkstraShortestPathsShortcut())
	vector< float > gv; //geodesic vector storing geodesic distances to all robust vertices (to all samples for interpolateCoarseMap() only)

	//dSpanningUnn spans whole mesh so that [9].dSpanningUnn[4] gives g(verts[9], verts[4]) in the original scale; normalized into [0,1] version is dSpanning[] s.t. max geodesic over the surface is 1
	float* dSpanningUnn, * dSpanning, dSpanned, dBFS;

	//corresponding heapNode for this vert to be used in DecreaseKey(heapNode, newKey)
	FibHeapNode* heapNode;

	//auxiliary variable for sampling and agd-based initialization and efficiency
	bool processed, sample, unchanged, robust; //in robust traversal list, namely the robustList

	Vertex(int i, float* c) : idx(i), coords(c), color(NULL), sphereColor(NULL), agd(0.0f), ecc(-1.0f), dSpanningUnn(NULL), matchIdx(-1), robust(false), sample(false), dBFS(INF) {};

	bool addVertNeighbor(int vertInd)
	{
		//add vertInd as a vert neighb of this vert if it won't cause a duplication

		pair< vertNeighborsIS::const_iterator, bool > pa;
		//O(log N) search for duplicates by the use of red-black tree (balanced binary search tree)
		pa = vertNeighborsSet.insert(vertInd);
		//pa.second is false if item already exists, and true if insertion succeeds
		return pa.second;
	}
	
	void removeVertNeighbor(int toBeRemoved)
	{
		//size_type erase (const key_type& x): Deletes all the elements matching x. Returns the 
		//number of elements erased. Since a set supports unique keys, erase will always return 1 or 0

		vertNeighborsSet.erase(toBeRemoved);
	}

	void removeTriNeighbor(int t)
	{
//this fnctn could've been used by contract/split/flip as well but i wrote it later; so, only createSphere() uses it

		int removeIdx;
		for (removeIdx = 0; removeIdx < (int) triNeighbors.size(); removeIdx++)
			if (triNeighbors[removeIdx] == t)
				break;

		if (removeIdx == triNeighbors.size()) { cout << "tri" << t << " not in vert" << idx << ".triNeighbs!\n"; exit(0); }

		triNeighbors.erase(triNeighbors.begin() + removeIdx);
	}

	void setNormal(float* nnormal)
	{
		normal = new float[3];
		unormal = new float[3];
		for (int i = 0; i < 3; i++)
			normal[i] = nnormal[i];

		//length of this vector
		float length = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
		//if (length >= ROUNDING_ERR) //0.00001f thrshld is bad 'cos 0.0000854606 and, undesiredly, 0.0000854606 >= 0.00001f holds
		if (length > 0)
		{
			unormal[0] = normal[0] / length;
			unormal[1] = normal[1] / length;
			unormal[2] = normal[2] / length; //frequent case
		}
		else
		{
			unormal[0] = unormal[1] = unormal[2] = 0.0f; //rare case
			cout << idx << " has 0000 vertex normal!\t" << length << endl; //happens if this vert has only 2 tri neighbs which are temporary 0-area
		}
	}
};

struct Edge
{
	//edges[idx] is this edge
	int idx;

	//idx to endpnt verts of this edge
	int v1i, v2i;

	//distance between v1i & v2i
	float length;

	Edge(int i, int i1, int i2, float l) : idx(i), v1i(i1), v2i(i2), length(l) {};
};

class Mesh
{
public:
	//use vector to allow random access of elements in O(1) time (to maintain index consistency, never physically erase elements)
	vector< Triangle* > tris;
	vector< Vertex* > verts;
	vector< Edge* > edges;
	vector< int > robustList, //robust matches in this list; typically very few and very confident matches are stored here
				  samples2, samples; //evenly-spaced samples

	bool secondMesh;

	float minGeoDist, maxGeoDist, maxEucDist, avgEdgeLen, //to get an idea about the scale of the object
		  isometricDistortion, minIsoDisto, radius, radiusN, minRadius, minRadiusN, closeEnoughUnn, nChanges;

	int windowSize, * vIdxsG, mrIdx, mrIdx2, worstGrd, id;

	//color and center of mass
	float* color, * com, * avgLocalRadius; //avg local radius for the i'th sample

	Mesh(bool sm, int i) : secondMesh(sm), id(i), isometricDistortion(9.0f), radius(-1.0f) { color = new float[3]; color[0] = color[1] = color[2] = 0.7f; };
	void loadObj(char* meshFile);
	void loadOff(char* meshFile);
	void loadPly(char* meshFile);
	void computeEvenlySpacedSamples(int N, bool earlyBreak = false, int startVertex = -1);
	void computeSamples(int N);
	void computeAGD(bool robustListInUse);
	void computeGV();
	float getIsometricDistortion(Mesh* mesh2, vector< int > tmpSafeCorrespTry);
	float getIsometricDistortionRL(Mesh* mesh2, vector< int > tmpSafeCorrespTry);
	float getRankDistortion(Mesh* mesh2, vector< int > tmpSafeCorrespTry);
	float getGroundDistortion(Mesh* mesh2, vector< int > tmpSafeCorrespTry, bool special = false);
	float getGroundDistortionFull(Mesh* mesh2, vector< int > tmpSafeCorrespTry);
	void additionalRotations();
	void computeCenterOfMass();
	void resultToFile(char* fName, Mesh* mesh2, bool removeOutliers, bool rlComputeMode);
	void resultFromFile(char* fName, Mesh* mesh2);
	void resultFromBIMFile(char* fName, Mesh* mesh2);
	void appendToFile(char* fName, Mesh* mesh2);
	void refillCorresp(Mesh* mesh2);
	void fillRobustTraversalList(char* fName);
	void adaptiveSampling(Mesh* mesh2, bool keyboardSampling);
	int adaptiveSamplingStep(Mesh* mesh2, int step);
	void adaptiveSampling2(Mesh* mesh2);
	void adaptiveSampling3(Mesh* mesh2, bool removeOutliers);
	void interpolateCoarseMap(Mesh* mesh2);	
	void bruteForceMap(Mesh* mesh2, int r);
	bool dijkstraShortestPaths(int sourceVert, float close = -1.0f);
	bool dijkstraShortestPathsShortcut(int sourceVert, float close, int generatorVert);
	void dijkstraShortestPathsEuc(int sourceVert, float close, int generatorVert);
	void dijkstraShortestPathsBFS(int sourceVert, float close);
	float dijkstraShortestPathsBFS2(int sourceVert, int & closestSample);
	void markNeighborhood(int sourceVert, float close, bool processedValue);
private:
	//edge information for display only
	float minEdgeLen, maxEdgeLen, edgeLenTotal;	
	vector< int > corresp, emCorresp; //consecutive 2 entries make 1 match of the whole corresp, e.g., corresp[0]-corresp[1], corresp[2]-corresp[3], where corresp[] is an idx to verts[] not samples[]
	void addVertexND(float* c);
	void addTriangle(int v1i, int v2i, int v3i);
	void addEdge(int v1i, int v2i);
	inline float calcTriArea(int t);
	int* getTrianglesSharedBy(Edge* bigEdge, int v1i = -1, int v2i = -1);
	int getEdgeSharedBy(int vi, int vj);
	void updateEdgeLengths();
	void updateNormals();
	void computeBoundingBox();	
	int getAnExtremeVert();
	void sortAGDs();
	void computeVertColors();
	void computeRadius(bool print = true);
	void computeRadiusEff();
	int computeSampleNghbs(int sample, int i);
	void computeAvgLocalRadius(int sample, int i);
	void computeRadiusBasedOn(int va, int s);
	float evalObjective(Mesh* mesh2, float alpha, bool firstCall = false);
	float evalObjectiveOnVert(Mesh* mesh2, float alpha, int vert, int s);
	float evalObjectiveWithCandidate(Mesh* mesh2, float alpha, int candidate, int sidx, int midx);
	float evalObjective3(Mesh* mesh2, float alpha, float& iso3, float& grd3);
};

#endif
