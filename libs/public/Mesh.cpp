#include "Mesh.h"

void Mesh::loadObj(char* meshFile)
{
#ifdef VERBOSE
	cout << "Mesh initializing (to model in " << meshFile << " file)...\t";
#else
	if (secondMesh)
		cout << "mesh2: " << id << "\n";
	else
		cout << "mesh1: " << id << "\t";
#endif

	FILE* fPtr;
	if (! (fPtr = fopen(meshFile, "r")))
	{
		cout << "cannot read " << meshFile << endl;
		exit(0);
	}

	char type;	//type of line: vertex line (v) or face line (f)
	float a, b, c;	//for face lines casting these to int will suffice
	minEdgeLen = INF;
	maxEdgeLen = -INF;
	edgeLenTotal = 0.0f;
	while (fscanf(fPtr, "%c %f %f %f\n", &type, &a, &b, &c) != EOF) //go till the end of file
		if (type == 'v')
		{
			//none of verts are removed/duplicated initially 'cos i read .obj
			float* coords = new float[3];
			coords[0] = a;
			coords[1] = b;
			coords[2] = c;
			addVertexND(coords); //ND: no duplicate check
		}
		else //if (type == 'f')
			addTriangle((int) a - 1, (int) b - 1, (int) c - 1); //-1: obj indices start at 1
	avgEdgeLen = edgeLenTotal/((int) edges.size());
	
#ifdef VERBOSE
	cout << "Mesh " << secondMesh << " in collection has " << (int) tris.size() << " tris, " << (int) verts.size() << " verts, " << (int) edges.size() << " edges\n\n";
#endif
}
void Mesh::loadOff(char* meshFile)
{
#ifdef VERBOSE
	cout << "Mesh initializing (to model in " << meshFile << " file)...\t";
#else
	if (secondMesh)
		cout << "mesh2: " << id << "\n";
	else
		cout << "mesh1: " << id << "\t";
#endif

	FILE* fPtr;
	if (! (fPtr = fopen(meshFile, "r")))
	{
		cout << "cannot read " << meshFile << endl;
		exit(0);
	}

	char off[25];
	fscanf(fPtr, "%s\n", &off); //cout << off << " type file\n";

	float a, b, c, d;	//for face lines and the 2nd line (that gives # of verts, faces, and edges) casting these to int will suffice
	fscanf(fPtr, "%f %f %f\n", &a, &b, &c);
	int nVerts = (int) a, v = 0;

	minEdgeLen = INF;
	maxEdgeLen = -INF;
	edgeLenTotal = 0.0f;
	while (v++ < nVerts) //go till the end of coords section
	{
		fscanf(fPtr, "%f %f %f\n", &a, &b, &c);

		//none of verts are removed/duplicated initially 'cos i read .obj
		float* coords = new float[3];
		coords[0] = a;
		coords[1] = b;
		coords[2] = c;
		addVertexND(coords); //ND: no duplicate check
	}

	//verts ready, time to fill triangles
	while (fscanf(fPtr, "%f %f %f %f\n", &d, &a, &b, &c) != EOF) //go till the end of file
		addTriangle((int) a, (int) b, (int) c); //no -1 'cos idxs start from 0 for off files
	avgEdgeLen = edgeLenTotal/((int) edges.size());

	if (secondMesh)
		additionalRotations();

	computeCenterOfMass();
	computeBoundingBox();
	updateNormals();

#ifdef VERBOSE
	cout << "min <= avg <= maxEdgeLen: " << minEdgeLen << " <= " << avgEdgeLen << " <= " << maxEdgeLen << "\n";
	cout << "Mesh " << secondMesh << " in collection has " << (int) tris.size() << " tris, " << (int) verts.size() << " verts, " << (int) edges.size() << " edges\n\n";
#endif
}
void Mesh::loadPly(char* meshFile)
{
#ifdef VERBOSE
	cout << "Mesh initializing (to model in " << meshFile << " file)...\t";
#else
	if (secondMesh)
		cout << "mesh2: " << id << "\n";
	else
		cout << "mesh1: " << id << "\t";
#endif

	FILE* fPtr;
	if (! (fPtr = fopen(meshFile, "r")))
	{
		cout << "cannot read " << meshFile << endl;
		exit(0);
	}

	char ply[25], nVertsStr[25];
	fscanf(fPtr, "%s\n", &ply); //skip "ply" at line 1
	fscanf(fPtr, "%s %s %s\n", &ply, &ply, &ply); //skip "format ascii 1.0" at next line
	fscanf(fPtr, "%s", &ply);
	if (strcmp(ply, "comment") == 0) //supporting 0 or 1 line comments only
	{
		fscanf(fPtr, "%s %s\n", &ply, &nVertsStr); //skip "comment VCGLIB generated" at next line
		fscanf(fPtr, "%s %s %s\n", &ply, &ply, &nVertsStr); //skip "element vertex 12500" at next line
	}
	else
		fscanf(fPtr, "%s %s\n", &ply, &nVertsStr); //skip "element vertex 12500" at next line; element already fread
	int nVerts = atoi(nVertsStr), v = 0;
	fscanf(fPtr, "%s %s %s\n", &ply, &ply, &ply); //skip "property float x" at next line
	fscanf(fPtr, "%s %s %s\n", &ply, &ply, &ply); //skip "property float y" at next line
	fscanf(fPtr, "%s %s %s\n", &ply, &ply, &ply); //skip "property float z" at next line
	fscanf(fPtr, "%s %s %s\n", &ply, &ply, &ply); //skip "element face 25000" at next line
	fscanf(fPtr, "%s %s %s %s %s\n", &ply, &ply, &ply, &ply, &ply); //skip "property list uchar int vertex_indices" at next line
	fscanf(fPtr, "%s\n", &ply); //skip "end_header" at next line

	float a, b, c, d;	//for face lines and the 2nd line (that gives # of verts, faces, and edges) casting these to int will suffice
	minEdgeLen = INF;
	maxEdgeLen = -INF;
	edgeLenTotal = 0.0f;
	while (v++ < nVerts) //go till the end of coords section
	{
		fscanf(fPtr, "%f %f %f\n", &a, &b, &c);

		//none of verts are removed/duplicated initially 'cos i read .obj
		float* coords = new float[3];
		coords[0] = a;
		coords[1] = b;
		coords[2] = c;
		addVertexND(coords); //ND: no duplicate check
	}

	//verts ready, time to fill triangles
	while (fscanf(fPtr, "%f %f %f %f\n", &d, &a, &b, &c) != EOF) //go till the end of file
		addTriangle((int) a, (int) b, (int) c);
	avgEdgeLen = edgeLenTotal/((int) edges.size());

#ifdef VERBOSE
	cout << "Mesh " << secondMesh << " in collection has " << (int) tris.size() << " tris, " << (int) verts.size() << " verts, " << (int) edges.size() << " edges\n\n";
#endif
}
void Mesh::computeBoundingBox()
{
	//info on bounding box of this mesh

	float minX, minY, minZ, maxX, maxY, maxZ; //bbox to be used in center-of-bbox calculation
	for (int v = 0; v < (int) verts.size(); v++)
		if (v == 0)
		{
			minX = maxX = verts[v]->coords[0];	minY = maxY = verts[v]->coords[1];	minZ = maxZ = verts[v]->coords[2];
		}
		else
		{
			if (verts[v]->coords[0] < minX)	minX = verts[v]->coords[0]; else if (verts[v]->coords[0] > maxX)	maxX = verts[v]->coords[0];
			if (verts[v]->coords[1] < minY)	minY = verts[v]->coords[1]; else if (verts[v]->coords[1] > maxY)	maxY = verts[v]->coords[1];
			if (verts[v]->coords[2] < minZ)	minZ = verts[v]->coords[2]; else if (verts[v]->coords[2] > maxZ)	maxZ = verts[v]->coords[2];
		}
	float cobX = (minX+maxX) / 2.0f, cobY = (minY+maxY) / 2.0f, cobZ = (minZ+maxZ) / 2.0f;
//	cout << "bbox.center: (" << cobX << ", " << cobY << ", " << cobZ << "); "
//		 << "bbox.(w,h,d): (" << maxX-minX << ", " << maxY-minY << ", " << maxZ-minZ << ")\n";	
}
void Mesh::computeCenterOfMass()
{
	//center of mass of this mesh that may be used elsewhere

	com = new float[3];
	com[0] = com[1] = com[2] = 0.0f;
	for (int v = 0; v < (int) verts.size(); v++)
	{						
		com[0] += verts[v]->coords[0];
		com[1] += verts[v]->coords[1];
		com[2] += verts[v]->coords[2];
	}
	for (int i = 0; i < 3; i++)
		com[i] /= ((float) verts.size());
	//cout << "center-of-mass: " << com[0] << ", " << com[1] << ", " << com[2] << " for mesh" << frameID << "\n";
}

void Mesh::additionalRotations()
{
	//apply additional x- y- and/or z-axis rotations to mesh

//	cout << "additional rotations for visual convenience..\n";
	float x = 0, y = 0, z = 0, theta = 90.0f * (PI / 180.0f); //radian rotation angle
	
/*	//x-axis rotation by theta degree
	for (int i = 0; i < (int) verts.size(); i++)
	{
		y = verts[i]->coords[1]*cos(theta) - verts[i]->coords[2]*sin(theta);
		z = verts[i]->coords[1]*sin(theta) + verts[i]->coords[2]*cos(theta);
		verts[i]->coords[1] = y;
		verts[i]->coords[2] = z;
	}//*/

/*	//z-axis rotation by theta degree
	for (int i = 0; i < (int) verts.size(); i++)
	{
		float x = verts[i]->coords[0]*cos(theta) - verts[i]->coords[1]*sin(theta);
		float y = verts[i]->coords[0]*sin(theta) + verts[i]->coords[1]*cos(theta);
		verts[i]->coords[0] = x;
		verts[i]->coords[1] = y;
	}//*/
	
/*	//y-axis rotation by theta degree
	theta = -90.0f * (PI / 180.0f);
	for (int i = 0; i < (int) verts.size(); i++)
	{
		x = verts[i]->coords[2]*sin(theta) + verts[i]->coords[0]*cos(theta);
		z = verts[i]->coords[2]*cos(theta) - verts[i]->coords[0]*sin(theta);
		verts[i]->coords[0] = x;
		verts[i]->coords[2] = z;
	}//*/

	//edge lengths are preserved after rotations but bounding box info might change
	//computeBoundingBox();
}

void Mesh::addVertexND(float* c)
{
	//fastly add verts w/ coords c to mesh; ND: no duplication check

	int vSize = (int) verts.size(); //size before push_back just-below
	verts.push_back(new Vertex(vSize, c));
}

void Mesh::addTriangle(int v1i, int v2i, int v3i)
{
	//add tri v1i-v2i-v3i to the triInd'th element of tris

	int triInd = (int) tris.size();

	tris.push_back( new Triangle(triInd, v1i, v2i, v3i) );

	//update each vertex with its new tri and vert neighbors
	verts[v1i]->triNeighbors.push_back(triInd);
	//addVertNeighbor returns true if verts[v1i] is not neighbor of [v2i]; so they can form a new edge
	if (verts[v1i]->addVertNeighbor(v2i))	addEdge(v1i, v2i);
	if (verts[v1i]->addVertNeighbor(v3i))	addEdge(v1i, v3i);
	//in other words, a 'false' returned by addVertNeighbor prevents adding the same edge if it already exists

	verts[v2i]->triNeighbors.push_back(triInd);
	verts[v2i]->addVertNeighbor(v1i);	//v1i v2i edge is (probably) added above; do not add v2i v1i edge!!!
	if (verts[v2i]->addVertNeighbor(v3i))	addEdge(v2i, v3i);	//v3i v2i is not added above, so add this if ok

	verts[v3i]->triNeighbors.push_back(triInd);
	verts[v3i]->addVertNeighbor(v1i);	//no addEdge risk, so no if control
	verts[v3i]->addVertNeighbor(v2i);
}

int Mesh::getEdgeSharedBy(int vi, int vj)
{
	//return idx of edge[vi, vj]

	for (int vie = 0; vie < (int) verts[vi]->edgeList.size(); vie++)
		for (int vje = 0; vje < (int) verts[vj]->edgeList.size(); vje++)
			if (verts[vj]->edgeList[vje] == verts[vi]->edgeList[vie])
				return verts[vj]->edgeList[vje];

	//edge not exist
	return -99;
}

void Mesh::updateNormals()
{
	//update normals of tris and verts in mesh

	//triangles
	for (int t = 0; t < (int) tris.size(); t++)
	{
		int v1i = tris[t]->v1i, v2i = tris[t]->v2i, v3i = tris[t]->v3i;
		tris[t]->setNormal(verts[v1i]->coords, verts[v2i]->coords, verts[v3i]->coords);
		tris[t]->setArea(verts[v1i]->coords, verts[v2i]->coords, verts[v3i]->coords);
	}

	//vertices
	for (int v = 0; v < (int) verts.size(); v++)
	{
		float* normalTotal = new float[3];
		normalTotal[0] = normalTotal[1] = normalTotal[2] = 0.0f;
		int howManyTriNeighb = 0;
		for (int tneighb = 0; tneighb < (int) verts[v]->triNeighbors.size(); tneighb++)		
			if (tris[ verts[v]->triNeighbors[tneighb] ]->area >= 0.001f/*MIN_TRIANGLE_AREA_ALLOWED*/)	//ignore temporary 0-area tris which occur by i.e. split
			{
				for (int i = 0; i < 3; i++)
					normalTotal[i] += tris[ verts[v]->triNeighbors[tneighb] ]->normal[i]; //maby a better way is to weight tri-normals w/ their areas for v.normal computation
				howManyTriNeighb++;
			}
		if (howManyTriNeighb != 0) //if 0, go on w/ the ex-normal
		{
			for (int i = 0; i < 3; i++)
				normalTotal[i] /= (float) howManyTriNeighb;
			verts[v]->setNormal(normalTotal);
		}
	}
}

inline float distanceBetween(float* v1, float* v2)
{
	//In fact, the keyword inline is not necessary. If the function is defined with its body directly and the function
	//has a smaller block of code, it will be automatically treated as inline by the compiler
	return (float) sqrt( (v2[0]-v1[0])*(v2[0]-v1[0]) + (v2[1]-v1[1])*(v2[1]-v1[1]) + (v2[2]-v1[2])*(v2[2]-v1[2]));
}
inline float distanceBetween2(float* v1, float* v2)
{
	//squared distance which skips sqrt part of distanceBetween() for efficiency; e.g. use this for comparison-based distances
	return (float) (v2[0]-v1[0])*(v2[0]-v1[0]) +  (v2[1]-v1[1])*(v2[1]-v1[1]) + (v2[2]-v1[2])*(v2[2]-v1[2]);
}

inline float Mesh::calcTriArea(int t)
{
	float* v1 = verts[ tris[t]->v1i ]->coords,
		 * v2 = verts[ tris[t]->v2i ]->coords,
		 * v3 = verts[ tris[t]->v3i ]->coords;

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

	//When it comes time to de-allocate them, I had them all in a single delete[] statement: delete[] a, b, c, d, e; this appears to only be de-allocating a and completely ignoring b, c, d, and e
	delete p_r;
	delete q_r;
	delete cross; //arrays allocated with new[] must be deallocated with delete[] so i should've added [] here but not a big deal

	return 0.5f * length;
}

void Mesh::addEdge(int v1i, int v2i)
{
	int initial = (int) edges.size();
	Edge* newcomer = new Edge(initial, v1i, v2i, distanceBetween(verts[v1i]->coords, verts[v2i]->coords));

	//update the list of edges of which verts[v1i] is a member
	verts[v1i]->edgeList.push_back(newcomer->idx);
	verts[v2i]->edgeList.push_back(newcomer->idx); //do same thing for [v2i]

	edges.push_back(newcomer);

	//information
	if (newcomer->length < minEdgeLen)
		minEdgeLen = newcomer->length;
	if (newcomer->length > maxEdgeLen)
	{
		maxEdgeLen = newcomer->length;
//cout << v1i << " " << v2i << " ml\n";
	}
	edgeLenTotal += newcomer->length;
}

int* Mesh::getTrianglesSharedBy(Edge* bigEdge, int v1i, int v2i)
{
	//tri t is shared by bigEdge iff it exists in the triNeigbors of verts[v1i] & [v2i]
	if (v1i == -1) //learn v1i and v2i from bigEdge
	{
		v1i = bigEdge->v1i;
		v2i = bigEdge->v2i;
	}
	int a = 0, tr1, tr2;
	int* shTris = new int[2];	//portable version (memo-friendly as well 'cos i delete shrdTris in split/flip())
	//seeing this took my 6 hours!!! prevent removed triangles from being selected as shared!
	for (int t1 = 0; t1 < (int) verts[v1i]->triNeighbors.size(); t1++)
	{
		tr1 = verts[v1i]->triNeighbors[t1];
		for (int t2 = 0; t2 < (int) verts[v2i]->triNeighbors.size(); t2++)
		{
			tr2 = verts[v2i]->triNeighbors[t2];
			if  (tris[tr2] != 0 && (tr1 == tr2))
				shTris[a++] = tr1; //or tr2
		}
	}
	//if called from split/flip, a is certainly incremented by 1; but it remains at 1 (no more ++) 
	//if bigEdge is a border edge (not in my case 'cos my mesh is closed); take precautions anyway;
	if (a == 1) shTris[1] = -1;	//number -1 must not change 'cos used by somewhere else
if (shTris[1] == -1) cout << v1i << "," << v2i << " is a border edge found in my closed-mesh :(\n\n"; //exit(0);}

//if called from contract, a may remain 0, which tells me to remove that edgesAdded[cea] neighbor to 2 removed tris
if (a == 0)	shTris[0] = -1;	//since that part of contract() is inactive, don't care here

	return shTris;
}

inline float maxOf(float a, float b) { return (a > b ? a : b); }
inline float minOf(float a, float b) { return (a < b ? a : b); }

void Mesh::updateEdgeLengths()
{
	//called after coords scaling which changes edge lengths change

	edgeLenTotal = 0.0f;
	minEdgeLen = INF;
	maxEdgeLen = -INF;
	for (int e = 0; e < (int) edges.size(); e++)
	{
		edges[e]->length = distanceBetween(verts[edges[e]->v1i]->coords, verts[edges[e]->v2i]->coords);
		if (edges[e]->length < minEdgeLen)
			minEdgeLen = edges[e]->length;
		if (edges[e]->length > maxEdgeLen)
			maxEdgeLen = edges[e]->length;
		edgeLenTotal += edges[e]->length;
	}
	avgEdgeLen = edgeLenTotal/((int) edges.size());
//	cout << "after updateEdgeLengths(): min <= avg <= maxEdgeLen: " << minEdgeLen << " <= " << avgEdgeLen << " <= " << maxEdgeLen << "\n";
}

bool Mesh::dijkstraShortestPaths(int sourceVert, float close)
{
	//computes shortest paths from sourceVert to all other verts that are at distance far or less

	if (verts[sourceVert]->dSpanningUnn) //don't recompute if already computed
		return true;
	verts[sourceVert]->dSpanning = new float[ (int) verts.size() ]; //in [0,1]
	verts[sourceVert]->dSpanningUnn = new float[ (int) verts.size() ]; //in original scale

	//use Fibonacci heap to get O(VlgV + e); minHeap takes O(VlgV + elgV), simple array takes O(V^2)
	FibHeap* h = new FibHeap();

	/////// original Dijkstra to compute shortest paths from sourceVert to all others ///////
	//initialization
	for (int v = 0; v < (int) verts.size(); v++)
	{
		verts[v]->dSpanned = (v == sourceVert ? 0.0f : INF);
		FibHeapNode* hn = new FibHeapNode(verts[v]->dSpanned, v); //v is a handle from headNode to verts,
		verts[v]->heapNode = hn;								  //and hn is an handle from verts to heapNode
		h->Insert(hn);
		//no delete hn; here 'cos it'll be extracted, used, and deleted below
	}

	while (true)
	{
		//extract vert w/ min dSpanned value
//not doing the brute force O(V) algo to extract min (useful for dense graph, i.e. simple array usage, which is not my case)
		FibHeapNode* minNode = h->ExtractMin(); //fast O(lgV) priority queue algo
		if (minNode == NULL)
			break; //no more verts w/ unknown shortest dist; so, we're done
		int minDidx = minNode->element;
		float minD = minNode->key;
		verts[sourceVert]->dSpanningUnn[minDidx] = minD; //remember original value before normalization in the end of this function
		delete minNode; //new hn above is deleted here to save memo [info: delete minNode; automatically calls minNode->~FibHeapNode(), i.e. its destructor]

		//minD is an ascending value; so, when i reach far enough verts, i can safely skip the rest since all verts visited so far have finalized their shortest paths,
		//which means any vertex that i visit after this pnt will be have dist >= minD to sourceVert
		//if (minD > far)  //far enough vert reached
		//	break;

		//relax each edge incident to minDidx
		for (int ve = 0; ve < (int) verts[minDidx]->edgeList.size(); ve++)
		{
			int e = verts[minDidx]->edgeList[ve];
			int va = (edges[e]->v1i == minDidx ? edges[e]->v2i : edges[e]->v1i);
			if (verts[minDidx]->dSpanned + edges[e]->length < verts[va]->dSpanned)
			{
				verts[va]->dSpanned = verts[minDidx]->dSpanned + edges[e]->length; //relaxation
				h->DecreaseKey(verts[va]->heapNode, verts[va]->dSpanned); //worst O(lgV), amortized O(1)
			}
		}
	} //end of while (true)
	/////// original Dijkstra to compute shortest paths from sourceVert to all others ///////

	delete h; //not enough 'cos i got nodes to delete [verified that disabling the while just-below will prevent memory deallocation and memory'll not be cleaned; so i must use this loop to actually clean the memory]
			  //no!!!!!! since all nodes are extracted and hence deleted above (no shortcut) i do not need just-below; just delete the h; this is not the case in dijkstraShortestPathsShortcut()
/*	while (true)
	{
		//ExtractNodeForDeletion() just returns in O(1) the min (root) to delete it; not maintaining the fib-heap property; use for memo dealloc only
		FibHeapNode* minNode = h->ExtractNodeForDeletion();
		if (minNode)
			delete minNode;
		else
			break;
	}
	delete h;*/

	if (maxGeoDist == -INF) //don't compute bad dSpanning[] w/ unhealthy min/maxGeoDist values in vain
		return false;
	//normalized dSpanning[] from dSpanningUnn[]
	//float denom = maxGeoDist - minGeoDist; //use the global min/maxGeoDist that spans the whole surface
//int nProcessed = 0;
	for (int i = 0; i < (int) verts.size(); i++)
	{
		//verts[sourceVert]->dSpanning[ i ] = (verts[sourceVert]->dSpanningUnn[i] - minGeoDist) / denom; //don't forget to change minGeoDist to minGeoDistG if you change denom
		verts[sourceVert]->dSpanning[ i ] = verts[sourceVert]->dSpanningUnn[i] / maxGeoDist; //minGeoDist=0 for sure so save some micro time by ignoring some operations
		if (verts[sourceVert]->dSpanningUnn[ i ] <= close) //close to this vertex so mark as unavailable (processed=true) to preserve evenly-spaced sampling in adaptiveSampling()
		{
			verts[i]->processed = true;
			verts[sourceVert]->patch.push_back(i); //to make markNeighborhood() more efficient, traverse through this small patch[] only (instead of a full vert[] traversal)
//nProcessed++;
		}
	}
//if (nProcessed) cout << nProcessed << " put in patch of " << sourceVert << endl;
	return false;
}
bool Mesh::dijkstraShortestPathsShortcut(int sourceVert, float close, int generatorVert)
{
	//same as dijkstraShortestPaths() except this breaks early after reaching a distance larger than close; and sets sourceVert.dSpannning[f] to a big number for nonclose vertices

	if (verts[sourceVert]->dSpanningUnn) //don't recompute if already computed
	{
		//markNeighborhood(sourceVert, close, true); copy markNeighborhood() here to avoid call overhead (compiler quite probably does this inlineing but anyway)
		for (int i = 0; i < (int) verts[sourceVert]->patch.size(); i++)
			verts[ verts[sourceVert]->patch[i] ]->processed = true;

		return true;
	}
	verts[sourceVert]->dSpanningUnn = new float[ (int) verts.size() ]; //in original scale

//problem [solved]:::::::: dSpanningUnn[w] will remain uninited for some distant w; dont want to use a verts[] traversal to init all of them 'cos that defeats the purpose;
//those unpredictable values may be detected in radius calculation (quite probably too large/small values are them); or just use dijkstraShortestPathsBFS2() in radius calcualtion (approx geodesic w/o dSpanningUnn[])
	for (int v = 0; v < (int) verts.size(); v++) //apparently this does not affect the timing (i guess new[verts.size()] above already takes linear time so this traversal has no extra cost; see info: in dijkstraShortestPathsBFS())
	{
		verts[sourceVert]->dSpanningUnn[v] = verts[v]->dSpanned = INF;
		verts[v]->heapNode = NULL;
	}

	//use Fibonacci heap to get O(VlgV + e); minHeap takes O(VlgV + elgV), simple array takes O(V^2)
	FibHeap* h = new FibHeap();
	//vertices farther away from generatorVert than X*close are useless for radius purposes (X>1 'cos sourceVert may be on the boundary of generatorVert.patch; so i should go 2*close from generatorVert)
//	float close2 = 2.0f*close;
//cout << sourceVert << " " << (int) verts[generatorVert]->patch2.size() << " sssss" << endl;
	for (int i = 0; i < (int) verts[generatorVert]->patch2.size(); i++) //twice larger than generatorVert.patch[]
//shortcut below (if(minD>close)break;) not gain anything if i insert all v vertices (O(vlgv + e) becomes O(vlgv)); so insert a subset of verts that are close to the generator, i.e., the ones in patch2[]
	{
		int v = verts[generatorVert]->patch2[i];
		verts[v]->dSpanned = (v == sourceVert ? 0.0f : INF);

		FibHeapNode* hn = new FibHeapNode(verts[v]->dSpanned, v); //v is a handle from headNode to verts,
		verts[v]->heapNode = hn;								  //and hn is an handle from verts to heapNode
		h->Insert(hn);
		//no delete hn; here 'cos it'll be extracted, used, and deleted below
	}
	while (true)
	{
		//extract vert w/ min dSpanned value
//not doing the brute force O(V) algo to extract min (useful for dense graph, i.e. simple array usage, which is not my case)
		FibHeapNode* minNode = h->ExtractMin(); //fast O(lgV) priority queue algo
		if (minNode == NULL)
		{
//cout << "shortcut never here\n"; exit(0); this may well occur since i disabled shortcutting below
			break; //no more verts w/ unknown shortest dist; so, we're done
		}
		int minDidx = minNode->element;
		float minD = minNode->key;
		verts[sourceVert]->dSpanningUnn[minDidx] = minD; //remember original value before normalization in the end of this function
		if (minD <= close) //now that shortcutting is disabled, sourceVert.patch would have been twice larger (as all patch2[] in the heap) w/o this if condition
		{
			verts[minDidx]->processed = true;
			verts[sourceVert]->patch.push_back(minDidx); //to make markNeighborhood() more efficient, traverse through this small patch[] only (instead of a full vert[] traversal)
		}
		delete minNode; //new hn above is deleted here to save memo [info: delete minNode; automatically calls minNode->~FibHeapNode(), i.e. its destructor]

/*		//minD is an ascending value; so, when i reach far enough verts, i can safely skip the rest since all verts visited so far have finalized their shortest paths, which means any vertex that i visit after this pnt will be have geodesic distance >= minD to sourceVert
		if (minD > close)//close2)  //far enough vert reached; close should work here but i prefer close2 to be safe (ensure dSpanningUnn[] filled for all potential samples[] around; no! patch'll be unnecesarrily big with close2)
		{
//cout << "shortcutting w/ " << minD << endl;
			break;
		} ensure dSpanningUnn[] filled for radius computation in GA by not shortcutting (heap already has subset of verts[], namely patch2[], so shortcutting not so crucial)*/

		bool earlyBreak = false; //if heap is too small then va below may not in the heap and verts[va]->heapNode would be NULL; early break to prevent crashing
		//relax each edge incident to minDidx
		for (int ve = 0; ve < (int) verts[minDidx]->edgeList.size(); ve++)
		{
			int e = verts[minDidx]->edgeList[ve];
			int va = (edges[e]->v1i == minDidx ? edges[e]->v2i : edges[e]->v1i);
			if (verts[minDidx]->dSpanned + edges[e]->length < verts[va]->dSpanned)
			{
				if (! verts[va]->heapNode)
				{
//cout << "null should not; increase X*close above\n";
					earlyBreak = true;
					break;
				}
				verts[va]->dSpanned = verts[minDidx]->dSpanned + edges[e]->length; //relaxation
				h->DecreaseKey(verts[va]->heapNode, verts[va]->dSpanned); //worst O(lgV), amortized O(1)
			}
		}
		if (earlyBreak)
			break;
	} //end of while (true)
/*	//delete h; not enough 'cos i got nodes to delete [verified that disabling the while just-below will prevent memory deallocation and memory'll not be cleaned; so i must use this loop to actually clean the memory]
	while (true)
	{
		//ExtractNodeForDeletion() just returns in O(1) the min (root) to delete it; not maintaining the fib-heap property; use for memo dealloc only
		FibHeapNode* minNode = h->ExtractNodeForDeletion();
		if (minNode)
		{
			//verts[sourceVert]->dSpanningUnn[ minNode->element ] = INF; //redundant 'cos i don't extract all verts[] here; some verts[] not in heap will still have undefined dSpanningUnn[] values so setting this to INF has no help
			delete minNode;
		}
		else
			break;
	}//i saw that block aboe is really useful (when shortcutting enabled) for memo recapture; but now (maybe since i dont insert all verts) it sometimes causes crushes so i disable it; # of insertions are small anyway so i should survive with this disabled*/
	delete h;
	return false;
}
void Mesh::dijkstraShortestPathsEuc(int sourceVert, float close, int generatorVert)
{
	//assigns euclidean distance (O(1)) instead of geodesic (O(vlogv)) to each sourceVert.dSpanningUnn[] entry that are close to generatorVert

	if (verts[sourceVert]->dSpanningUnn) //don't recompute if already computed
	{
		//markNeighborhood(sourceVert, close, true); copy markNeighborhood() here to avoid call overhead (compiler quite probably does this inlineing but anyway)
		for (int i = 0; i < (int) verts[sourceVert]->patch.size(); i++)
			verts[ verts[sourceVert]->patch[i] ]->processed = true;

		return;
	}
	verts[sourceVert]->dSpanningUnn = new float[ (int) verts.size() ]; //in original scale
	for (int v = 0; v < (int) verts.size(); v++) //apparently this does not affect the timing (i guess new[verts.size()] above already takes linear time so this traversal has no extra cost; see info: in dijkstraShortestPathsBFS())
		verts[sourceVert]->dSpanningUnn[v] = INF;

/*	//linear code (assuming dSpanning/Unn[] ready, which is the case for adaptiveSampling2()) [?????????? secs to complete 1 getFittestMember2()]
	for (int v = 0; v < (int) verts.size(); v++)
		//could have been even more efficient w/ squared dist (no sqrt) but then accuracy problems; dSpanningUnn[] are already filled by geodesics (for evenly-spaced samples (ns in evalSolution2()); they are comparable w/ actual distances not squared distances
		verts[sourceVert]->dSpanningUnn[v] = distanceBetween(verts[sourceVert]->coords, verts[v]->coords);//*/
	//efficient version that traverses a small-size generatorVert.patch2[] only instead of the verts[]; generally patch2.size = 900. verts.size=12500 [?????????? secs to complete 1 getFittestMember2()]
	for (int i = 0; i < (int) verts[generatorVert]->patch2.size(); i++)
	{
		verts[sourceVert]->dSpanningUnn[ verts[generatorVert]->patch2[i] ] = distanceBetween(verts[sourceVert]->coords, verts[ verts[generatorVert]->patch2[i] ]->coords);//*/
		verts[ verts[generatorVert]->patch2[i] ]->processed = true;
		verts[sourceVert]->patch.push_back(verts[generatorVert]->patch2[i]);
	}
}

void Mesh::dijkstraShortestPathsBFS(int sourceVert, float close)
{
	//BreadthFirstSearch from source (normally BFS continues until all nodes visited, but here i quickly return once a nonclose vert is visited)

	//verts[sourceVert]->distToClosestSampleDetermined = false; //this will be true if newInSamples2 is encountered below [not in use]
	if (verts[sourceVert]->dSpanningUnn) //don't recompute if already computed; but make the precomputed patch vertices unavailable (processed=true) before returning
	{
//if (verts[sourceVert]->patch.empty()){cout<<"cant be empty here\n";exit(0);}
		//markNeighborhood(sourceVert, close, true); copy markNeighborhood() here to avoid call overhead (compiler quite probably does this inlineing but anyway)
		for (int i = 0; i < (int) verts[sourceVert]->patch.size(); i++)
			verts[ verts[sourceVert]->patch[i] ]->processed = true;
		
		return;
	}
	verts[sourceVert]->dSpanningUnn = new float[ (int) verts.size() ]; //in original scale
//info: at first the cost of new[] is very low, as it just has to increment a pointer into a free memory block; but after a few thousand news and deletes, free memory becomes a complex set of variable-sized free blocks, and then it requires a fancy algo to select best of free blocks w/o taking much time (firstfit, bestfit); see dijkstraShortestPathsShortcut().top
//info: new float[verts.size()]() initializes every entry to 0; w/o () any initialization may happen; int array[100]={0}; tells the compiler to set these 100 ints to 0, which the compiler can optimize freely; for (i = 0; i < 100; ++i)array[i] = 0; forces the compiler work harder; nonzero init by std::fill(array, array+100, 42); in O(n) time

	for (int v = 0; v < (int) verts.size(); v++) //apparently this does not affect the timing (i guess new[verts.size()] above already takes linear time so this traversal has no extra cost; see info: in dijkstraShortestPathsBFS())
	{
		verts[sourceVert]->dSpanningUnn[v] = INF;
//		verts[v]->dBFS = INF; i must do dBFS = INF in reachedNonclose anyway (o/w i'd require verts traversal in dijkstraShortestPathsBFS2() which slows down execution by half; if you disable dijkstraShortestPathsBFS2() later then i can disable dBFS = INF in reachedNonclose (and enable this line)
	}

	//for (int v = 0; v < (int) verts.size(); v++) verts[v]->dBFS = INF; //clear effect(s) of previous call(s)
	//i want to avoid verts traversal just-above for efficiency; so remember the visited verts in visited to undo their dBFS values at the end of this function (so that i start here w/ dBFS=INF in upcoming iters)
	//vector< int > visited; instead of this local variable, use sourceVert.patch to remember the visited verts below; i wont need to do a BFS later (to set processed=true) in case this sourceVert calls here again (detected by dSpanningUnn!=NULL)
//if (! verts[sourceVert]->patch.empty()){cout<<"cant be nonempty here\n";exit(0);}

	queue< int > q;
	q.push(sourceVert);
	verts[sourceVert]->dBFS = 0.0f; //length of unweighted shortest path from source to source is 0
	verts[sourceVert]->patch.push_back(sourceVert); //visited.push_back(sourceVert);
	verts[sourceVert]->dSpanningUnn[sourceVert] = 0.0f;
	verts[sourceVert]->processed = true;
//	bool distToClosestSampleOK = false; //for radius computation in GA.evalSolution2(), make sure that sourceVert.dSpanningUnn[x] is determined where x is a sample already in evolving samples2; wrong!!!! 'cos 1st new sample will never find
//										//another one as there is only itself in (finalized part of) samples2; similarly, when there are few samples in (finalized part of) samples2, i need to extend the BFS frontier a lot to reach the first close sample, which ruins the patch knowledge
	while (! q.empty())
	{
		int v = q.front();
		q.pop();

		//look at each adjacent vertices of v
		bool reachedNonclose = false; //true if i reach a distant vertex, in which case i early-stop BFS
		set< int, std::less< int > >::const_iterator neighbVertIdx;
		for (neighbVertIdx = verts[v]->vertNeighborsSet.begin(); neighbVertIdx != verts[v]->vertNeighborsSet.end(); neighbVertIdx++)
		{
			int va = (*neighbVertIdx);
			if (verts[va]->dBFS == INF)
			{
				verts[va]->dBFS = verts[v]->dBFS + 1.0f; //update d of this adjacent
				q.push(va); //va.parent = v; could also be done here but not using parents in my applicaiton
				verts[sourceVert]->patch.push_back(va); //visited.push_back(va); //all verts whose dBFS updated are stored
				float dist = //distanceBetween(verts[sourceVert]->coords, verts[va]->coords); //just-below is a better approximation
							 verts[va]->dBFS * avgEdgeLen; //BFS finds the shortest path between two nodes sourceVert and va, with path length measured by number of edges; so multiply it w/ avgEdgeLen; assuming uniform edge lengths on the mesh
				verts[sourceVert]->dSpanningUnn[va] = dist;
				verts[va]->processed = true;
				//if (verts[va]->newInSamples2) verts[sourceVert]->distToClosestSampleDetermined = true; //not in use ('cos if-condition should never be true and hence this is useless)
				if (dist > close)
					reachedNonclose = true;
			}
		}
		if (reachedNonclose)
		{
			//undo dBFS values and return; this is not done in classical BFS but in my version i cancelled verts[v]->dBFS = INF; above so this is required
			for (int u = 0; u < (int) verts[sourceVert]->patch.size(); u++)
				verts[ verts[sourceVert]->patch[u] ]->dBFS = INF; //no need 'cos i reset dBFS = INF via verts traversal above

			return;
		}
	}
//	return -1; //all verts visited the key i am looking for is not found
}
float Mesh::dijkstraShortestPathsBFS2(int sourceVert, int & closestSample)
{
	//same as dijkstraShortestPathsBFS() except this one return the (approximate) geodesic distance to the first sample that is encountered

	//for (int v = 0; v < (int) verts.size(); v++) verts[v]->dBFS = INF; //clear effect(s) of previous call(s)
	//i want to avoid verts traversal just-above for efficiency; so remember the visited verts in visited to undo their dBFS values at the end of this function (so that i start here w/ dBFS=INF in upcoming iters)
	vector< int > visited; //verified that enabling vert traversal just-above slows down execution by half; so visited[] bookkeeping is mandatory

	queue< int > q;
	q.push(sourceVert);
	verts[sourceVert]->dBFS = 0.0f; //length of unweighted shortest path from source to source is 0
	visited.push_back(sourceVert);
	while (! q.empty())
	{
		int v = q.front();
		q.pop();

		//look at each adjacent vertices of v
		set< int, std::less< int > >::const_iterator neighbVertIdx;
		for (neighbVertIdx = verts[v]->vertNeighborsSet.begin(); neighbVertIdx != verts[v]->vertNeighborsSet.end(); neighbVertIdx++)
		{
			int va = (*neighbVertIdx);
			if (verts[va]->dBFS == INF)
			{
				verts[va]->dBFS = verts[v]->dBFS + 1.0f; //update d of this adjacent
				visited.push_back(va); //all verts whose dBFS updated are stored
				if (verts[va]->sample)
				{
					float dist = //distanceBetween(verts[sourceVert]->coords, verts[va]->coords); //just-below is a better approximation
								 verts[va]->dBFS * avgEdgeLen; //BFS finds the shortest path between two nodes sourceVert and va, with path length measured by number of edges; so multiply it w/ avgEdgeLen; assuming uniform edge lengths on the mesh
//if(sourceVert==12425)cout << sourceVert << " " << va << " has " << distanceBetween(verts[sourceVert]->coords, verts[va]->coords) << " ==? " << dist << endl;
//if(v==12425)cout << v << " " << va << " has " << distanceBetween(verts[v]->coords, verts[va]->coords) << endl;
//smatch=va;
					//undo dBFS values and return; this is not done in classical BFS but in my version i cancelled verts[v]->dBFS = INF; above so this is required
					for (int u = 0; u < (int) visited.size(); u++) verts[ visited[u] ]->dBFS = INF;
					closestSample = va;
					return dist;
				}
				q.push(va); //va.parent = v; could also be done here but not using parents in my applicaiton
			}
		}
	}
	cout << "never here dijksbfs2\n";
	exit(0);
//return -1.0f; //all verts visited the key i am looking for is not found
}

void Mesh::markNeighborhood(int sourceVert, float close, bool processedValue)
{
	//all verts that are close to sourceVert set their processed value to processedValue

/*	//linear code (assuming dSpanning/Unn[] ready, which is the case for adaptiveSampling2()) [6.5 secs to complete 1 getFittestMember2()]
	for (int v = 0; v < (int) verts.size(); v++)
		if (verts[sourceVert]->dSpanningUnn[v] <= close)
			verts[v]->processed = processedValue;//*/
	//efficient version that traverses a small-size sourceVert.patch[] only instead of the verts[]; generally patch.size = 300. verts.size=12500 [3.5 secs to complete 1 getFittestMember2()]
	for (int i = 0; i < (int) verts[sourceVert]->patch.size(); i++)
	{
		verts[ verts[sourceVert]->patch[i] ]->processed = processedValue;
//if (verts[sourceVert]->dSpanningUnn[ verts[sourceVert]->patch[i] ] > close) {cout << "too distant " << verts[sourceVert]->dSpanningUnn[ verts[sourceVert]->patch[i] ] << " > " << close; exit(0);}
	}//*/
}

inline float angleBetween(float* v1, float* v2)
{
	//returns smallest radian angle b/w normalized v1 & v2 via v1 DOT v2 = v1Len * v2Len * cosTheta (len = 1 for normalized, hence acos(dot) is the theta)

	float cosTheta = 0; //v1.dot(v2);
	for (int i = 0; i < 3; i++)
		cosTheta += v1[i]*v2[i];
	if (cosTheta < -1.0f)	cosTheta = -1.0f;
	if (cosTheta > 1.0f)	cosTheta = 1.0f;
	return acos(cosTheta); //* (180/PI); no radian to float to save time (besides computeGaussianCurvature() expects radian output)
}

float Mesh::getIsometricDistortion(Mesh* mesh2, vector< int > tmpSafeCorrespTry)
{
	//returns D_iso = distortion, i.e. subtraction of normalized geodesics, of the correspondence lying in the tmpSafeCorrespTry list using tmpSafeCorrespTry itself as the traversalList

	float isometryCostTry = 0.0f;//, acceptableIso = isometricDistortion; note that isometricDistortion is not up-to-date during GA tournaments; fix this if you want to use it here [tried using it but no gain at all so notInUse]
//minIsoDisto = INF; //necessary only for getRankDistortion() [notInUse]
	int n = (int) tmpSafeCorrespTry.size() / 2;
	for (int s = 0; s < n; s++)
	{
		int bvs = tmpSafeCorrespTry[2*s], bvb = tmpSafeCorrespTry[2*s + 1]; //tmpSafeCorrespTry[even idxs] ~ this and [odd] ~ mesh2
		float vertexBasedIso = 0.0f;
		for (int ss = 0; ss < n; ss++) //traversal
		{
			int s1 = tmpSafeCorrespTry[2*ss], s2 = tmpSafeCorrespTry[2*ss + 1];
			if (s1 == bvs) //when bvs-bvb is a bad match, s1-s2 will wrongly award it by += 0 effect when s1=bvs (and hence s2=bvb); to prevent this, skip s1=bvs case; if bvs-bvb is a good match then this += 0 has no harm
				continue;

			//if bvs vs. bvb is a good correspondence, then geoDist(bvs, s1) = geoDist(bvb, s2) for all s1 & s2 in tmpSafeCorrespTry
			float t1s1 = verts[bvs]->dSpanning[s1], t2s2 = mesh2->verts[bvb]->dSpanning[s2];
			vertexBasedIso += fabs(t1s1 - t2s2); //normalized difference here is small if tbv1 vs. tbv2 is a good correspondence
		}
		vertexBasedIso /= (n-1); //in computation of vertexBasedIso of bvs-bvb, 1 match, bvs-bvb itself, is skipped during traversal, and therefore n-1 += performed
//		if (n != 1) //no div by zero [cannot happen in this project so disabling to save micro time]
//			vertexBasedIso /= (n-1); //in computation of vertexBasedIso of bvs-bvb, 1 match, bvs-bvb itself, is skipped during traversal, and therefore n-1 += performed
//		else
//			vertexBasedIso = 0.0f; //due to symmetry problem, all costs will be the same and hence best selected arbitrarily
		isometryCostTry += vertexBasedIso;
		verts[bvs]->diso = mesh2->verts[bvb]->diso = vertexBasedIso; //to save time i could have avoided division on vertexBasedIso above (since i'll use diso values only for ranking so division is an extra operation)
//if (vertexBasedIso < minIsoDisto) minIsoDisto = vertexBasedIso;
		/*if (s == 0 && vertexBasedIso > acceptableIso) //check the first 7 (currently just 1) items (extremities in FPS) and if either of them has an unacceptable/big iso then don't bother to go over the whole map; just quit with a bad overall distortion
		{
//this shortcut trick is ~useful for robustListInUse=false only; 100 samples, 100 tournament take 46 secs without this trick, 30 secs with this trick; still slow to use mapItSelf as traversal list)
//			cout << acceptableIso << " " << vertexBasedIso << "\n";
			return vertexBasedIso * 10.0f; //bad overall distortion is returned which causes this map/chromosome to be depromoted in genetic algorithm
		}//*/
	} //end of s
	//overall isometry cost of this tmpSafeCorrespTry combination
	return (isometryCostTry / n);
}
float Mesh::getIsometricDistortionRL(Mesh* mesh2, vector< int > tmpSafeCorrespTry)
{
	//same as getIsometricDistortion() except this uses robustList as the traversalList

	float isometryCostTry = 0.0f;
//minIsoDisto = INF; //necessary only for getRankDistortion() [notInUse]
	int n = (int) tmpSafeCorrespTry.size() / 2, m = (int) robustList.size() / 2;
	for (int s = 0; s < n; s++)
	{
		int nAdds = 0, bvs = tmpSafeCorrespTry[2*s], bvb = tmpSafeCorrespTry[2*s + 1]; //tmpSafeCorrespTry[even idxs] ~ this and [odd] ~ mesh2
		float vertexBasedIso = 0.0f;
		for (int ss = 0; ss < m; ss++) //traversal
		{
			int s1 = robustList[2*ss], s2 = robustList[2*ss + 1];
			if (s1 == bvs) //when bvs-bvb is a bad match, s1-s2 will wrongly award it by += 0 effect when s1=bvs (and hence s2=bvb); to prevent this, skip s1=bvs case; if bvs-bvb is a good match then this += 0 has no harm
				continue;

			//if bvs vs. bvb is a good correspondence, then geoDist(bvs, s1) = geoDist(bvb, s2) for all s1 & s2 in tmpSafeCorrespTry
			//float t1s1 = verts[bvs]->dSpanning[s1], t2s2 = mesh2->verts[bvb]->dSpanning[s2]; dSpanning[] defined for s1 & s2 only since m << n in this mode; this gives the same t1s1 and t2s2 as just-below
			float t1s1 = verts[s1]->dSpanning[bvs], t2s2 = mesh2->verts[s2]->dSpanning[bvb];
			vertexBasedIso += fabs(t1s1 - t2s2); //normalized difference here is small if tbv1 vs. tbv2 is a good correspondence
			nAdds++;
		}
		//vertexBasedIso /= (m-1); //in computation of vertexBasedIso of bvs-bvb, 1 match, bvs-bvb itself, is skipped during traversal, and therefore m-1 += performed
		vertexBasedIso /= nAdds; //m-1 is rarely wrong (when the robustList is due to computeEvenlySpacedSamples() and bvs is due to computeSamples(), s1 == bvs will never hold; so -1 wrong in that time)

//		if (n != 1) //no div by zero [cannot happen in this project so disabling to save micro time]
//			vertexBasedIso /= (n-1); //in computation of vertexBasedIso of bvs-bvb, 1 match, bvs-bvb itself, is skipped during traversal, and therefore n-1 += performed
//		else
//			vertexBasedIso = 0.0f; //due to symmetry problem, all costs will be the same and hence best selected arbitrarily
		isometryCostTry += vertexBasedIso;
		verts[bvs]->diso = mesh2->verts[bvb]->diso = vertexBasedIso; //to save time i could have avoided division on vertexBasedIso above (since i'll use diso values only for ranking so division is an extra operation)
//if (vertexBasedIso < minIsoDisto) minIsoDisto = vertexBasedIso;
	} //end of s
//cout << isometryCostTry / n << "in\t\t";

	//overall isometry cost of this tmpSafeCorrespTry combination
	return (isometryCostTry / n);
}

float Mesh::getGroundDistortion(Mesh* mesh2, vector< int > tmpSafeCorrespTry, bool special)
{
	//returns D_grd = ground-truth distortion, based on the assumption that mesh1.verts[i] goes to mesh2.verts[i]; when robustList is in use, this gives Dgrd only for robustList (no!! see 1st comment below); o/w same as getGroundDistortionFull()

	float groundTruthCost = 0.0f;
	int n = (int) tmpSafeCorrespTry.size() / 2, nAdds = 0;
	for (int s = 0; s < n; s++)
	{
		if (verts[ tmpSafeCorrespTry[2*s] ]->dSpanningUnn) //when robustListInUse=true, dSpanningUnn is null for samples that are not in the robustList (no!! FPS computes dSpanningUnn; this dSpanningUnn=NULL comment is valid only if computeSamples() in use)
		{
			if (special && verts[ tmpSafeCorrespTry[2*s] ]->matchIdx == -1) //all outliers were removed from tmpSafeCorrespTry() in refillCorresp(); still i need special flag 'cos GA calls here while robusts.matchIdxs are still undecided
				cout << "WARNING: called at a special time, namely in the very end; at this time those with matchIdx=-1 may never exist in tmpSafeCorrespTry; something wrong!\n";
			if (tmpSafeCorrespTry[2*s + 1] >= (int) verts.size()) //may happen when mesh2.verts.size > mesh1.verts.size (i-i mapping assumption fails in this case so return -1)
				return -1.0f;
			groundTruthCost += verts[ tmpSafeCorrespTry[2*s] ]->dSpanning[ tmpSafeCorrespTry[2*s + 1] ]; //assuming i-i ground-truth mapping
			nAdds++;
verts[ tmpSafeCorrespTry[2*s] ]->dgrd = mesh2->verts[ tmpSafeCorrespTry[2*s + 1] ]->dgrd = verts[ tmpSafeCorrespTry[2*s] ]->dSpanning[ tmpSafeCorrespTry[2*s + 1] ];
//for mesh2.v i could use mesh2->verts[ tmpSafeCorrespTry[2*s + 1] ]->dSpanning[ tmpSafeCorrespTry[2*s] ] which is ~same as verts[ tmpSafeCorrespTry[2*s] ]->dSpanning[ tmpSafeCorrespTry[2*s + 1] ];
		}
	}
	if (nAdds == 0)
		cout << "weird getgrddistortion()" << groundTruthCost << endl;

	return (groundTruthCost / nAdds);
}
float Mesh::getGroundDistortionFull(Mesh* mesh2, vector< int > tmpSafeCorrespTry)
{
	//same as getGroundDistortion() except here even if dSpanning not ready yet, this one calls dijkstra to make it ready and consequently gives distortion covering the full shape, not just the robustList as in getGroundDistortion()

	float groundTruthCost = 0.0f;
	int n = (int) tmpSafeCorrespTry.size() / 2, nAdds = 0, tooMany = 1001;
	if (n >= 500)
		cout << "getGroundDistortionFull will take a while due to " << n-(int)robustList.size() << " new dijkstras\n";
	for (int s = 0; s < n; s++)
	{
		dijkstraShortestPaths(tmpSafeCorrespTry[2*s]); //returns immediately if dSpanning computed already; so no inefficiency for that
		if (tmpSafeCorrespTry[2*s + 1] >= (int) verts.size()) //may happen when mesh2.verts.size > mesh1.verts.size (i-i mapping assumption fails in this case so return -1)
			return -1.0f;
		groundTruthCost += verts[ tmpSafeCorrespTry[2*s] ]->dSpanning[ tmpSafeCorrespTry[2*s + 1] ]; //assuming i-i ground-truth mapping (maxGeoDist already set so good dSpanning values here)
		nAdds++;
verts[ tmpSafeCorrespTry[2*s] ]->dgrd = mesh2->verts[ tmpSafeCorrespTry[2*s + 1] ]->dgrd = verts[ tmpSafeCorrespTry[2*s] ]->dSpanning[ tmpSafeCorrespTry[2*s + 1] ];
//for mesh2.v i could use mesh2->verts[ tmpSafeCorrespTry[2*s + 1] ]->dSpanning[ tmpSafeCorrespTry[2*s] ] which is ~same as verts[ tmpSafeCorrespTry[2*s] ]->dSpanning[ tmpSafeCorrespTry[2*s + 1] ];
		if (nAdds == tooMany)
		{
			cout << "too many dijkstra computations are prevented by early-breaking; the result is not for the full mesh but for the first " << tooMany << " matches\n";
			break;
		}
	}
	return (groundTruthCost / nAdds);
}
float Mesh::getRankDistortion(Mesh* mesh2, vector< int > tmpSafeCorrespTry)
{
	//returns D_rank = distortion, i.e. minIso*rankDifference of each match in tmpSafeCorrespTry

return 0;

/*	float rankCost = 0.0f, alpha = minIsoDisto / 5.0f;
	int n = (int) tmpSafeCorrespTry.size() / 2;
	for (int s = 0; s < n; s++)
	{
		int bvs = tmpSafeCorrespTry[2*s], bvb = tmpSafeCorrespTry[2*s + 1]; //tmpSafeCorrespTry[even idxs] ~ this and [odd] ~ mesh2
		float vertexBasedCost = fabs((float) verts[bvs]->agdOrder - (float) mesh2->verts[bvb]->agdOrder);
		if (vertexBasedCost <= windowSize)
			vertexBasedCost = 0.0f; //no rank distortion for this vertex since its rank and its correspondent's rank are in the same window, i.e. compatible
		else
			vertexBasedCost *= alpha;

		rankCost += vertexBasedCost;
		verts[bvs]->drank = mesh2->verts[bvb]->drank = vertexBasedCost;
	} //end of s

	//overall rank cost of this tmpSafeCorrespTry combination
	return (rankCost / n);*/
}

void Mesh::computeEvenlySpacedSamples(int N, bool earlyBreak, int startVertex)
{
	//sampling of N evenly-spaced FPS samples

	if (! samples.empty())
		return; //already computed so do nothing

#ifdef VERBOSE
	cout << "Evenly-spaced sampling [" << N << "]......\n";
#endif
	//fixed number (N) of fps (farthest point sampling) samples goes to samples[] first of which is an extremity vertex or startVertex
	int extVert = (startVertex == -1 ? getAnExtremeVert() : startVertex); //converges to the same sampling no matter what extVert is; getAnExtremeVert() is still good 'cos i am matching the samples and i want the 1st ones to be compatible, e.g. extremities, not random startVertexes
	for (int v = 0; v < (int) verts.size(); v++)
		verts[v]->processed = false; //initialize this to prevent duplications in samples; true means added to samples
	samples.push_back(extVert);
	verts[extVert]->sample = true;
	dijkstraShortestPaths(extVert); //dSpanningUnn[] to be used below is now ready, dSpanning is bad due to unhealthy min/maxGeoDist values (to be corrected at the bottom)
	verts[extVert]->processed = true; //to prevent duplications in samples; true means added to samples
	float maxDist0 = -1.0f; //very first maxDist stored here to early-break
	//fill the remaining N-1 items via fps for uniform distribution that lets nice coverage on models
	//complexity of FPS below is O(NV + NVlogV), where V is for farthest point search, and VlogV is for dijkstra in the end that fills dSpanningUnn[] to be used in the next iter; one can reduce the first V to logV by efficient searching via kd-tree or sth, hence O(NlogV + NVlogV)
	//as a result, 24 secs for N=500 on a V=12.5K scape model, 0.4 sec for N=9, 4 secs for N=100, 85 secs for N=1000
	while ((int) samples.size() < N)
	{
		//in fps, the next sample candidate x_i \in all vertices finds its closest existing sample x_clo and remembers d_i = d(x_i, x_clo); if d_i is the max among all other candidates closest remembered dists, then x_i is selected as the next sample;
		//here my candidates are restricted to nothing, i.e. i use all verts as candidates; so, find the closest existing sample e2 \in samples to s \in verts and add s to samples as nextSample if that closest dist is very big, i.e. the biggest among all closest dists for different s's
		float maxDist = -INF;
		int nextSample = -1;				
		for (int s = 0; s < (int) verts.size(); s++) //fps selects from amongst all vertices
		{
			int e2 = -1;
			if (verts[s]->processed || verts[s]->edgeList.empty()) //isolated verts should never be samples (even if they'd have dSapnning=INF values); they occur on microholes/holes/viewing classes of shrec11 dataset; val=6 but disconnected verts still problem so not a good solution
				continue; //no chance to select s as a sample 'cos it's already added or is an isolated vert

			float minDist = verts[ samples[0] ]->dSpanningUnn[s]; //=INF is tricky 'cos objects may be scaled to those high coordinates			
			for (int e = 0; e < (int) samples.size(); e++) //find the closest existing base e \in samples to s \in verts
				if (verts[ samples[e] ]->dSpanningUnn[s] <= minDist)
				{
					minDist = verts[ samples[e] ]->dSpanningUnn[s];
					e2 = samples[e];
				}
			if (verts[e2]->dSpanningUnn[s] > maxDist)
			{
				maxDist = verts[e2]->dSpanningUnn[s];
				nextSample = s;
			}
		} //end of for s
		if (nextSample == -1) {cout << "can't happen in FPS\n"; exit(0);}
		maxDist0 = (maxDist0 == -1.0f ? maxDist : maxDist0);
		if (earlyBreak && maxDist < maxDist0/3.0f) //current nextSample is not sufficiently far away from the closest sample so break the FPS right now
			break;		
		samples.push_back(nextSample);
		verts[nextSample]->sample = true;
		verts[nextSample]->processed = true; //added, so not add again later; even if not added ('cos isolated) make processed=true to discard this in upcoming iters as a sample candidate
		dijkstraShortestPaths(nextSample); //upcoming iteration needs samples.dSpanningUnn which is not currently ready when selecting from the pool of all verts
#ifdef VERBOSE
			if (earlyBreak) cout << "sample # " << (int) samples.size() << " w/ distance " << maxDist << " to the closest sample\n";
#endif		
//if (earlyBreak && (int) samples.size() == 5) break;
	} //end of while N
//for (int v = 0; v < (int) verts.size(); v++) cout << verts[ 6 ]->dSpanningUnn[ v ] << "\t"; //if verts6 is disconnected this prints INF for all except v=6 (prints 0 to itself); easier: check valence edgeList.size

	//set min/maxGeoDist properly which in turn gives me a chance to set the dSpanning[] values properly
	minGeoDist = INF; //will always be 0
	maxGeoDist = -INF;
//int nWarns = 0;
	for (int i = 0; i < (int) samples.size(); i++)
		for (int v = 0; v < (int) verts.size(); v++)
		{
			if (verts[ samples[i] ]->dSpanningUnn[ v ] < minGeoDist)
				minGeoDist = verts[ samples[i] ]->dSpanningUnn[ v ];
			if (verts[ samples[i] ]->dSpanningUnn[ v ] > maxGeoDist)
				maxGeoDist = verts[ samples[i] ]->dSpanningUnn[ v ];
if (verts[ samples[i] ]->dSpanningUnn[ v ] == INF)
{
//	verts[ samples[i] ]->dSpanningUnn[ v ] = distanceBetween(verts[ samples[i] ]->coords, verts[v]->coords);
//	if (nWarns++ < 5)
////cout<<"WARNING: " << samples[i] << " " << v << " (valence: " << verts[ samples[i] ]->edgeList.size() << ") not connected by a path\n";//; using euclidean distance instead: " << verts[ samples[i] ]->dSpanningUnn[ v ] << endl;
//	else if (nWarns == 5) cout << "no more WARNING prints; there may be other warnings\n";
////exit(0);
}
		}
if (maxGeoDist == INF) //{cout << "disconnected vert(s) should be canceled\n";exit(0);} //guarantees that mesh is not disconnected (especially manually cut meshes can be bad); i may not crush if bases i'll use not hit these disconnected ones but be safe and stop here
	//maxGeoDist = 245; //meshes w/ disconnected verts, e.g. all microholes/holes/viewing meshes of shrec11 manually learn their maxGeoDist using a known maxGeoDistG from frame 0
	maxGeoDist = 3;//this is dataset specific; for weird data which still has maxGeoDist == INF at this point, i.e. it has a disconnect component, set the max distance by visual inspection
#ifdef VERBOSE
else cout << "max geodesic distance on this mesh: " << maxGeoDist << endl;
#endif
	//clean old samples.dSpanning values
	for (int i = 0; i < (int) samples.size(); i++)
	{
		if (verts[ samples[i] ]->dSpanning) //may already be deleted in for-loop above
			delete [] verts[ samples[i] ]->dSpanning;
		if (verts[ samples[i] ]->dSpanningUnn)
			delete [] verts[ samples[i] ]->dSpanningUnn; //arrays allocated with new[] must be deallocated with delete[]
		verts[ samples[i] ]->dSpanning = verts[ samples[i] ]->dSpanningUnn = NULL;

		//dSpanning[] good now 'cos min/maxGeoDist ready; dSpanning[]'ll be used during distortion computations of maps b/w shape samples
		dijkstraShortestPaths(samples[i]);
		
#ifdef VERBOSE
		if (i < 5 || i >= N-5) //print the first and last 5 sample ids
			cout << samples[i] << " ";
		else if (i == 5)
			cout << " .. ";
#endif
	}
#ifdef VERBOSE
	cout << "\n";
#endif
}
int Mesh::getAnExtremeVert()
{
	//returns idx of an extreme vertex, e.g. on hand or toe

	//vertex that is farthest away from arbitraryVert is definitely an extreme, where arbitraryVert can be anywhere on mesh (center, tips, head, ..)
	int arbitraryVert = 0, result = -1;
	while (verts[arbitraryVert]->edgeList.empty())
		arbitraryVert++; //start w/ a nonisolated arbitrary vertex
	dijkstraShortestPaths(arbitraryVert); //arbitraryVert.dSpanningUnn[] is now ready, dSpanning is bad due to unhealthy min/maxGeoDist values
	float maxDist = -INF;
	for (int s = 0; s < (int) verts.size(); s++) //fps selects from amongst all vertices
		if (verts[arbitraryVert]->dSpanningUnn[s] > maxDist && ! verts[s]->edgeList.empty()) //2nd condition ensures isolated verts are not selected as extremes
			{
				maxDist = verts[arbitraryVert]->dSpanningUnn[s];
				result = s;
			}
	//arbitraryVert.dSpanning[] undefined 'cos maxGeoDist unknown now; so clear dSpanning[] (and let potential future calls, if any, enter dijkstra by clearing dSpanningUnn[] too)
	delete [] verts[arbitraryVert]->dSpanning;
	delete [] verts[arbitraryVert]->dSpanningUnn;
	verts[arbitraryVert]->dSpanning = verts[arbitraryVert]->dSpanningUnn = NULL;
	return result;
}

void Mesh::computeSamples(int N)
{
	//sampling of N ~evenly-spaced samples; samples from fillRobustTraversalList() added directly and then Euclidean dists used to fill the rest (typically N >> 100 here)

#ifdef VERBOSE
	cout << "Sampling [" << N << "]......\n";
#endif
	if (N < (int) robustList.size() / 2)
	{
		cout << "too small N; make it bigger\n";
		exit(0);
	}
	if (robustList.empty())
	{
		cout << "robustList empty; fill it\n";
		exit(0);
	}
	
	//for the normalizer maxGeoDist get an extreme vertex
	int extVert = getAnExtremeVert();
	dijkstraShortestPaths(extVert); //dSpanningUnn[] to be used below is now ready, dSpanning is bad due to unhealthy min/maxGeoDist values (to be corrected just-below)
	maxGeoDist = -INF;
	minGeoDist = INF; //will always be 0
	for (int v = 0; v < (int) verts.size(); v++)
	{
		if (verts[ extVert ]->dSpanningUnn[ v ] < minGeoDist)
			minGeoDist = verts[ extVert ]->dSpanningUnn[ v ];
		if (verts[ extVert ]->dSpanningUnn[ v ] > maxGeoDist)
			maxGeoDist = verts[ extVert ]->dSpanningUnn[ v ];
if (verts[ extVert ]->dSpanningUnn[ v ] == INF){cout<<extVert<< " " << v << " not connected by a path\n"; exit(0);}
	}
if (maxGeoDist == INF) {cout << "disconnected vert(s) should be canceled\n";exit(0);} //guarantees that mesh is not disconnected (especially manually cut meshes can be bad); i may not crush if bases i'll use not hit these disconnected ones but be safe and stop here
#ifdef VERBOSE
else cout << "max geodesic distance on this mesh: " << maxGeoDist << endl;
#endif
	//clean old extVert.dSpanning value in case it is wanted to be recomputed below (with a good maxGeoDist)
	if (verts[extVert]->dSpanning) //may already be deleted in for-loop above
		delete [] verts[extVert]->dSpanning;
	if (verts[extVert]->dSpanningUnn)
		delete [] verts[extVert]->dSpanningUnn; //arrays allocated with new[] must be deallocated with delete[]
	verts[extVert]->dSpanning = verts[extVert]->dSpanningUnn = NULL;
	
	//other initializations
	for (int v = 0; v < (int) verts.size(); v++)
		verts[v]->processed = false; //initialize this to prevent duplications in samples; true means added to samples
	int delta = (secondMesh ? 1 : 0);

	//samples in robustList are added directly without any computation; i need them w/ their dSpanning[] 'cos they'll be used in getIsometricDistortion(); so do dijkstraShortestPaths() in this loop
	for (int i = 0; i < (int) robustList.size(); i += 2)
	{
		samples.push_back(robustList[i+delta]);
		verts[ robustList[i+delta] ]->sample = true;
		dijkstraShortestPaths(robustList[i+delta]); //dSpanning is good due to good min/maxGeoDist values set above
		verts[ robustList[i+delta] ]->processed = true; //to prevent duplications in samples; true means added to samples
	}

	if (N == 12500) //all verts will be samples; special case; this is not even sampling
		while ((int) samples.size() < N)
			for (int s = 0; s < (int) verts.size(); s++)
			{
				if (! verts[s]->sample)
				{
					samples.push_back(s);
					verts[s]->sample = true;
					verts[s]->processed = true; //added, so not add again later		
				}
			}
	else //usual case
		//fill the remaining N-x items via euclidean-based fps for uniform distribution that lets nice coverage on models; x = (int) robustList.size()/2
		//complexity of FPS below is O(NV), where V is for farthest point search; one can reduce the first V to logV by efficient searching via kd-tree or sth, hence O(NlogV)	
		while ((int) samples.size() < N)
		{
			//in fps, the next sample candidate x_i \in all vertices finds its closest existing sample x_clo and remembers d_i = d(x_i, x_clo); if d_i is the max among all other candidates closest remembered dists, then x_i is selected as the next sample;
			//here my candidates are restricted to nothing, i.e. i use all verts as candidates; so, find the closest existing sample e2 \in samples to s \in verts and add s to samples as nextSample if that closest dist is very big, i.e. the biggest among all closest dists for different s's
			float maxDist = -INF;
			int nextSample = -1;				
			for (int s = 0; s < (int) verts.size(); s++) //fps selects from amongst all vertices
			{
				int e2 = -1;
				if (verts[s]->processed)
					continue; //no chance to select s as a sample 'cos it's already added or is an interior vert

				float dist, minDist = distanceBetween2(verts[ samples[0] ]->coords, verts[s]->coords); //verts[ samples[0] ]->dSpanningUnn[s]; //=INF is tricky 'cos objects may be scaled to those high coordinates			
				for (int e = 0; e < (int) samples.size(); e++) //find the closest existing base e \in samples to s \in verts
				{
					dist = distanceBetween2(verts[ samples[e] ]->coords, verts[s]->coords);
					//if (verts[ samples[e] ]->dSpanningUnn[s] <= minDist) geodesic version
					if (dist <= minDist) //euclidean version
					{
						//minDist = verts[ samples[e] ]->dSpanningUnn[s]; //geodesic version
						minDist = dist; //euclidean version
						e2 = samples[e];
					}
				}
				dist = distanceBetween2(verts[e2]->coords, verts[s]->coords);
				//if (verts[e2]->dSpanningUnn[s] > maxDist)
				if (dist > maxDist)
				{
					//maxDist = verts[e2]->dSpanningUnn[s]; //geodesic version
					maxDist = dist; //euclidean version
					nextSample = s;
				}
			} //end of for s
			if (nextSample == -1) {cout << "can't happen in FPS\n"; exit(0);}
			samples.push_back(nextSample);
			verts[nextSample]->sample = true;
			verts[nextSample]->processed = true; //added, so not add again later		
			//dijkstraShortestPaths(nextSample); //upcoming iteration needs samples.dSpanningUnn which is not currently ready when selecting from the pool of all verts
												 //not needed for euclidean-based FPS; so i get rid of this O(VlogV) complexity which is called for N times :)
		} //end of while N

#ifdef VERBOSE
	//clean old samples.dSpanning values
	for (int i = 0; i < (int) samples.size(); i++)
		if (i < 7 || i >= N-7) //print the first and last 5 sample ids
			cout << samples[i] << " ";
		else if (i == 7)
			cout << " .. ";
	cout << "\n";
	if (secondMesh && N == 100)
		cout << "\nWARNING: incompatible samples for EM comparisons (if this is the intent)\n\n"; //EM uses 100 FPS samples; below some samples (not calling dijkstra) will be slightly different/incompatible
#endif
}

//////////// quicksort /////////////////
/*int partition(float* a, int* vIdxs, int p, int r) { //using float a[], int vIdxs[] again does call-by-ref to arrays
  float x = a[r];
  int xv = vIdxs[r];
  int j = p - 1;
  for (int i = p; i < r; i++) {

    if (x <= a[i]) {
      j = j + 1;
	  //exchange a[i] & a[j]
      float temp = a[j];
      a[j] = a[i];
      a[i] = temp;
	  //exchange corresponding indices vIdxs[i] & vIdxs[j]
      int tmp = vIdxs[j];
      vIdxs[j] = vIdxs[i];
      vIdxs[i] = tmp;
    }
  }
  //exchange a[j+1] & a[r]
  a[r] = a[j + 1];
  a[j + 1] = x;
  //exchange corresponding indices vIdxs[j+1] & vIdxs[r]
  vIdxs[r] = vIdxs[j + 1];
  vIdxs[j + 1] = xv;

  return (j + 1);
}
void quickSort(float* a, int* vIdxs, int p, int r) {
	//quicksort from 24bytes.com/Quick-Sort.html
  if (p < r) {
    int q = partition(a, vIdxs, p, r);
    quickSort(a, vIdxs, p, q - 1);
    quickSort(a, vIdxs, q + 1, r);
  }
}//*/

/*int partition2(float a[], int vIdxs[], int l, int r) {
   int i, j, index = vIdxs[l], tmp;
   float pivot = a[l], t;
   i = l; j = r+1;
		
   while( 1)
   {
   	 do ++i; while( a[i] <= pivot && i <= r );
   	 do --j; while( a[j] > pivot );
   	 if( i >= j ) break;
   	 t = a[i]; a[i] = a[j]; a[j] = t;
	  //exchange corresponding indices vIdxs[i] & vIdxs[j]
      tmp = vIdxs[i];
      vIdxs[i] = vIdxs[j];
      vIdxs[j] = tmp;
   }
   t = a[l]; a[l] = a[j]; a[j] = t;
   //exchange corresponding indices
   tmp = vIdxs[l];
   vIdxs[l] = vIdxs[j];
   vIdxs[j] = tmp;

   return j;
}
void quickSort2(float a[], int vIdxs[], int l, int r)
{
	//quicksort from comp.dit.ie/rlawlor/Alg_DS/sorting/quickSort.c

   int j;
   if( l < r ) 
   {
   	   //divide and conquer
       j = partition2( a, vIdxs, l, r);
       quickSort2( a, vIdxs, l, j-1);
       quickSort2( a, vIdxs, j+1, r);
   }	
}//*/
// An iterative implementation of quick sort
int partitionIter(float arr[], int vIdxs[], int l, int h)
{
	/* This function is same in both iterative and recursive*/
    float x = arr[h];
    int i = (l - 1), xv = vIdxs[h];
 
    for (int j = l; j <= h- 1; j++)
    {
        if (arr[j] <= x)
        {
            i++;
			//exchange a[i] & a[j]
			float temp = arr[j];
			arr[j] = arr[i];
			arr[i] = temp;
	  	    //exchange corresponding indices vIdxs[i] & vIdxs[j]
			int tmp = vIdxs[j];
			vIdxs[j] = vIdxs[i];
			vIdxs[i] = tmp;
        }
    }
	//exchange a[i+1] & a[h]
    arr[h] = arr[i + 1];
    arr[i + 1] = x;
    //exchange corresponding indices vIdxs[i+1] & vIdxs[h]
    vIdxs[h] = vIdxs[i + 1];
    vIdxs[i + 1] = xv;

    return (i + 1);
}
void quickSortIterative(float arr[], int vIdxs[], int l, int h)
{
    // Create an auxiliary stack
    int* stack = new int[ h - l + 1 ];
 
    // initialize top of stack
    int top = -1;
 
    // push initial values of l and h to stack
    stack[ ++top ] = l;
    stack[ ++top ] = h;
 
    // Keep popping from stack while is not empty
    while ( top >= 0 )
    {
        // Pop h and l
        h = stack[ top-- ];
        l = stack[ top-- ];
 
        // Set pivot element at its correct position
        // in sorted array
        int p = partitionIter( arr, vIdxs, l, h );
 
        // If there are elements on left side of pivot,
        // then push left side to stack
        if ( p-1 > l )
        {
            stack[ ++top ] = l;
            stack[ ++top ] = p - 1;
        }
 
        // If there are elements on right side of pivot,
        // then push right side to stack
        if ( p+1 < h )
        {
            stack[ ++top ] = p + 1;
            stack[ ++top ] = h;
        }
    }
	delete [] stack;
}
//////////// quicksort ends /////////////////
void insertionSort(float* A, int s, int* vIdxs)
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
bool isDuplicated(int v, vector< int > vec)
{
	//returns true is if v already exists in the odd indices of vec

	for (int i = 1; i < (int) vec.size(); i += 2)
		if (vec[i] == v)
			return true;
	return false;
}
void Mesh::resultToFile(char* fName, Mesh* mesh2, bool removeOutliers, bool rlComputeMode)
{
	//fprints computed map b/w this mesh and mesh2

#ifdef VERBOSE
	cout << "fprinting computed map to " << fName << "..\n";
#endif
	//to fprintf overall isometric distortion i need to set up the corresp vector
	int N = (int) samples.size();
	corresp.clear();
	for (int i = 0; i < N; i++)
	{
		int v2 = verts[ samples[ i ] ]->matchIdx;
		if (v2 == -1)
		{
#ifdef VERBOSE
			cout << "WARNING: unmatched mesh1 sample: " << samples[ i ] << " (normal if called after adaptiveSampling)\n"; //normal 'cos outliers entering to adaptiveSampling() remain matchIdx=-1
#endif
			continue; //-1 in corresp will make getIsometricDistortion() below crash
//			exit(0);
		}
#ifdef VERBOSE
		if (isDuplicated(v2, corresp)) //this condition guarantees bijection
		{
			cout << "WARNING: many-to-1 match: " << samples[ i ] << "-" << v2 << "\t" << v2 << " duplicated\n";
//			exit(0);
		}
#endif
		corresp.push_back(samples[ i ]);
		corresp.push_back(v2);
//cout << samples[ i ] << "\t" << v2 << "\t" << verts[ samples[ i ] ]->diso << endl;
	}
	
	if (rlComputeMode)
	{		
		//i've less samples so not all diso is reliable after GA optimization (due to coarse sampling); so recompute diso values using the top3 matches as the safeList; top3 is considered to be sufficiently good when dealing with just 10 matches as in here;
		//a more principled way is to detect a diso difference jump value (as done in my Symmetric Flip paper) but that requires average of the first 2 differences (2 in Deformation-Driven paper, 10 in my Flip paper); hence top3 would be selected anyway
		int n = (int) corresp.size() / 2;
		float* disoVals = new float[n];
		int* vIdxs = new int[n];
		//getIsometricDistortion(mesh2, corresp); //fill diso values [getFittestMember() call in GA.go() already called this for the fittest chromosome (this); so this call is redundant]
		for (int i = 0; i < n; i++)
		{
			disoVals[i] = verts[ corresp[2*i] ]->diso;
			vIdxs[i] = i;
		}
		insertionSort(disoVals, n, vIdxs); //call by ref to disoVals (redundant) and vIdxs (output to be used below); descending order
		robustList.clear(); //redundant
		for (int s = n-1; s >= 0; s--) //go from best to worst
		//for (int s = 0; s < n; s++) //go from worst to best		
		{
//cout << vIdxs[s] << "\t" << corresp[ 2*vIdxs[s] ] << " - " << corresp[ 2*vIdxs[s] + 1 ] << " with " << disoVals[s] << " in robustList\n";
			robustList.push_back(corresp[ 2*vIdxs[s] ]); //vIdxs[ss=0] is the worst (highest diso) ss=1 next worst and so on
			robustList.push_back(corresp[ 2*vIdxs[s] + 1 ]);
			if ((int) robustList.size() == 6) //3 matches make 6 entries in robustList
				break;
		}
		delete [] disoVals;
		delete [] vIdxs;

		//no outlier removal in this mode
		removeOutliers = false;
	}

	FILE* fPtr = fopen(fName, "w");
#ifdef OUTLIER_FREE_DISTORTION
	isometricDistortion = (! robustList.empty() ? getIsometricDistortionRL_outlierFree(mesh2, corresp) : getIsometricDistortion_outlierFree(mesh2, corresp)); //diso values updated now
#else
	isometricDistortion = (! robustList.empty() ? getIsometricDistortionRL(mesh2, corresp) : getIsometricDistortion(mesh2, corresp)); //diso values updated now
#endif
#ifndef DO_EM_ALGO //for EM_ALGO outliers already handled before coming to this function so skip it here
	if (removeOutliers)
	{
		//genetic algo may rarely leave an outlier match that is X+ times worse than the average isometryCost; so, just make their matchIdx=-1 to keep the map 1-to-1 (not onto anymore (hence not bijection; just an injection) 'cos some mesh2.verts will be unmatched)
		bool outlierDetected = false;
		float nOutliers = 0, X = 3.0f;//(rlComputeMode ? 1.25f : 3.0f);//(afterAdaptiveSampling ? 3.0f : 2.0f));//for rlComputeMode case i'm selecting the robustList for future usage so be very very picky and select only the very bests
																								 //isometricDistortion is reduced greatly after adaptiveSampling; so to prevent unnecessary outlier removals, increase X
		for (int i = 0; i < (int) samples.size(); i++)
		{
int m = (verts[ samples[i] ]->matchIdx != -1 ? verts[ samples[i] ]->matchIdx : 0);//cout << verts[ samples[i] ]->diso << "\t" << verts[ samples[i] ]->dSpanning[m] << endl;//last one is dgrd (not stored in v.dgrd yet)
			if (verts[ samples[i] ]->diso > X*isometricDistortion && verts[ samples[i] ]->matchIdx != -1 /*||samples[i]==8382||samples[i]==2339||samples[i]==2425*/) //!=-1 is there for the 2nd call to resultToFile()
			{
#ifdef VERBOSE
cout << "outlierrrrrrrrrrr: " << samples[i] << " -> " << verts[ samples[i] ]->matchIdx << " w/ diso = " << verts[ samples[i] ]->diso << " vs. avg isometricDistortion = " << isometricDistortion << "  (dgrd=" << verts[ samples[i] ]->dSpanning[m] << ")\n";
#endif
				verts[ samples[i] ]->sample = mesh2->verts[ verts[ samples[i] ]->matchIdx ]->sample = false; //still in samples[] but not sample anymore; hence Painter.getMatchingLines/SpheresSep will skip this
				verts[ samples[i] ]->matchIdx = mesh2->verts[ verts[ samples[i] ]->matchIdx ]->matchIdx = -1;
				nOutliers++;
				outlierDetected = true;
				if (verts[ samples[i] ]->robust)
				{
					cout << "a robustList match should never be an outlier; sth wrong\n";
//					exit(0);
				}
			}
		}
		if (outlierDetected)
		{
#ifdef VERBOSE
			cout << nOutliers << " outliers removed\told isometricDistortion = " << isometricDistortion << endl;
#endif
			refillCorresp(mesh2); //global corresp is updated now
			isometricDistortion = (! robustList.empty() ? getIsometricDistortionRL(mesh2, corresp) : getIsometricDistortion(mesh2, corresp));
			//corresp[] includes nonoutliers (no matchIdx=-1 sample); transfer these nonoutliers to samples[] 'cos i may continue execution for adaptive sampling, which may be confused by outliers
			samples.clear();
			mesh2->samples.clear();
			for (int s = 0; s < (int) corresp.size() / 2; s++)
			{
				samples.push_back(corresp[2*s]);
				mesh2->samples.push_back(corresp[2*s + 1]);
			}
		}
	}
#endif
	float groundTruthDistortion = getGroundDistortion(mesh2, corresp, true),
		  groundDistoFull = getGroundDistortionFull(mesh2, corresp),
		  isoDistoNoRL = (rlComputeMode ? -1 : getIsometricDistortion(mesh2, corresp)); //isometric distortion that traverses map itself; it is hence comparable with other algos, eg my EM algo, where there's no robustList; diso values updated again unless this is rlComputeMode
	//fprintf(fPtr, "%d %f %f %f %f %f\n", (int) corresp.size() / 2, isometricDistortion, isoDistoNoRL, groundDistoFull, maxGeoDist, mesh2->maxGeoDist);
	fprintf(fPtr, "%d %f %f %f %f\n", (int) corresp.size() / 2, isoDistoNoRL, groundDistoFull, maxGeoDist, mesh2->maxGeoDist); //isometricDistortion is based on robust list, hence not comparable with other methods; isoDistoNoRL fprinted here corresponds to the D_iso in paper
#ifdef VERBOSE
	cout << "\n" << isometricDistortion << ", " << isoDistoNoRL << " && " << groundDistoFull << " for Diso.robust, Diso && ground-truth distortions of the resulting map\n";
#else
	cout << "Diso (based on robustList) = " << isometricDistortion << "; Diso (comparable w/ other methods) = " << isoDistoNoRL << " && Dgrd = " << groundDistoFull << "\n";
#endif
	for (int i = 0; i < (int) corresp.size(); i += 2)
	{
		fprintf(fPtr, "%d\t%d\n", corresp[i], corresp[i+1]);
//		if (rlComputeMode || i % 20 == 0)
//			cout << corresp[i] << "-" << corresp[i+1] << "\t" << verts[ corresp[i] ]->agdOrder << " vs. " << mesh2->verts[ corresp[i+1] ]->agdOrder << "\t" << verts[ corresp[i] ]->diso << "\t" << verts[ corresp[i] ]->dgrd << "\t" << verts[ corresp[i] ]->agd << endl;
#ifdef VERBOSE
//		if (verts[ corresp[i] ]->dgrd > 0.25f) //toe to knee normalized geo distance is 0.25f; assumes availability of i-i ground truth pairs
//			cout << "-------------bad-------------" << corresp[i] << "-" << corresp[i+1] << "\t" << verts[ corresp[i] ]->agdOrder << " vs. " << mesh2->verts[ corresp[i+1] ]->agdOrder << "\t" << verts[ corresp[i] ]->diso << "\t" << verts[ corresp[i] ]->dgrd << "\t" << verts[ corresp[i] ]->drank << endl;
#endif
	}
	fclose(fPtr);

	//some statistics on extremes
	float minIso = INF, minGrd = INF, maxIso = -INF, maxGrd = -INF;
	int mii, mig, mai, mag;
	for (int i = 0; i < N; i++)
	{
		if (verts[ corresp[2*i] ]->diso < minIso)
		{
			minIso = verts[ corresp[2*i] ]->diso;
			mii = corresp[2*i];
		}
		if (verts[ corresp[2*i] ]->dgrd < minGrd)
		{
			minGrd = verts[ corresp[2*i] ]->dgrd;
			mig = corresp[2*i];
		}
		if (verts[ corresp[2*i] ]->diso > maxIso)
		{
			maxIso = verts[ corresp[2*i] ]->diso;
			mai = corresp[2*i];
		}
		if (verts[ corresp[2*i] ]->dgrd > maxGrd /*&& corresp[2*i]!=9358*/) //see second worst w/ the 2nd condition
		{
			maxGrd = verts[ corresp[2*i] ]->dgrd;
			mag = corresp[2*i];
			worstGrd = corresp[2*i]; mesh2->worstGrd = corresp[2*i+1]; //just to see worst grd truth match
		}
	}
#ifdef VERBOSE
	cout << "minIso distortion\t" << minIso << "\t" << mii << "-" << verts[mii]->matchIdx << "(grd= " << verts[mii]->dgrd << ")\n" << "minGrd distortion\t" << minGrd << "\t" << mig << "-" << verts[mig]->matchIdx << "(iso= " << verts[mig]->diso << ")\n" <<
		    "maxIso distortion\t" << maxIso << "\t" << mai << "-" << verts[mai]->matchIdx << "(grd= " << verts[mai]->dgrd << ")\n" << "maxGrd distortion\t" << maxGrd << "\t" << mag << "-" << verts[mag]->matchIdx << "(iso= " << verts[mag]->diso << ")\ndone!\n";
#endif
//cout << sizeof(mesh2) << "\t" << sizeof(*mesh2) << "\t" << sizeof(int) << "\t" << sizeof(N) << "\n"; //prints 4 112 4 4; so mesh2 pointer is efficient to transfer, e.g. just like an integer variable
}

void Mesh::appendToFile(char* fName, Mesh* mesh2)
{
	//fappends computed full (no outlier removal) map b/w this mesh and mesh2 so that i can fread it later as an initial population

#ifdef VERBOSE
	cout << "appending computed map to " << fName << "..\n";
#endif	
	int N = (int) samples.size();
	//if this map currently exists in fName then do not append it in vain (no harm if appended but not a good practice, i.e. save space, unrepeated (hence unbiased/unweighted) maps)
	bool exists = false, appendAlways = true; //skip the exists test and append at all times
	FILE* fPtr;
	if (! appendAlways)
	{
		fPtr = fopen(fName, "r");
		if (fPtr)
		{
			int n, v1, v2;
			while (fscanf(fPtr, "%d\n", &n) != EOF) //go till the end of file
			{
				if (N != n)
				{
					cout << "incompatible map in file: " << fName << ": " << n << " vs. " << N << endl;
					exit(0);
				}
				bool differentMatch = false;
				for (int i = 0; i < N; i++)
				{
					fscanf(fPtr, "%d\t%d\n", &v1, &v2);
					if (v1 != samples[i])
					{
						cout << "incompatible map in file: " << fName << ": " << v1 << " vs. " << samples[i] << endl;
						exit(0);
					}
					if (v2 != verts[ samples[i] ]->matchIdx) //v1 == samples[i] always true here so i just check v2
						differentMatch = true; //don't breakk; here 'cos i need to fscanf'te remaining matches 'till this N-set is completely read out
				}
				if (! differentMatch)
				{
					exists = true;
					cout << "current GA map already exists in file so not appended\n";
					break;
				}
			}
			fclose(fPtr);
		}
	}
	if (exists)
		return; //not append

	///////////////////// appending /////////////////////
	fPtr = fopen(fName, "a");
	if (fPtr) //if folder not exists then fPtr becomes NULL
	{
		fprintf(fPtr, "%d\n", N); //special flag (map size) to detect the beginning of this map later when i fread it
		for (int i = 0; i < N; i++)
		{
			int v2 = verts[ samples[ i ] ]->matchIdx;
			if (v2 == -1)
			{
				cout << "WARNING: unmatched mesh1 sample: " << samples[ i ] << endl; //" (normal if called after adaptiveSampling)\n"; never called after adaptiveSampling so cannot happen
				exit(0);
			}
			if (isDuplicated(v2, corresp)) //this condition guarantees bijection
			{
				cout << "WARNING: many-to-1 match: " << samples[ i ] << "-" << v2 << "\t" << v2 << " duplicated\n";
				exit(0);
			}
			fprintf(fPtr, "%d\t%d\n", samples[ i ], v2);
		}
		fclose(fPtr);
	}
	///////////////////// appending ends /////////////////////
}

void Mesh::resultFromFile(char* fName, Mesh* mesh2)
{
	//freads precomputed map b/w this mesh and mesh2

#ifdef VERBOSE
	cout << "freading precomputed map from " << fName << "..\n";
#endif
	FILE* fPtr;
	if (! (fPtr = fopen(fName, "r")))
	{
		cout << "cannot read " << fName << endl;
		exit(0);
	}

	//sprintf(fName, "output\\robustList\\%d-%d.dat", id, mesh2->id);
	sprintf(fName, "input\\robustMaps\\%d-%d.dat", id, mesh2->id);
	fillRobustTraversalList(fName); //this should match w/ fName (and also ensure that you used this robustList in the computation of fName)

	int N;
	float isoDistoNoRL, groundDistoFull;
	corresp.clear();
	fscanf(fPtr, "%d %f %f %f %f %f\n", &N, &isometricDistortion, &isoDistoNoRL, &groundDistoFull, &maxGeoDist, &mesh2->maxGeoDist);
	for (int i = 0; i < N; i++)
	{
		int v1, v2;
		fscanf(fPtr, "%d\t%d\n", &v1, &v2);

		if (v1 == -1 || v2 == -1 || v1 > (int) verts.size() || v2 > (int) mesh2->verts.size())
		{
			cout << "bad index " << v1 << " || " << v2 << "\n";
			exit(0);
		}
		corresp.push_back(v1);
		corresp.push_back(v2);
		verts[v1]->matchIdx = v2;
		mesh2->verts[v2]->matchIdx = v1; //dual operation
		verts[v1]->sample = mesh2->verts[v2]->sample = true; //to enable line coloring
		//since maxGeoDist is fread, i can safely compute dSpanning[]; w/o maxGeoDist i'd need to call computeEvenlySpacedSamples()
		dijkstraShortestPaths(v1);
		mesh2->dijkstraShortestPaths(v2);
	}
	fclose(fPtr);

	float newIsoDistoNoRL = getIsometricDistortion(mesh2, corresp),
		  newIsometricDistortion = 9.9f,//getIsometricDistortionRL(mesh2, corresp), //diso values after rl-based distoriton computation will be used to decided min/ax distortions below (makes sense 'cos outliers removed based on these disos)
		  newGroundDistoFull = getGroundDistortionFull(mesh2, corresp);
	float minIso = INF, minGrd = INF, maxIso = -INF, maxGrd = -INF;
	int mii, mig, mai, mag;
	for (int i = 0; i < N; i++)
	{
		if (verts[ corresp[2*i] ]->diso < minIso)
		{
			minIso = verts[ corresp[2*i] ]->diso;
			mii = corresp[2*i];
		}
		if (verts[ corresp[2*i] ]->dgrd < minGrd)
		{
			minGrd = verts[ corresp[2*i] ]->dgrd;
			mig = corresp[2*i];
		}
		if (verts[ corresp[2*i] ]->diso > maxIso)
		{
			maxIso = verts[ corresp[2*i] ]->diso;
			mai = corresp[2*i];
		}
		if (verts[ corresp[2*i] ]->dgrd > maxGrd /*&& corresp[2*i]!=9358*/) //see second worst w/ the 2nd condition
		{
			maxGrd = verts[ corresp[2*i] ]->dgrd;
			mag = corresp[2*i];
			worstGrd = corresp[2*i]; mesh2->worstGrd = corresp[2*i+1]; //just to see worst grd truth match
		}
	}
#ifdef VERBOSE
	cout << isometricDistortion << " == " << newIsometricDistortion << "\n" << isoDistoNoRL << " == " << newIsoDistoNoRL << "\n" << groundDistoFull << " == " << newGroundDistoFull << "\n\n";
	cout << "minIso distortion\t" << minIso << "\t" << mii << "-" << verts[mii]->matchIdx << "(grd= " << verts[mii]->dgrd << ")\n" << "minGrd distortion\t" << minGrd << "\t" << mig << "-" << verts[mig]->matchIdx << "(iso= " << verts[mig]->diso << ")\n" <<
		    "maxIso distortion\t" << maxIso << "\t" << mai << "-" << verts[mai]->matchIdx << "(grd= " << verts[mai]->dgrd << ")\n" << "maxGrd distortion\t" << maxGrd << "\t" << mag << "-" << verts[mag]->matchIdx << "(iso= " << verts[mag]->diso << ")\n\ndone!\n";
#endif
}

void Mesh::resultFromBIMFile(char* fName, Mesh* mesh2)
{
	//freads precomputed BIM map b/w this mesh and mesh2

#ifdef VERBOSE
	cout << "freading precomputed BIM map from " << fName << "..\n";
#endif
	FILE* fPtr;
	if (! (fPtr = fopen(fName, "r")))
	{
		cout << "cannot read " << fName << endl;
		exit(0);
	}

	int N = 100;
	computeEvenlySpacedSamples(N); //my algorithm has correspondence info of these 100 samples of this mesh1 so create them again for the BIM algo
	mesh2->computeEvenlySpacedSamples(N); //inits maxGeoDist which is required for mesh2.dSpanning[] values

	corresp.clear();
	int v1, v2;
	while (fscanf(fPtr, "%d\n", &v1) != EOF)
	{
		fscanf(fPtr, "%d\n", &v2);
		for (int s = 0; s < (int) samples.size(); s++)
			if (samples[s] == v1)
			{
				//below copied from resultFromFile()
				corresp.push_back(v1);
				corresp.push_back(v2);
				verts[v1]->matchIdx = v2;
				mesh2->verts[v2]->matchIdx = v1; //dual operation
				verts[v1]->sample = mesh2->verts[v2]->sample = true; //to enable line coloring
				//since maxGeoDist is fread, i can safely compute dSpanning[]; w/o maxGeoDist i'd need to call computeEvenlySpacedSamples()
				dijkstraShortestPaths(v1);
				mesh2->dijkstraShortestPaths(v2);
				break;
			}
	}
	fclose(fPtr);
	if ((int) corresp.size() != 2 * (int) samples.size())
	{
		cout << "WARNING: sizes should have matched\n";	
		exit(0);
	}

	float newIsoDistoNoRL = getIsometricDistortion(mesh2, corresp),
		  newGroundDistoFull = getGroundDistortionFull(mesh2, corresp);
	float minIso = INF, minGrd = INF, maxIso = -INF, maxGrd = -INF;
	int mii, mig, mai, mag;
	for (int i = 0; i < (int) corresp.size() / 2; i++)
	{
		if (verts[ corresp[2*i] ]->diso < minIso)
		{
			minIso = verts[ corresp[2*i] ]->diso;
			mii = corresp[2*i];
		}
		if (verts[ corresp[2*i] ]->dgrd < minGrd)
		{
			minGrd = verts[ corresp[2*i] ]->dgrd;
			mig = corresp[2*i];
		}
		if (verts[ corresp[2*i] ]->diso > maxIso)
		{
			maxIso = verts[ corresp[2*i] ]->diso;
			mai = corresp[2*i];
		}
		if (verts[ corresp[2*i] ]->dgrd > maxGrd /*&& corresp[2*i]!=9358*/) //see second worst w/ the 2nd condition
		{
			maxGrd = verts[ corresp[2*i] ]->dgrd;
			mag = corresp[2*i];
			worstGrd = corresp[2*i]; mesh2->worstGrd = corresp[2*i+1]; //just to see worst grd truth match
		}
	}
#ifdef VERBOSE
	cout << newIsoDistoNoRL << " &&&&&&&&& " << newGroundDistoFull << "\n\n";
	cout << "minIso distortion\t" << minIso << "\t" << mii << "-" << verts[mii]->matchIdx << "(grd= " << verts[mii]->dgrd << ")\n" << "minGrd distortion\t" << minGrd << "\t" << mig << "-" << verts[mig]->matchIdx << "(iso= " << verts[mig]->diso << ")\n" <<
		    "maxIso distortion\t" << maxIso << "\t" << mai << "-" << verts[mai]->matchIdx << "(grd= " << verts[mai]->dgrd << ")\n" << "maxGrd distortion\t" << maxGrd << "\t" << mag << "-" << verts[mag]->matchIdx << "(iso= " << verts[mag]->diso << ")\n\ndone!\n";
#endif
}

void Mesh::computeAGD(bool robustListInUse)
{
	//computes average geodesic distance (AGD) descriptor for each sample point

#ifdef VERBOSE
	cout << "computing AGDs..\n";
#endif
	float avgAGD = 0.0f;
/*	if (robustListInUse)
		for (int i = 0; i < (int) samples.size(); i++)
		{
			int nAdds = 0;
			for (int j = 0; j < (int) robustList.size(); j += 2)
			{
				int s = robustList[j + (secondMesh ? 1 : 0)];
				if (s != samples[i])
				{
					verts[ samples[i] ]->agd += verts[s]->dSpanning[ samples[i] ]; //dSpanning defined for robustList samples for sure
					nAdds++;
					if (verts[s]->dSpanning[ samples[i] ] > verts[ samples[i] ]->ecc)
						verts[ samples[i] ]->ecc = verts[s]->dSpanning[ samples[i] ];
				}
			}
			verts[ samples[i] ]->agd /= nAdds;
			avgAGD += verts[ samples[i] ]->agd;
		}
	else*/
		for (int i = 0; i < (int) samples.size(); i++)
		{
			if (! verts[ samples[i] ]->dSpanningUnn)
			{
#ifdef VERBOSE
				cout << "weird; sample vert " << samples[i] << " must have learned its geodesics by now; no problem it's learning right now\n";
#endif
				dijkstraShortestPaths(samples[i]);
			}
			//nAdds stuff may be used here as well but (int) verts.size() is too big so does not matter whether i divide (int) verts.size() or (int) verts.size() - 1
			for (int j = 0; j < (int) verts.size(); j++)
			{
				verts[ samples[i] ]->agd += verts[ samples[i] ]->dSpanning[j]; //dSpanning defined for all samples for sure
				if (verts[ samples[i] ]->dSpanning[j] > verts[ samples[i] ]->ecc)
					verts[ samples[i] ]->ecc = verts[ samples[i] ]->dSpanning[j];
			}
			verts[ samples[i] ]->agd /= (int) verts.size();
			avgAGD += verts[ samples[i] ]->agd;
		}
	avgAGD /= (int) samples.size();
	
#ifdef VERBOSE
	cout << "average agd = " << avgAGD << "\n\n";
#endif
	sortAGDs();
}

void Mesh::computeGV()
{
	//computes geodesic vector (GV) descriptor for each sample point; gv for a sample is a set of geodesic distances to all robustList vertices on the corresponding mesh

	if (robustList.empty())
	{
		//GV computation requires robust matches in robustList; note that if robustListInUse=false then getIsometricDistortion() will be called which does not use this robustList
		cout << "warning robustList empty!!!!!\n";
		exit(0);
	}

#ifdef VERBOSE
	cout << "computing GVs..\n";
#endif
	for (int j = 0; j < (int) robustList.size(); j += 2)
		dijkstraShortestPaths(robustList[j + (secondMesh ? 1 : 0)]); //redundant for mesh1; necessary for mesh2 only if fillRobustTraversalList() uses a file computed by adaptiveSampling (mesh2 samples adaptively changed so not compatible w/ FPS samples here)
	//similar to AGD computation; instead of adding distances-to-robustMatches (agd), store each distance-to-robustMatch in gv (geodesic vector)
	for (int i = 0; i < (int) samples.size(); i++)
		for (int j = 0; j < (int) robustList.size(); j += 2)
			verts[ samples[i] ]->gv.push_back(verts[ robustList[j + (secondMesh ? 1 : 0)] ]->dSpanning[ samples[i] ]); //dSpanning defined for robustList samples for sure
#ifdef VERBOSE
	cout << "done!\n\n";
#endif
}

void Mesh::refillCorresp(Mesh* mesh2)
{
	//clears and then refills corresp w/ current samples

	//empty it
	corresp.clear();
	//refill it
	for (int i = 0; i < (int) samples.size(); i++)
	{
		int bv1 = samples[i], bv2 = verts[bv1]->matchIdx;
		if (bv2 == -1)
			continue;

		corresp.push_back(bv1); //next safe correspondence goes to following 2 slots in corresp
		corresp.push_back(bv2);
	}
}

void Mesh::sortAGDs()
{
	//creates and fills agdOrder[] w/ vertex idxs corresponding to the descending values of vertex AGD values

#ifdef VERBOSE
	cout << "sorting verts w.r.t. AGD values..\n";
#endif
	int N = (int) samples.size();
	//efficient nlogn sorting, n = # of verts
	float* agdVals = new float[N];
	int* vIdxs = new int[N]; //same idea as in sortedIdxs
	for (int i = 0; i < N; i++)
	{
		agdVals[i] = verts[ samples[i] ]->agd;
		vIdxs[i] = i;
	}
//quickSort2(agdVals, vIdxs, 0, N); //call be ref to agdVals (redundant) and vIdxs (output to be used below)
//quickSort and quicksort2 sometimes lead to `invalid entry error! 0 1392007378' error for some weird reason (they always crush when vIdxs not global); so i switch it w/ nonrecursive/simple O(N^2) insertionSort since N is small, e.g. N=100 samples
//quickSortIterative(agdVals, vIdxs, 0, N); even this iterative one crushes when vIDxs not global; weird!
	insertionSort(agdVals, N, vIdxs); //call by ref to gcVals (redundant) and vIdxs (output to be used below)
	for (int i = 0; i < N; i++)
	{
		//vIdxs[0] is now associated with the max AGD value, vIdxs[1] ~ nextMaxAGD, vIdxs[2] ~ nextMaxAGD, and so on; so tell the vIdxs[0]'th sample that he has the highest order (i=0)
		verts[ samples[ vIdxs[i] ] ]->agdOrder = i; //verts[ samples[ i ] ]->agdOrder = vIdxs[i]; was my wrong version; took 5 days to see :(
//cout<<vIdxs[i]<<" " << agdVals[i] << endl;
//sanity check
for (int j = i-1; j >= 0; j--) if (verts[ samples[ vIdxs[i] ] ] == verts[ samples[ vIdxs[j] ] ]) { cout << "duplicate error! " << i << " " << vIdxs[i] << endl; exit(0); }
	}
//sanity check
for (int j = N-1; j >= 0; j--) if (verts[ samples[j] ]->agdOrder >= N) { cout << "invalid entry error! " << j << " " << verts[ samples[j] ]->agdOrder << endl; exit(0); }
	
//for (int i = 0; i < N; i++) cout << "verts" << samples[i] << "\thas agdOrder of\t" << verts[ samples[i] ]->agdOrder << " and agdValue of " << verts[ samples[i] ]->agd << endl;

	delete [] agdVals;
	delete [] vIdxs;
#ifdef VERBOSE
	cout << "done!\n\n";
#endif
}

void Mesh::fillRobustTraversalList(char* fName)
{
	//fread the robust map file to fill the robustList with; i could have also run that algo instead but this keeps the code simple

	FILE* fPtr;
	if (! (fPtr = fopen(fName, "r")))
	{
		cout << "cannot read robust map file " << fName << "\t>>>> run in mode=1 first\n";
		exit(0);
	}
	int size, v1, v2;
	float a, b, c, d;
	fscanf(fPtr, "%d %f %f %f %f\n", &size, &a, &b, &c, &d);
#ifdef VERBOSE
	cout << "robust list for genetic algo being filled (" <<  size << ")\n";
#endif
	for (int m = 0; m < size; m++)
	{
		fscanf(fPtr, "%d\t%d\n", &v1, &v2);
		robustList.push_back(v1);
		robustList.push_back(v2);
//#ifndef LMDS_INIT //if lmds init will be used i need matchIdxs=-1 for all verts in initialSpectralMatching(); no!!!!! i do it there explicitly; forceRobustMatchesInInit affects non-lmds mappings (due to initMatchCandids); besides initialSpectralMatching() uses mesh2.matchIdxs not this mesh; so no problem if i set matchIdx here
		verts[v1]->matchIdx = v2; //required for forceRobustMatchesInInit case; will be overwritten in the end of GA.go()
//#endif
//cout << v1 << "\t" << v2 << endl;
	}
#ifdef VERBOSE
	cout << "done!\n\n";
#endif
}

//#define USE_CLOSEST_ROBUST_MATCH //undefine this to use geodesics to ALL robust matches, i.e. the gv vector; define to use only the closest robust match [info: if this is in function def, then it affects another function; so for good readability put it outside of any function definition]
void Mesh::adaptiveSampling(Mesh* mesh2, bool keyboardSampling)
{
	//change mesh2.samples (nonsafe ones) to better match fixed mesh1.samples; replace bad samples with patch vertices (only 1 ordering (e.g. worst to best) can be tried unlike the GA's multiple orderings)

#ifdef VERBOSE
	cout << "adaptive sampling on mesh2 to improve correspondences..\n";
#endif
	int n = (int) samples.size();
	mesh2->computeRadius();

	//each sample learns its patch by selecting the verts that are closer than radiusN
	float totalPatchSize = 0.0f;
//mesh2->closeEnoughUnn = mesh2->radius / 8;
	mesh2->closeEnoughUnn = mesh2->radius; //note that i can use big radius here (not radius/8) 'cos GA will eventualy hit a correct ordering with even a big radius
	for (int i = 0; i < n; i++)
	{
		mesh2->verts[ mesh2->samples[i] ]->patch.clear();//necessary only if adaptiveSampling() will be called 2+ times; hence redundant here
		for (int v = 0; v < (int) mesh2->verts.size(); v++)
			if (mesh2->verts[ mesh2->samples[i] ]->dSpanningUnn[ v ] <= mesh2->closeEnoughUnn)
				mesh2->verts[ mesh2->samples[i] ]->patch.push_back(v); //mesh2->samples[i] itself is also in the patch
		totalPatchSize += (int) mesh2->verts[ mesh2->samples[i] ]->patch.size();
	}
#ifdef VERBOSE
	cout << "avg patch size = " << totalPatchSize / n << " vertices\n";
#endif
	for (int v = 0; v < (int) mesh2->verts.size(); v++)
		mesh2->verts[v]->processed = mesh2->verts[v]->sample = false; //processed=true means unavailable (already inserted to samples[] or made unavailable by being close to a final sample); also reset the sample values
	bool robustListUnaffected = true; //set this to true to keep the robustList matches intact, i.e. unaffected by adaptive sampling
	if (robustListUnaffected)
		for (int v = 0; v < (int) mesh2->verts.size(); v++)
			if (mesh2->verts[v]->robust) //everyone sufficiently close to robust v (including v itself) is marked as unavailable (via processed=true) to preserve evenly-spaced sampling
				for (int i = 0; i < (int) mesh2->verts.size(); i++)
					if (mesh2->verts[v]->dSpanningUnn[i] <= mesh2->closeEnoughUnn)
						mesh2->verts[i]->processed = true;

	if (keyboardSampling)
		return; //below will be done by keyboard hits later

	int gvSize = (int) mesh2->verts[ mesh2->samples[0] ]->gv.size(); //gvSize'll be same (and equal to robustList.size/2) for all samples
#ifndef USE_CLOSEST_ROBUST_MATCH
	if (gvSize == 0)
		cout << "\nWARNING: GVs are empty\n\n";
#endif
	//process mesh2 samples from worst to best w.r.t. diso
	bool orderedProcessing = true;
	int* vIdxs;
	if (orderedProcessing)
	{
		float* disoVals = new float[n];
		vIdxs = new int[n]; //same idea as in sortedIdxs
		for (int i = 0; i < n; i++)
		{
			disoVals[i] = mesh2->verts[ mesh2->samples[i] ]->diso;
			vIdxs[i] = i;
		}
		insertionSort(disoVals, n, vIdxs); //call by ref to disoVals (redundant) and vIdxs (output to be used below)
		delete [] disoVals;
	}
bool first=true;//for (int s = 0; s < n; s++)cout << mesh2->samples[s] << "s\t";cout<<endl;
//FILE* fPtr = fopen("dellater.dat", "w");float minC = INF, maxC = -INF, totalC = 0, nAddsC = 0;
	for (int s = 0; s < n; s++) //go from worst to best [doing the worst first makes sense 'cos the late-comers will probably not move due to marked/processed surroundings; and bests (latecomers) are ok to not move]
	//for (int s = n-1; s >= 0; s--) //go from best to worst [less accurate than worst to best]
	{
		int ns = mesh2->samples[ (orderedProcessing ? vIdxs[s] : s) ]; //ns for (sorted) next sample (vIdxs[ns=0] is the worst (highest diso) ss=1 next worst and so on)

		mesh2->verts[ns]->sample = true; //all verts are reset to sample=false above; so ns begins as a sample (may change below)
//cout << s << "-" << ns << "\t\t" << mesh2->verts[ns]->diso << (mesh2->verts[ns]->matchIdx == -1) << "\t" << mesh2->verts[ns]->robust << endl;
		if (mesh2->verts[ns]->matchIdx == -1) //outlier mesh2.sample will not be processed further 'cos it has no match
		{
			mesh2->verts[ns]->sample = false; //outliers are not samples anymore (discard them in radius computations below; dijkstraShortestPathsBFS2() also requires sample=true/false knowledge)
			continue; //don't update outlier samples
		}

		if (robustListUnaffected && mesh2->verts[ns]->robust) //robust samples are fixed and cannot move if we are in robustListUnaffected mode
			continue; //don't update robust samples

		int bestPatchVert = -1;
#ifdef USE_CLOSEST_ROBUST_MATCH
		//replace ns w/ the vert in ns.patch which has the most compatible geodesic distance when compared with X = verts[ mesh2->verts[ns]->matchIdx ] to closestRobustIn1 geodesic distance
		int closestRobustIn1 = -1, closestRobustIn2 = -1;
		float X = INF;
		for (int i = 0; i < (int) robustList.size(); i += 2)
			if (verts[ robustList[i] ]->dSpanning[ mesh2->verts[ns]->matchIdx ] < X)
			{
				X = verts[ robustList[i] ]->dSpanning[ mesh2->verts[ns]->matchIdx ];
				closestRobustIn1 = robustList[i];
				closestRobustIn2 = robustList[i+1];
			}
if (closestRobustIn1 == -1) { cout << "no -1\n"; exit(0);}

		float minCost = INF; 
		for (int p = 0; p < (int) mesh2->verts[ns]->patch.size(); p++) //patch includes ns itself so ns may not be replaced at all (if it beats all other patch vertices)
		{
			int pv = mesh2->verts[ns]->patch[p];
			if (mesh2->verts[pv]->processed && pv != ns) //processed=true means unavailable to be used as a potential final sample; 2nd condition enables ns itself to be tested as candidate, which gives ns a change to remain as a sample
				continue; //thanks to 2nd condition, ns escapes this continue even if ns.processed=true; this gives me a change to not update the existing sample; seeing this took my 5 weeks :(

			//find geodesic distance of p to desired robustList mesh2.vert (closestRobustIn2)
			float Y = mesh2->verts[ closestRobustIn2 ]->dSpanning[pv];
			if (fabs(X - Y) < minCost)
			{
				minCost = fabs(X - Y);
				bestPatchVert = pv;
			}
		}
#else
//fprintf(fPtr, "\n\nfor %d::::: ", ns);
		//replace ss w/ the vert in ss.patch which has the most compatible gv w/ verts[ mesh2->verts[ss]->matchIdx ]->gv
		float minCost = INF;
		for (int p = 0; p < (int) mesh2->verts[ns]->patch.size(); p++) //patch includes ss itself so ss may not be replaced at all (if it beats all other patch vertices)
		{
			int pv = mesh2->verts[ns]->patch[p];
			if (mesh2->verts[pv]->processed && pv != ns) //processed=true means unavailable to be used as a potential final sample; 2nd condition enables ns itself to be tested as candidate, which gives ns a change to remain as a sample
				continue; //thanks to 2nd condition, ns escapes this continue even if ns.processed=true; this gives me a change to not update the existing sample; seeing this took my 5 weeks :(

			//find geodesic vector of pv to robustList mesh2.vertices
			if (mesh2->verts[pv]->gv.empty()) //may be already filled due to a previous chromosome
				for (int i = 0; i < (int) robustList.size(); i += 2) //mesh1->robustList = mesh2->robustList so use either of them
					mesh2->verts[pv]->gv.push_back(mesh2->verts[ robustList[i+1] ]->dSpanning[pv]);

			float cost = 0.0f;
			for (int i = 0; i < gvSize; i++)
				cost += fabs(verts[ mesh2->verts[ns]->matchIdx ]->gv[i] - mesh2->verts[pv]->gv[i]);
			cost /= gvSize;
//fprintf(fPtr, "%f\t", cost);if (cost<minC) minC = cost; if (cost>maxC) maxC = cost; totalC += cost;nAddsC++;
			if (cost < minCost)
			{
				minCost = cost;
				bestPatchVert = pv;
			}
		}
#endif
		if (bestPatchVert != ns && bestPatchVert != -1) //rarely bestPatchVert remains -1 since all the patch vertices were processed (made unavailable) by neighboring sample's dijkstra calls
		{
#ifdef VERBOSE
cout << "old sample " << ns << " shifted " << mesh2->verts[ns]->dSpanning[bestPatchVert] << " unit to the new sample " << bestPatchVert << "\t\t" << s << endl;
#endif
if (first){mesh2->mrIdx = ns;mesh2->mrIdx2=bestPatchVert;first=false;}
			//update variables accordingly
			mesh2->verts[ns]->sample = false; //not a sample anymore

//			if (finalCall) //dont alter matchIdx values unless this is the final call; for intermediate calls, setting matchIdx = -1 will make the upcoming calls wrongly continue; above (outlier part)
			{
				mesh2->verts[bestPatchVert]->matchIdx = mesh2->verts[ns]->matchIdx;
				mesh2->verts[ns]->matchIdx = -1;
				verts[ mesh2->verts[bestPatchVert]->matchIdx ]->matchIdx = bestPatchVert;
			}
			mesh2->samples[ (orderedProcessing ? vIdxs[s] : s) ] = bestPatchVert; //vIdxs[s] was holding ns but now it is replaced by bestPatchVert
			mesh2->verts[bestPatchVert]->sample = true; //just became a sample
			
			//mark everyone sufficiently close to new sample (including bestPatchVert itself) as unavailable to preserve evenly-spaced sampling; to do so first set the patch of bestPatchVert by filling dSpanningUnn[]
			bool immediateReturn = mesh2->dijkstraShortestPaths(bestPatchVert, mesh2->closeEnoughUnn); //since evalSolution2() is called multiple times, bestPatchVert.dSpanningUnn might have been nonNULL which returns this call immediately (w/o setting processed); make sure to set processed values via markNeighborhood() in that case
			if (immediateReturn)
				mesh2->markNeighborhood(bestPatchVert, mesh2->closeEnoughUnn, true);
		}
		else //current ns will still act as a sample in this chromosome; so set its patch to unavailable just like above
			//same as above but never use bestPatchVert here
			mesh2->markNeighborhood(ns, mesh2->closeEnoughUnn, true);//ns.dijkstraShortestPaths already called for sure (since ns is an existing sample before adaptiveSampling(); so skip dijkstraShortestPaths/Shortcut/Euc() call here
	}
//fprintf(fPtr, "\n\nmin & avg & max costs = %f & %f & %f\n", minC, totalC/nAddsC, maxC);fclose(fPtr); //min & avg & max costs = 0.012842 & 0.043618 & 0.112773 (toe-to-knee normalized geo is 0.25)
	if (orderedProcessing)
		delete [] vIdxs;

	//new radius based on the current sampling in samples[]
	float radius2 = 0.0f, minRadius2 = INF;//, radiusN2 = 0.0f;
	int nDegenerates = 0, nAdds = 0, closest;
	for (int i = 0; i < n; i++) //this radius computation is mandatory for dijkstraShortestPathsBFS() mode; this or just-above (comment-out) can be used for other modes
	{
		if (! mesh2->verts[ mesh2->samples[i] ]->sample)
			continue; //discard outliers in radius computation
		float minDist = INF;
		for (int j = 0; j < n; j++) //this radius computation is mandatory for dijkstraShortestPathsBFS() mode; this or just-above (comment-out) can be used for other modes
			if (i != j && mesh2->verts[ mesh2->samples[j] ]->sample && mesh2->verts[ mesh2->samples[i] ]->dSpanningUnn[ mesh2->samples[j] ] < minDist) //2nd condition: discard outliers in radius computation
			{
				minDist = mesh2->verts[ mesh2->samples[i] ]->dSpanningUnn[ mesh2->samples[j] ];
				closest = mesh2->samples[j];
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
			minDist = mesh2->dijkstraShortestPathsBFS2(mesh2->samples[i], closest); //call by ref to closest so that it keeps the closest sample to samples[i]*/
		}

		radius2 += minDist;
		nAdds++;
		//radiusN2 += minDistN;
#ifdef MIN_RADIUS_IN_USE
		if (minDist < minRadius2)
		{
			minRadius2 = minDist;
			mesh2->mrIdx = mesh2->samples[i]; //index that gives the min radius
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
//	if (finalCall)
	{
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

#ifdef VERBOSE
#ifdef MIN_RADIUS_IN_USE
	float denom = (mesh2->minRadius > minRadius2 ? mesh2->minRadius : minRadius2);
	cout << "fitness = " << 1.0f - fabs(mesh2->minRadius - minRadius2) / denom << endl;
#else
	float denom = (mesh2->radius > radius2 ? mesh2->radius : radius2);
	cout << "fitness = " << 1.0f - fabs(mesh2->radius - radius2) / denom << endl;
#endif
#endif

	char fName[250]; sprintf(fName, "output\\%d-%das.dat", id, mesh2->id);
	resultToFile(fName, mesh2, true, false); //as: adaptive sampling
}

int Mesh::adaptiveSamplingStep(Mesh* mesh2, int step)
{
	//same as adaptiveSampling() except this is bound to keyboard hit and does samping step by step as the key is hit

	bool orderedProcessing = true; //process samples from worst to best
	if (step >= (int) mesh2->samples.size())
	{
		if (step > (int) mesh2->samples.size())
			cout << "do nothing as all samples were processed\n";
		else
		{
			//copied from adaptiveSampling()
			mesh2->computeRadius();
#ifdef VERBOSE
			cout << nChanges << " out of " << (int) mesh2->samples.size() - (int) robustList.size()/2 << " nonrobust samples have been replaced\ndone!\n\n";
#endif
			if (orderedProcessing)
				delete [] vIdxsG;

			char fName[250]; sprintf(fName, "output\\%d-%das.dat", id, mesh2->id);
			resultToFile(fName, mesh2, true, false); //as: adaptive sampling
		}
		return -1;
	}
	if (radius < 0)
	{
		cout << "call adaptiveSampling() first w/ keyboardSampling=true\n";
		return -1;
	}
	
	if (orderedProcessing && step == 0)
	{		
		int n = (int) mesh2->samples.size();
		if (orderedProcessing)
		{
			float* disoVals = new float[n];
			vIdxsG = new int[n]; //same idea as in sortedIdxs
			for (int i = 0; i < n; i++)
			{
				disoVals[i] = mesh2->verts[ mesh2->samples[i] ]->diso;
				vIdxsG[i] = i;
			}
			insertionSort(disoVals, n, vIdxsG); //call by ref to disoVals (redundant) and vIdxsG (output to be used below)
			delete [] disoVals;
		}
	}
	
	//copied from adaptiveSampling()
	int ss = mesh2->samples[ (orderedProcessing ? vIdxsG[step] : step) ];
#ifdef VERBOSE
cout << step << "-" << ss << "\t\t" << mesh2->verts[ss]->diso << "\t" << mesh2->verts[ss]->dgrd << "\n";
#endif
	if (mesh2->verts[ss]->robust || mesh2->verts[ss]->matchIdx == -1) //outlier mesh2.sample will not be processed further 'cos it has no match
		return -1; //don't update robust samples in the robustList

	int bestPatchVert = -1;
#ifdef USE_CLOSEST_ROBUST_MATCH //affected even if i define this in adaptiveSampling()
	//replace ss w/ the vert in ss.patch which has the most compatible geodesic distance when compared with X = verts[ mesh2->verts[ss]->matchIdx ] to closestRobustIn1 geodesic distance
	int closestRobustIn1 = -1, closestRobustIn2 = -1;
	float X = INF;
	for (int i = 0; i < (int) robustList.size(); i += 2)
		if (verts[ robustList[i] ]->dSpanning[ mesh2->verts[ss]->matchIdx ] < X)
		{
			X = verts[ robustList[i] ]->dSpanning[ mesh2->verts[ss]->matchIdx ];
			closestRobustIn1 = robustList[i];
			closestRobustIn2 = robustList[i+1];
		}
if (closestRobustIn1 == -1) { cout << "no -1\n"; exit(0);}

	float minCost = INF; 
	for (int p = 0; p < (int) mesh2->verts[ss]->patch.size(); p++) //patch includes ss itself so ss may not be replaced at all (if it beats all other patch vertices)
	{
		if (mesh2->verts[ mesh2->verts[ss]->patch[p] ]->processed) //unavailable to be used as a potential final sample
			continue;

		//find geodesic distance of p to desired robustList mesh2.vert (closestRobustIn2)
		float Y = mesh2->verts[ closestRobustIn2 ]->dSpanning[ mesh2->verts[ss]->patch[p] ];
		if (fabs(X - Y) < minCost)
		{
			minCost = fabs(X - Y);
			bestPatchVert = mesh2->verts[ss]->patch[p];
		}
	}
#else
	int gvSize = (int) verts[ samples[0] ]->gv.size(); //gvSize'll be same (and equal to robustList.size/2) for all samples
if (gvSize == 0) cout << "\nWARNING: GVs are empty\n\n";
	//replace ss w/ the vert in ss.patch which has the most compatible gv w/ verts[ mesh2->verts[ss]->matchIdx ]->gv
	float minCost = INF;
	vector< float > pgv; //geodesic vector for a patch vertex
	for (int p = 0; p < (int) mesh2->verts[ss]->patch.size(); p++) //patch includes ss itself so ss may not be replaced at all (if it beats all other patch vertices)
	{
		if (mesh2->verts[ mesh2->verts[ss]->patch[p] ]->processed) //unavailable to be used as a potential final sample
			continue;

		//find geodesic vector of p to robustList mesh2.vertices
		pgv.clear();
		for (int i = 0; i < (int) robustList.size(); i += 2)
			pgv.push_back(mesh2->verts[ robustList[i+1] ]->dSpanning[ mesh2->verts[ss]->patch[p] ]);

		float c, cost = 0.0f, maxDiff = -INF;
		for (int i = 0; i < gvSize; i++)
		{
			c = fabs(verts[ mesh2->verts[ss]->matchIdx ]->gv[i] - pgv[i]);
			cost += c;
			if (c > maxDiff)
				maxDiff = c;
		}
		if (cost < minCost) //select best/min of average gv-pgv compatiblity
		{
			minCost = cost;
			bestPatchVert = mesh2->verts[ss]->patch[p];
		}//*/
/*		if (maxDiff < minCost) //select best/min of worst gvi-pgvi pair
		{
			minCost = maxDiff;
			bestPatchVert = mesh2->verts[ss]->patch[p];
		}//*/
	}
#endif
	if (bestPatchVert != ss && bestPatchVert != -1) //rarely bestPatchVert remains -1 since all the patch vertices were processed (made unavailable) by neighboring sample's dijkstra calls
	{
		nChanges++;
#ifdef VERBOSE
		cout << "old sample " << ss << " shifted " << mesh2->verts[ss]->dSpanning[bestPatchVert] 
			 << " unit to the new sample " << bestPatchVert << "\n";// -- " << closestRobustIn1 << endl;//cout << closestRobustIn1 << " - " << closestRobustIn2 << " w/ grdDistortion = " << verts[ closestRobustIn1 ]->dSpanning[ closestRobustIn2 ] << " was used\n\n";}
#endif
		//update samples[] matchIdx and sample variables accordingly
		mesh2->verts[ss]->sample = mesh2->verts[ss]->processed = false; //processed=false makes it available for being a final sample in upcoming iterations
		mesh2->verts[bestPatchVert]->sample = mesh2->verts[bestPatchVert]->processed = true; //just became a final sample; unavailable for upcoming iterations
		mesh2->verts[bestPatchVert]->matchIdx = mesh2->verts[ss]->matchIdx;
		mesh2->verts[ss]->matchIdx = -1;
		verts[ mesh2->verts[bestPatchVert]->matchIdx ]->matchIdx = bestPatchVert;
		mesh2->samples[ (orderedProcessing ? vIdxsG[step] : step) ] = bestPatchVert; //vIdxs[s] was holding ss but now it is replaced by bestPatchVert
		//also mark everyone sufficiently close to final sample (including bestPatchVert itself) as unavailable to preserve evenly-spaced sampling
		mesh2->dijkstraShortestPaths(bestPatchVert, mesh2->closeEnoughUnn);
		return bestPatchVert;
	}
	return -1;
}

void Mesh::adaptiveSampling2(Mesh* mesh2)
{
	//change mesh2.samples (nonrobust ones) to better match fixed mesh1.samples; do FPS multiple times with new starting points (preserves evenly-spacing perfectly)

#ifdef VERBOSE
	cout << "adaptive sampling on mesh2 to improve correspondences..\n";
#endif
	for (int i = 0; i < (int) robustList.size(); i += 2)
	{
		//clear previous effects
		int N = (int) mesh2->samples.size();
		for (int j = 0; j < N; j++)
			mesh2->verts[ mesh2->samples[j] ]->sample = false;
		mesh2->samples.clear();
		//create new samples starting from the next robust vertex robustList[i+1]
		mesh2->computeEvenlySpacedSamples(N, false, robustList[i+1]);
	}

	char fName[250]; sprintf(fName, "output\\%d-%das.dat", id, mesh2->id);
	resultToFile(fName, mesh2, true, false); //as: adaptive sampling
}

float totalFT = 0.0f, totalFTAdds = 0.0f, minFT = INF, maxFT = -INF, minConstituentFT = INF, maxConstituentFT = -INF,  //singletonTerm statistics from evalObjective()
	  totalRT = 0.0f, totalRTAdds = 0.0f, minRT = INF, maxRT = -INF, minConstituentRT = INF, maxConstituentRT = -INF, nf = 1.0f;  //pairwiseTerm statistics from evalObjective()
//#define COMMIT_AT_ONCE //define this to delay update to samples[] until all samples are processed; then commit/aaply all changes at once (just like in laplacian smoothing); undefine to update samples[] as soon as improved bestVa is found
#define PICK_FROM_1RING //define this to select the candidate from the 1-ring neighborhood of the current sample; undefine this to pick randomly from 1 patch of the sample; since samples change adaptively, i need the patch of all verts to apply this mode; use bfs for fast approx patches
#define GEO_INSTEAD_OF_EUC //define this for more accurate objective function based on geodesic distances; undefine for faster geodesic approximate: the euclidean distances                      ^not less accurate; actually MATCH_LOCAL_RADII sucks (avgLocalRadiusSucks.png); minLocalRadius (not avg) also sucks
//#define MATCH_LOCAL_RADII //define this to match mesh1.avgLocalRadius[i] vs. mesh2.avgLocalRadius[i] in the singletonTerm of the objective; undefine to match mesh1.radiusN - mesh2.radiusN globally (less^ accurate but simpler code, e.g. no maintenance on sampleNghbs[])
#define LOCAL_IMPROVEMENT //define this to check changes in objective function around the local neighborhood (fast); undefine to check the changes globally (always decrease guarantee)
void Mesh::adaptiveSampling3(Mesh* mesh2, bool removeOutliers)
{
	//applies cooridante descent, a derivation-free optimization algo, to minimize my adaptive sampling objective function composed of a pairwiseTerm and a singletonTerm

	int n = (int) mesh2->samples.size();
#ifdef VERBOSE
	cout << "adaptive sampling on mesh2 to improve correspondences (" << n << ")..\n";	
#endif
	//radius of both sides necessary in the singletonTerm (also in closeness definition in case i create patches/deeine localities)
	computeRadius();
	mesh2->computeRadius();

	float totalPatchSize = 0.0f;
#ifndef PICK_FROM_1RING //pick from patches mode; so i need patches
	//patches for the existing samples can be derived easily via existing dSpanningUnn[]
	for (int i = 0; i < n; i++)
	{
		mesh2->verts[ mesh2->samples[i] ]->patch.clear(); //necessary only if adaptiveSampling3() will be called 2+ times; hence redundant here
		for (int v = 0; v < (int) mesh2->verts.size(); v++)
			if (mesh2->verts[ mesh2->samples[i] ]->dSpanningUnn[ v ] <= mesh2->radius)
				mesh2->verts[ mesh2->samples[i] ]->patch.push_back(v); //mesh2->samples[i] itself is also in the patch
		totalPatchSize += (int) mesh2->verts[ mesh2->samples[i] ]->patch.size();
	}
#ifdef VERBOSE
	cout << "avg patch size = " << totalPatchSize / n << " vertices\n";
#endif
	//patches for the potential samples can be derived via O(V) BFS (instead of O(VlgV) dijkstra); note that BFS/Dijkstra will be made V times for each potential; so talking about O(V^2) and O(V^2lgV) respectively
	totalPatchSize = 0.0f;
	int nAdds = 0;
	for (int i = 0; i < (int) mesh2->verts.size(); i++)
	{
		if (mesh2->verts[ i ]->sample)
			continue; //samples already know their patch[]

		//for (int v = 0; v < (int) mesh2->verts.size(); v++) mesh2->verts[v]->dBFS = INF; //clear effect(s) of previous call(s)
		//avoid verts traversal just-above for efficiency by remembering the visited verts (in patch[]) to undo their dBFS values at the end of this function
		queue< int > q;
		q.push(i);
		mesh2->verts[i]->dBFS = 0.0f; //length of unweighted shortest path from source to source is 0
		//mesh2->verts[i]->patch.push_back(i); don't put itself to the patch 'cos i don't want to pick itself as sample candidate below
		while (! q.empty())
		{
			int v = q.front();
			q.pop();

			//look at each adjacent vertices of v
			bool reachedNonclose = false; //true if i reach a distant vertex, in which case i early-stop BFS
			set< int, std::less< int > >::const_iterator neighbVertIdx;
			for (neighbVertIdx = mesh2->verts[v]->vertNeighborsSet.begin(); neighbVertIdx != mesh2->verts[v]->vertNeighborsSet.end(); neighbVertIdx++)
			{
				int va = (*neighbVertIdx);
				if (mesh2->verts[va]->dBFS == INF)
				{
					mesh2->verts[va]->dBFS = mesh2->verts[v]->dBFS + 1.0f; //update d of this adjacent
					q.push(va); //va.parent = v; could also be done here but not using parents in my applicaiton
					mesh2->verts[i]->patch.push_back(va); //visited.push_back(va); //all verts whose dBFS updated are stored
					float dist = //distanceBetween(mesh2->verts[i]->coords, mesh2->verts[va]->coords); //just-below is a better approximation
								 mesh2->verts[va]->dBFS * mesh2->avgEdgeLen; //BFS finds the shortest path between two nodes i and va, with path length measured by number of edges; so multiply it w/ avgEdgeLen; assuming uniform edge lengths on the mesh
					if (dist > mesh2->radius)
						reachedNonclose = true;
				}
			}
			if (reachedNonclose)
			{
				//undo dBFS values and return; this is not done in the classical BFS but in my version i cancelled verts[v]->dBFS = INF; above so this is required for the upcoming i
				for (int u = 0; u < (int) mesh2->verts[i]->patch.size(); u++)
					mesh2->verts[ mesh2->verts[i]->patch[u] ]->dBFS = INF; //no need 'cos i reset dBFS = INF via verts traversal above
				mesh2->verts[ i ]->dBFS = INF; //since i itself not in patch, set it to INF here explicitly
				break; //from while-loop
			}
		} //end of while
		totalPatchSize += (int) mesh2->verts[ i ]->patch.size();
		nAdds++;		
	}
#ifdef VERBOSE
	cout << "avg bfs-based patch size = " << totalPatchSize / nAdds << " vertices\n";
#endif
#else
#ifdef VERBOSE
	cout << "PICK_FROM_1RING mode active\n";
#endif
#endif
#ifdef VERBOSE
#ifdef COMMIT_AT_ONCE
	cout << "COMMIT_AT_ONCE mode active\n";
#endif
#endif

#ifndef GEO_INSTEAD_OF_EUC
#ifdef VERBOSE
	cout << "max euclidean distance being computed..\n";
#endif
	//max euclidean distance over mesh2 will also be needed in case i use approx/fast objective function below
	mesh2->maxEucDist = -INF;
	for (int u = 0; u < (int) mesh2->verts.size(); u++) //nested-loop over FPS samples should also be fine but this is done once so use whole vertices to be perfectly exact
		for (int v = 0; v < (int) mesh2->verts.size(); v++)
		{
			float dist = distanceBetween(mesh2->verts[u]->coords, mesh2->verts[v]->coords);
			if (dist > mesh2->maxEucDist)
				mesh2->maxEucDist = dist;
		}
#ifdef VERBOSE
	cout << "max euclidean distance on mesh2: " << mesh2->maxEucDist << endl;
#endif
#endif
#ifdef MATCH_LOCAL_RADII
	avgLocalRadius = new float[n]; //i'th avg radius to be used in singletonTerm
	mesh2->avgLocalRadius = new float[n]; //i'th avg radius to be used in singletonTerm
	float avgNghbhoodSize = 0.0f, avgNghbhoodSize2 = 0.0f;
	for (int i = 0; i < n; i++)
	{
		int sample = mesh2->samples[i];
		int si2 = mesh2->computeSampleNghbs(sample, i), si = computeSampleNghbs(mesh2->verts[sample]->matchIdx, i);

		avgNghbhoodSize += si;
		avgNghbhoodSize2 += si2;

#ifdef VERBOSE
		if (si < 3 || si > 10 || si2 < 3 || si2 > 10)
		{
			cout << "WARNING: bad # of sample neighbors: " << si << " & " << si2 << " for " << mesh2->verts[sample]->matchIdx << " & " << sample << endl;
//			exit(0);
		}
		float sanity = 2.0f; //>1
		if (avgLocalRadius[i] > radiusN*sanity || avgLocalRadius[i] < radiusN/sanity || mesh2->avgLocalRadius[i] > mesh2->radiusN*sanity || mesh2->avgLocalRadius[i] < mesh2->radiusN/sanity)
			cout << "WARNING: bad local radius: " << avgLocalRadius[i] << " & " << mesh2->avgLocalRadius[i] << endl;
//if (i == 2) cout << si << " " << si2 << "\t" << samples[i]  << " " << mesh2->samples[i] << "\n\n";
#endif
	}
#ifdef VERBOSE
	cout << "avg neighborhood sizes in mesh1 & mesh2 = " << avgNghbhoodSize/n << " & " << avgNghbhoodSize2/n << "\n";
#endif
//for (int i = 0; i < verts[ samples[2] ]->sampleNghbs.size(); i++) cout << verts[ samples[2] ]->sampleNghbs[i] << "\t";cout<<endl;
//for (int i = 0; i < mesh2->verts[ mesh2->samples[2] ]->sampleNghbs.size(); i++) cout << mesh2->verts[ mesh2->samples[2] ]->sampleNghbs[i] << "\t";cout<<endl;
//cout << avgLocalRadius[2] << " " << mesh2->avgLocalRadius[2] << "\n";
	//rearrrange avgLocalRadius[] so that avgLocalRadius[i] keeps the avg local radius of the following mesh1 vertex: mesh2.samples[i].matchIdx
	for (int i = 0; i < n; i++)
	{
		int j;
		for (j = 0; j < n; j++)
			if (samples[i] == mesh2->verts[ mesh2->samples[j] ]->matchIdx)
				break;
		//swap avgLocalRadius[j] and avgLocalRadius[i], which will finalize the location of avgLocalRadius[j]
		float tmp = avgLocalRadius[j];
		avgLocalRadius[j] = avgLocalRadius[i];
		avgLocalRadius[i] = tmp;
	}
//cout << avgLocalRadius[2] << " " << mesh2->avgLocalRadius[2] << "\n\n";
	//sanitiy check for (int i = 0; i < n; i++) cout << avgLocalRadius[i] << "\t\t" << mesh2->avgLocalRadius[i] << "\t\t" << avgLocalRadius[i] - mesh2->avgLocalRadius[i] << endl;exit(0);
#endif

	float objVal, minObjVal = INF;
	int nImproveds, nIters = 0;//count=0;
	vector< int > bestSample2;
//FILE* fPtrObj = fopen("as-obj.dat", "w"), * fPtrIso = fopen("as-iso.dat", "w"), * fPtrGrd = fopen("as-grd.dat", "w"); //for plots in paper; 1 line per iteration
do
{
	nImproveds = 0;
	
	float alpha = 0.5f;//0.5f;//1.0f;//0.5f;//0.0f; //1.0f //objective function = (1-alpha)*pairwiseTerm + alpha*singletonTerm
	//alpha=0.0f objective always decreases :) alpha=1.0f increases-decreases (inf loop even occurs sometimes) so minimizing mesh1.radiusN - mesh2.radiusN difference is unstable (also affects good pairwiseTerm when alpha > 0 in that alpha=0.25 not always decrease);
	//solution1: avoid local improvement makes global improvement assuumption and eval whole objective for each candidate (instead of evalObjectiveOnVert); less efficient; i dont want to lose local impr --> global impr assumption [not in use]
	//solution2: mesh1.radiusN - mesh2.radiusN is too global and unpredictable after local improvements; minimize mesh1.avgLocalRadius[i] - mesh2.avgLocalRadius[i] difference instead [in use]
	//solution1 comment applies when COMMIT_AT_ONCE = true; if you apply changes immediately (COMMIT_AT_ONCE=false) then local improvement helps global improvement and it always decreases

	objVal = evalObjective(mesh2, alpha, nIters == 0);//redundant if using currentObjVal below; may still be useful to show that the overall objective is decreasing; besides it'll auto-compute the normalization factor nf in the 1st iteration

//float iso3, grd3, objVal3 = evalObjective3(mesh2, alpha, iso3, grd3);fprintf(fPtrObj, "%f\n", objVal3); fprintf(fPtrIso, "%f\n", iso3); fprintf(fPtrGrd, "%f\n", grd3);

	if (nIters == 0)
	{
#ifdef VERBOSE
		cout << "initial objective function value: " << objVal << " (alpha = " << alpha << ")\n";
//cout << "\navg pairwiseTerm eo = " << totalRT / totalRTAdds << "\t" << minRT << " ++ " << maxRT << "\tconstituent: " << minConstituentRT << " ** " << maxConstituentRT << endl
//    << "avg singletonTerm eo = " << totalFT / totalFTAdds << "\t" << minFT << " ++ " << maxFT << "\tconstituent: " << minConstituentFT << " ** " << maxConstituentFT << "\n\n";
#endif
	}
	//int improved1 = -1, improved2 = -1, improvedS = -1; //show the last improvement (from improved1 to 2)
	for (int s = 0; s < n; s++)
	{
		int ns = mesh2->samples[s]; //ns for next sample

		//if (mesh2->verts[ns]->matchIdx == -1)  continue; //outlier mesh2.sample will not be processed further 'cos it has no match [not needed 'cos samples[] not have any outlier in it for sure; all matchIdx-related if conditions removed below and from evalObjective()]
//		if (robustListUnaffected && mesh2->verts[ns]->robust) //robust samples are fixed and cannot move if we are in robustListUnaffected mode
//			continue; //don't update robust samples; no privilege for robusts 'cos then (nAddsR + nAddsCR) denominator here would not match w/ nAddsR in evalObjective(); dont complicate things and disable this

#ifdef LOCAL_IMPROVEMENT
		float currentObjVal = evalObjectiveOnVert(mesh2, alpha, ns, s); //an improvement on ns-related terms implies an improvement on the overall objective; so save time and focus on ns-related improvement here
		int bestVa = -1;
#ifdef PICK_FROM_1RING
		//check 1-ring neighborhood of ns to find an improvement in objective function; if found go to that direction/neighbor (that is, replace ns w/ va so that va is the new/adapted/better sample now)
		set< int, std::less< int > >::const_iterator neighbVertIdx;		
		for (neighbVertIdx = mesh2->verts[ns]->vertNeighborsSet.begin(); neighbVertIdx != mesh2->verts[ns]->vertNeighborsSet.end(); neighbVertIdx++)
		{
			int va = (*neighbVertIdx);
#else //pick from the ns.patch the next candidate

		for (int c = 0; c < 6; c++) //neighbVertIdx above will also iterate ~avgVertDegree times, which is 6 in a closed 2-manifold
		{
			int va = mesh2->verts[ns]->patch[ rand() % (int) mesh2->verts[ns]->patch.size() ]; //copied randoi(int b) from GA.cpp where b = (int) mesh2->verts[ns]->patch.size()
#endif
			if (mesh2->verts[va]->sample)
				continue; //if this candidate becomes the winner bestVa, then 2+ mesh1.sample would be using this va as the match, which would cause logical errors in runtime; be simple/safe and prevent an existing/active sample from being selected as a new/improved sample

			//objective function using this candidate (to be compared w/ objective function using the generator ns, namely the currentObjVal)			
			/////////////// singletonTerm //////////////////////
			float candidSingletonTerm = 0.0f;//INF;//0.0f; use 0.0 if averaging, inf if minimizing
#ifdef MATCH_LOCAL_RADII
			//compute current avgLocalRadius (candidSingletonTerm) based on the ns.sampleNghbs[] and candidate va; adapted from computeAvgLocalRadius()			
			for (int j = 0; j < (int) mesh2->verts[ns]->sampleNghbs.size(); j++)
				candidSingletonTerm +=
#ifdef GEO_INSTEAD_OF_EUC
					mesh2->verts[ mesh2->verts[ns]->sampleNghbs[j] ]->dSpanning[va];
#else
					distanceBetween(mesh2->verts[ mesh2->verts[ns]->sampleNghbs[j] ]->coords, mesh2->verts[va]->coords) / maxEucDist;
#endif
/*#ifdef GEO_INSTEAD_OF_EUC
					if (mesh2->verts[ mesh2->verts[ns]->sampleNghbs[j] ]->dSpanning[va] < candidSingletonTerm)
						candidSingletonTerm = mesh2->verts[ mesh2->verts[ns]->sampleNghbs[j] ]->dSpanning[va];
#else
					if (distanceBetween(mesh2->verts[ mesh2->verts[ns]->sampleNghbs[j] ]->coords, mesh2->verts[va]->coords) / maxEucDist < candidSingletonTerm)
						candidSingletonTerm = distanceBetween(mesh2->verts[ mesh2->verts[ns]->sampleNghbs[j] ]->coords, mesh2->verts[va]->coords) / maxEucDist;
#endif*/
			candidSingletonTerm /= ((int) mesh2->verts[ns]->sampleNghbs.size());
			candidSingletonTerm = fabs(avgLocalRadius[s] - candidSingletonTerm); //assigned value is in [0,1] as avgLocalRadius[s] is normalized; in practice in [0, 0.05] as radius is just a small distance over the surface (as # samples increases radius decreases)
#else
			mesh2->computeRadiusBasedOn(va, s); //mesh2.radiusN & mesh2.minRadiusN updated and ready
			candidSingletonTerm = fabs(radiusN - mesh2->radiusN); //in [0,1] as radiusN is normalized; in practice in [0, 0.05] as radius is just a small distance over the surface (as # samples increases radius decreases)
#endif
			/////////////// singletonTerm ends //////////////////////
			///////////////// pairwiseTerm ///////////////////////////
			float candidPairwiseTerm = 0.0f;
			for (int s2 = 0; s2 < n; s2++)
			{
				int ns2 = mesh2->samples[s2]; //next sample different from ns
				if (s2 == s) //s2==s is to stop ns=ns2; no symmetric stuff here 'cos ns-related va must interact w/ all samples other than ns
					continue;

				float dist = 
#ifdef GEO_INSTEAD_OF_EUC
					fabs(mesh2->verts[ns2]->dSpanning[va] - verts[ mesh2->verts[ns2]->matchIdx ]->dSpanning[ mesh2->verts[ns]->matchIdx ]);
					//fabs(mesh2->verts[ns2]->dSpanning[va] - avgLocalRadius[s]);
					//-mesh2->verts[ns2]->dSpanning[va]; //ns2.dSpanning[] ready for sure 'cos ns2 is an existing sample whose dSpanning[] are either known before this function or learnt in a previous iter's mesh2->dijkstraShortestPaths(mesh2->samples[s]); call below
					//mesh2->verts[ns2]->dSpanning[va]; //minimizing this would collect all samples in center (assuming no feature term and no LOCAL_REPULSION)
#else
					fabs((distanceBetween(mesh2->verts[va]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist) - (distanceBetween(verts[ mesh2->verts[ns]->matchIdx ]->coords, verts[ mesh2->verts[ns2]->matchIdx ]->coords) / mesh2->maxEucDist)); //mesh1.maxEucDist undefined so ~wrong/close denom after minus
					//fabs((distanceBetween(mesh2->verts[va]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist) - avgLocalRadius[s]); //avgLocalRadius[] is based on geodesics but euc vs. goe should not be a problem within a small patch
					//-distanceBetween(mesh2->verts[va]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist;
					//distanceBetween(mesh2->verts[va]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist; //minimizing this would collect all samples in center (assuming no feature term)
#endif
						
				candidPairwiseTerm += dist; //n-1 adds in total will be normalized below
			}
			///////////////// pairwiseTerm ends ///////////////////////////

			float candidObjVal = ((1.0f - alpha) * candidPairwiseTerm / (n-1)) + (alpha * candidSingletonTerm * nf); //denominator n-1 puts the candidPairwiseTerm in [0,1] (when avgLocalRadius[] not in use); assume candidSingletonTerm is in [0.01, 0.11] with mean=0.03
			//denominator n-1 puts the candidPairwiseTerm in [-1,0] (-ve dists); repulsion.mean = -0.36; assume candidSingletonTerm is in [0.01, 0.11] with mean=0.03; mean ratio ~ 10 is factored in candidSingletonTerm via normalization factor nf (nf is auto-computed)

//if (s==3) cout << mesh2->verts[ns]->matchIdx << " - " << ns << "\t" << candidObjVal << " vs. " << currentObjVal << "\t" << currentObjVal - candidObjVal << endl;

			if (currentObjVal - candidObjVal > 1e-5) //there is a significant improvement :)
			{
				bestVa = va;
				//new and improved objVal
				currentObjVal = candidObjVal;
//if (s==3) cout << "(" << ns << " will be improved to " << bestVa << ")\n";
			}
		} //end of neighbVertIdx
#else
		float currentObjVal = evalObjective(mesh2, alpha); //a global improvement will be checked so remember the initial global objective value
		int bestVa = -1;
#ifdef PICK_FROM_1RING
		//check 1-ring neighborhood of ns to find an improvement in objective function; if found go to that direction/neighbor (that is, replace ns w/ va so that va is the new/adapted/better sample now)
		set< int, std::less< int > >::const_iterator neighbVertIdx;		
		for (neighbVertIdx = mesh2->verts[ns]->vertNeighborsSet.begin(); neighbVertIdx != mesh2->verts[ns]->vertNeighborsSet.end(); neighbVertIdx++)
		{
			int va = (*neighbVertIdx);
#else //pick from the ns.patch the next candidate

		for (int c = 0; c < 6; c++) //neighbVertIdx above will also iterate ~avgVertDegree times, which is 6 in a closed 2-manifold
		{
			int va = mesh2->verts[ns]->patch[ rand() % (int) mesh2->verts[ns]->patch.size() ]; //copied randoi(int b) from GA.cpp where b = (int) mesh2->verts[ns]->patch.size()
#endif
			if (mesh2->verts[va]->sample)
				continue; //if this candidate becomes the winner bestVa, then 2+ mesh1.sample would be using this va as the match, which would cause logical errors in runtime; be simple/safe and prevent an existing/active sample from being selected as a new/improved sample

			//objective function using this candidate (to be compared w/ objective function using the generator ns, namely the currentObjVal)			
			float candidObjVal = evalObjectiveWithCandidate(mesh2, alpha, va, s, mesh2->verts[ns]->matchIdx);
			if (currentObjVal - candidObjVal > 1e-5) //there is a significant improvement :)
			{
				bestVa = va;
				//new and improved objVal
				currentObjVal = candidObjVal;
//if (s==3) cout << "(" << ns << " will be improved to " << bestVa << ")\n";
			}
		} //end of neighbVertIdx
#endif
		mesh2->verts[ns]->newPosition = ns; //in case it doesn't change just-below (no improvement) set newPosition to itself
		if (bestVa != -1) //there is an improvement :)
		{
			//improved = true;
			nImproveds++;
#ifdef COMMIT_AT_ONCE
			mesh2->verts[ns]->newPosition = bestVa; //commit this update after outer s-loop finishes; so upcoming iterations of this s-loop use nonupdated samples[], just like in laplacian smoothing
			mesh2->verts[ns]->sample = false; mesh2->verts[bestVa]->sample = true; //not committing the update to samples[] but i must update the sample variable to prevent same sample land on top of each other during commit below
#else
			//update samples[] as soon as improved bestVa is found
			//replace ns w/ bestVa, which is a better/new/adapted sample
			mesh2->samples[s] = bestVa;
			//updated the related fields from the ns values to new bestVa values
			mesh2->verts[bestVa]->matchIdx = mesh2->verts[ns]->matchIdx;
			//mesh2->verts[ns]->matchIdx = -1; don't do this due to `bestVa lands ..' problem explained below: ns index may be pointing to a (innocent) vertex that is still active; making ns.matchIdx=-1 will also affect it and code will crash 'cos active vert will use its matchIdx as an index to verts[]
			verts[ mesh2->verts[bestVa]->matchIdx ]->matchIdx = bestVa;
if (mesh2->verts[bestVa]->sample) exit(0); //cout << "bestVa lands on an existing sample other than the current ns; matchIdx value may be problematic now\n"; this cout cannot happen now thanks to the if(sample)continue in bestVa selection
			//[old] if bestVa landed on an existing sample in the prev do-while iteration, current ns has 2+ samples (current ns really must be deleted but others are innoncent); deleteing ns.pairwiseTermTo[] will crash one of the remaining active ns's later; so dont do it
			//[old] by the same reasoning, ns.sample and ns.matchIdx values not so reliable either; ns.sample not in use but ns.matchIdx is; don't worry 'cos this bestVa landing should not occur w/ a good energy function; if occurs, then you may keep a reference counter on ns
			//[new] thanks to the if(sample)continue in bestVa selection, bestVa cannot land on an existing sample now sp enable memo-recapture; sample and matchIdx values are also fully safe now
			mesh2->verts[ns]->sample = false;
			mesh2->verts[bestVa]->sample = true;
#ifdef GEO_INSTEAD_OF_EUC
			mesh2->dijkstraShortestPaths(mesh2->samples[s]); //returns immediately for samples which already know their dSpanningUnn[]
#endif
#ifdef MATCH_LOCAL_RADII
			//transfer ns.sampleNeighbs to this new sample
			mesh2->verts[bestVa]->sampleNghbs = mesh2->verts[ns]->sampleNghbs;
			//update sampleNeighbs of other samples for the entry that involves ns (since ns is not a sample anymore, replace it w/ bestVa)
			for (int i = 0; i < n; i++)
				if (i != s)// && mesh2->verts[ns]->dSpanning[ mesh2->samples[i] ] < 3.0f * mesh2->radius) //for efficiency don't let the ns (u=s) and the distant samples do the search below
					for (int j = 0; j < (int) mesh2->verts[ mesh2->samples[i] ]->sampleNghbs.size(); j++)
						if (mesh2->verts[ mesh2->samples[i] ]->sampleNghbs[j] == ns)
						{
							mesh2->verts[ mesh2->samples[i] ]->sampleNghbs[j] = bestVa;
							break;
						}

#endif
#endif
		}
//cout << s << " " << ns << " " << bestVa << endl;

//if(count++ == 0)mesh2->mrIdx = ns; if(bestVa!=-1)mesh2->mrIdx2 = bestVa; break;
	} //end of s
//if (improved) cout << endl;//"at least one sample provided an improvement on objective value\n\n";

#ifdef COMMIT_AT_ONCE
	//commit changes at once here, i.e. after the s-loop just-above finishes; sample.newPosition'll be used to apply these changes
	for (int s = 0; s < n; s++)
	{
		int ns = mesh2->samples[s]; //ns for next sample

		//replace ns w/ bestVa (remembered by ns.newPosition), which is a better/new/adapted sample; note that ns.newPosition = ns happens when ns is not improved
		int newPosition = mesh2->verts[ns]->newPosition;
		if (newPosition == ns)
			continue; //no improvement on this s'th sample so do nothing
		mesh2->samples[s] = newPosition;
		//updated the related fields from the ns values to new bestVa values
		mesh2->verts[newPosition]->matchIdx = mesh2->verts[ns]->matchIdx;
		//mesh2->verts[ns]->matchIdx = -1; don't do it (see `bestVa lands ..' problem above)
		verts[ mesh2->verts[newPosition]->matchIdx ]->matchIdx = newPosition;
#ifdef GEO_INSTEAD_OF_EUC
		mesh2->dijkstraShortestPaths(mesh2->samples[s]); //returns immediately for samples which already know their dSpanningUnn[]
#endif
#ifdef MATCH_LOCAL_RADII
			//transfer ns.sampleNeighbs to this new sample
			mesh2->verts[newPosition]->sampleNghbs = mesh2->verts[ns]->sampleNghbs;
			//update sampleNeighbs of other samples for the entry that involves ns (since ns is not a sample anymore, replace it w/ newPosition)
			for (int i = 0; i < n; i++)
				if (i != s)// && mesh2->verts[ns]->dSpanning[ mesh2->samples[i] ] < 3.0f * mesh2->radius) //for efficiency don't let the ns (i=s) and the distant samples do the little search below
					for (int j = 0; j < (int) mesh2->verts[ mesh2->samples[i] ]->sampleNghbs.size(); j++)
						if (mesh2->verts[ mesh2->samples[i] ]->sampleNghbs[j] == ns)
						{
							mesh2->verts[ mesh2->samples[i] ]->sampleNghbs[j] = newPosition;
							break;
						}

#endif
//if (mesh2->verts[newPosition]->sample) cout << "newPosition lands on an existing sample other than the current ns; matchIdx value may be problematic now\n"; //this always prints (no problem) 'cos i set sample=true above to prevent sample collisions
		//[old] if bestVa landed on an existing sample in the prev do-while iteration, current ns has 2+ samples (current ns really must be deleted but others are innoncent); deleteing ns.pairwiseTermTo[] will crash one of the remaining active ns's later; so dont do it
		//[old] by the same reasoning, ns.sample and ns.matchIdx values not so reliable either; ns.sample not in use but ns.matchIdx is; don't worry 'cos this bestVa landing should not occur w/ a good energy function; if occurs, then you may keep a reference counter on ns
		//[new] thanks to the if(sample)continue in bestVa selection, bestVa cannot land on an existing sample now sp enable memo-recapture; sample and matchIdx values are also fully safe now
//		mesh2->verts[ns]->sample = false; mesh2->verts[newPosition]->sample = true; //these 2 statements are redundat 'cos already done above
	} //end of s
#endif
	
	float newObj = evalObjective(mesh2, alpha);
#ifdef VERBOSE
	cout << nImproveds << " improvements at iter # " << nIters << "\n";//\tobjective: " << objVal << " --> " << newObj << (nImproveds > 0 && newObj >= objVal ? "\tWARNING: not decreased :(\n" : "\n");
#endif
	if (newObj < minObjVal)
	{
		minObjVal = newObj;
		bestSample2 = mesh2->samples;
	}

//if (nIters == 1) cout << "\navg pairwiseTerm eo = " << totalRT / totalRTAdds << "\t" << minRT << " ++ " << maxRT << "\tconstituent: " << minConstituentRT << " ** " << maxConstituentRT << endl
//				      << "avg singletonTerm eo = " << totalFT / totalFTAdds << "\t" << minFT << " ++ " << maxFT << "\tconstituent: " << minConstituentFT << " ** " << maxConstituentFT << "\n\n";
//if(nIters==4)break;
//break;
	if ((int)verts.size()>150000&&nIters==7)break;//memo leakage here causes occasional crashes on large meshes so don't do more than 7 iterations
} while (nImproveds > 0 && nIters++ < 200); //2nd condition never happens :)
//fclose(fPtrObj);fclose(fPtrIso);fclose(fPtrGrd);

//for (int i = 0; i < verts[ samples[2] ]->sampleNghbs.size(); i++) cout << verts[ samples[2] ]->sampleNghbs[i] << "\t";cout<<endl;
//for (int i = 0; i < mesh2->verts[ mesh2->samples[2] ]->sampleNghbs.size(); i++) cout << mesh2->verts[ mesh2->samples[2] ]->sampleNghbs[i] << "\t";cout<<endl;
//cout << avgLocalRadius[2] << " " << mesh2->avgLocalRadius[2] << "\n\n";

//cout << "\navg pairwiseTerm eo = " << totalRT / totalRTAdds << "\t" << minRT << " ++ " << maxRT << "\tconstituent: " << minConstituentRT << " ** " << maxConstituentRT << endl
//	 << "avg singletonTerm eo = " << totalFT / totalFTAdds << "\t" << minFT << " ++ " << maxFT << "\tconstituent: " << minConstituentFT << " ** " << maxConstituentFT << "\n\n";

/*	//in case the objective not always decreasing, i might have an objective value that is smaller than the current min objective value; roll-back to that actual min objective value
	for (int r = 0; r < n; r++)
	{
		int old = mesh2->samples[r], s1 = samples[r], s2 = bestSample2[r]; //s1 from mesh1 in correspondence w/ s2 from mesh2
		mesh2->samples[r] = s2;
		mesh2->verts[s2]->matchIdx = s1;
		verts[s1]->matchIdx = s2;
		mesh2->verts[old]->sample = false;
		mesh2->verts[s2]->sample = true;
	}
	cout << "rolled-back obj = " << evalObjective(mesh2, 0.5f) << endl;//*/

	//quite probably new samples added which need their geodesics for distortion computations in resultToFile()
	for (int s = 0; s < n; s++)
		mesh2->dijkstraShortestPaths(mesh2->samples[s]); //returns immediately for samples which already know their dSpanningUnn[]; in GEO_INSTEAD_OF_EUC mode these geodesics are already known by now so returns immediately

	mesh2->computeRadius();

#ifndef VERBOSE
	cout << "after adaptiveSampling: "; //resultToFile()'ll print the rest of this line
#endif
	char fName[250]; sprintf(fName, "output\\%d-%d ga+as.dat", id, mesh2->id);
	resultToFile(fName, mesh2, removeOutliers, false); //as: adaptive sampling
}
float Mesh::evalObjective(Mesh* mesh2, float alpha, bool firstCall)
{
	//evaluates the objective function of adaptive sampling

	float pairwiseTerm = 0.0f, singletonTerm = 0.0f; //objective function = (1-alpha)*pairwiseTerm + alpha*singletonTerm
	int n = (int) mesh2->samples.size(), nAddsR = 0;
//minConstituentRT = minConstituentFT = INF; maxConstituentRT = maxConstituentFT = -INF;
	for (int s = 0; s < n; s++)
	{
		int ns = mesh2->samples[s]; //ns for next sample

		///////////////// pairwiseTerm ///////////////////////////
		for (int s2 = 0; s2 < n; s2++)
		{
			int ns2 = mesh2->samples[s2]; //next sample different from ns
			if (s2 <= s) //s2 <= s is to stop ns=ns2 (s=s2) and s2<s case from computing symmetric stuff below (symmetric 'cos s2>s already computed those distances)
				continue;

			float dist = 
#ifdef GEO_INSTEAD_OF_EUC
				fabs(mesh2->verts[ns]->dSpanning[ns2] - verts[ mesh2->verts[ns]->matchIdx ]->dSpanning[ mesh2->verts[ns2]->matchIdx ]);
				//fabs(mesh2->verts[ns]->dSpanning[ns2] - mesh2->avgLocalRadius[s]);
				//-mesh2->verts[ns]->dSpanning[ns2]; //ns/ns2.dSpanning[] ready for sure 'cos ns/ns2 is an existing sample 
				//mesh2->verts[ns]->dSpanning[ns2]; //minimizing this would collect all samples in center (assuming no feature term and no LOCAL_REPULSION)				
#else
				fabs((distanceBetween(mesh2->verts[ns]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist) - (distanceBetween(verts[mesh2->verts[ns]->matchIdx]->coords, verts[ mesh2->verts[ns2]->matchIdx ]->coords) / mesh2->maxEucDist)); //mesh1.maxEucDist undefined so ~wrong/close denom after minus
				//fabs((distanceBetween(mesh2->verts[ns]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist) - mesh2->avgLocalRadius[s]);
				//-distanceBetween(mesh2->verts[ns]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist;
				//distanceBetween(mesh2->verts[ns]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist; //minimizing this would collect all samples in center (assuming no feature term)
#endif
			
			pairwiseTerm += dist; //dist is already in [0,1]
			nAddsR++; //nAddsR addsto pairwiseTerm in total will be normalized below

//if (dist < minConstituentRT) minConstituentRT = dist;if (dist > maxConstituentRT) maxConstituentRT = dist;
		}
		///////////////// pairwiseTerm ends ///////////////////////////

		/////////////// singletonTerm //////////////////////
#ifdef MATCH_LOCAL_RADII
		mesh2->computeAvgLocalRadius(ns, s); //based on the ns.sampleNghbs[]
		singletonTerm += fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]); //added value is in [0,1] as avgLocalRadius[s] is normalized; in practice in [0, 0.05] as radius is just a small distance over the surface (as # samples increases radius decreases)
		//singletonTerm = fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]); //assigned value is in [0,1] as avgLocalRadius[s] is normalized; in practice in [0, 0.05] as radius is just a small distance over the surface (as # samples increases radius decreases)
//if (fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]) < minConstituentFT) minConstituentFT = fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]);if (fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]) > maxConstituentFT) maxConstituentFT = fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]);
#endif
		/////////////// singletonTerm ends //////////////////////
	}

	/////////////// singletonTerm //////////////////////
#ifdef MATCH_LOCAL_RADII
	singletonTerm /= n; //n additions in total to singletonTerm (not averaging any more so invalid comment)
#else
	mesh2->computeRadius(false); //mesh2.radiusN/minRadiusN updated
	//compare radiusN (maby later minRadiusN too) w/ the fixed radius in mesh1 (that i want to coverge to)
	singletonTerm = fabs(radiusN - mesh2->radiusN); //in [0,1] as radiusN is normalizedl in practice in [0, 0.05] as radius is just a small distance over the surface (as # samples increases radius decreases)
#endif
	/////////////// singletonTerm ends //////////////////////
	pairwiseTerm /= nAddsR; //denominator puts the pairwiseTerm  in [0,1] (when avgLocalRadius[] not in use), and assume the singletonTerm is already in [0.01, 0.11] with mean=0.03
						    //denominator puts the pairwiseTerm in [-1,0] (-ve dists); pairwiseTerm.mean = -0.36, and assume the singletonTerm is already in [0.01, 0.11] with mean=0.03; mean ratio ~ 10 is factored in singletonTerm via normalization factor nf (nf is auto-computed)

//totalFT += (singletonTerm);totalFTAdds++;if((singletonTerm)<minFT)minFT=(singletonTerm); if(singletonTerm>maxFT)maxFT=singletonTerm;totalRT += (pairwiseTerm);totalRTAdds++;if((pairwiseTerm) < minRT)minRT=pairwiseTerm; if(pairwiseTerm> maxRT)maxRT=pairwiseTerm;
	if (firstCall) //in the first call totalFTAdds = totalRTAdds = 1; ????????????????????????????should i do it at all calls????????????????????????????
		nf = fabs(pairwiseTerm) / fabs(singletonTerm); //ratio of average pairwiseTerm and average singletonTerm; this ratio will be factored into singletonTerm to obtain meaningful contributions of the two terms to the objective function
//cout<<nf;exit(0);

	return ((1.0f - alpha) * pairwiseTerm) + (alpha * singletonTerm * nf);
}
float Mesh::evalObjectiveWithCandidate(Mesh* mesh2, float alpha, int candidate, int sidx, int midx)
{
	//same as evalObjective() except this one uses candidate instead of the sidx'th sample; candidate.matchIdx is unknown so use midx instead

#ifdef GEO_INSTEAD_OF_EUC
	mesh2->dijkstraShortestPaths(candidate); //candidate.dSpanning[] needed below
#endif
	float pairwiseTerm = 0.0f, singletonTerm = 0.0f; //objective function = (1-alpha)*pairwiseTerm + alpha*singletonTerm
	int n = (int) mesh2->samples.size(), nAddsR = 0;
//minConstituentRT = minConstituentFT = INF; maxConstituentRT = maxConstituentFT = -INF;
	for (int s = 0; s < n; s++)
	{
		int ns = (s != sidx ? mesh2->samples[s] : candidate); //ns for next sample

		///////////////// pairwiseTerm ///////////////////////////
		for (int s2 = 0; s2 < n; s2++)
		{
			int ns2 = (s2 != sidx ? mesh2->samples[s2] : candidate); //next sample different from ns
			if (s2 <= s) //s2 <= s is to stop ns=ns2 (s=s2) and s2<s case from computing symmetric stuff below (symmetric 'cos s2>s already computed those distances)
				continue;

			float dist = 
#ifdef GEO_INSTEAD_OF_EUC
				fabs(mesh2->verts[ns]->dSpanning[ns2] - verts[ (s != sidx ? mesh2->verts[ns]->matchIdx : midx) ]->dSpanning[ (s2 != sidx ? mesh2->verts[ns2]->matchIdx : midx) ]);
				//fabs(mesh2->verts[ns]->dSpanning[ns2] - mesh2->avgLocalRadius[s]);
				//-mesh2->verts[ns]->dSpanning[ns2]; //ns/ns2.dSpanning[] ready for sure 'cos ns/ns2 is an existing sample 
				//mesh2->verts[ns]->dSpanning[ns2]; //minimizing this would collect all samples in center (assuming no feature term and no LOCAL_REPULSION)				
#else
				fabs((distanceBetween(mesh2->verts[ns]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist) - (distanceBetween(verts[ (s != sidx ? mesh2->verts[ns]->matchIdx : midx) ]->coords, verts[ (s2 != sidx ? mesh2->verts[ns2]->matchIdx : midx) ]->coords) / mesh2->maxEucDist)); //mesh1.maxEucDist undefined so ~wrong/close denom after minus
				//fabs((distanceBetween(mesh2->verts[ns]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist) - mesh2->avgLocalRadius[s]);
				//-distanceBetween(mesh2->verts[ns]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist;
				//distanceBetween(mesh2->verts[ns]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist; //minimizing this would collect all samples in center (assuming no feature term)
#endif
			
			pairwiseTerm += dist; //dist is already in [0,1]
			nAddsR++; //nAddsR addsto pairwiseTerm in total will be normalized below

//if (dist < minConstituentRT) minConstituentRT = dist;if (dist > maxConstituentRT) maxConstituentRT = dist;
		}
		///////////////// pairwiseTerm ends ///////////////////////////
		
		/////////////// singletonTerm //////////////////////
#ifdef MATCH_LOCAL_RADII
		cout << "computeAvgLocalRadius() not adjusted for evalObjectiveWithCandidate (an easy job)\n"; exit(0);
		mesh2->computeAvgLocalRadius(ns, s); //based on the ns.sampleNghbs[]
		singletonTerm += fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]); //added value is in [0,1] as avgLocalRadius[s] is normalized; in practice in [0, 0.05] as radius is just a small distance over the surface (as # samples increases radius decreases)
		//singletonTerm = fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]); //assigned value is in [0,1] as avgLocalRadius[s] is normalized; in practice in [0, 0.05] as radius is just a small distance over the surface (as # samples increases radius decreases)
//if (fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]) < minConstituentFT) minConstituentFT = fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]);if (fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]) > maxConstituentFT) maxConstituentFT = fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]);
#endif
		/////////////// singletonTerm ends //////////////////////
	}

	/////////////// singletonTerm //////////////////////
#ifdef MATCH_LOCAL_RADII
	singletonTerm /= n; //n additions in total to singletonTerm (not averaging any more so invalid comment)
#else
	mesh2->computeRadiusBasedOn(candidate, sidx); //mesh2.radiusN/minRadiusN updated

	//compare radiusN (maby later minRadiusN too) w/ the fixed radius in mesh1 (that i want to coverge to)
	singletonTerm = fabs(radiusN - mesh2->radiusN); //in [0,1] as radiusN is normalizedl in practice in [0, 0.05] as radius is just a small distance over the surface (as # samples increases radius decreases)
#endif
	/////////////// singletonTerm ends //////////////////////
	pairwiseTerm /= nAddsR; //denominator puts the pairwiseTerm  in [0,1] (when avgLocalRadius[] not in use), and assume the singletonTerm is already in [0.01, 0.11] with mean=0.03
						    //denominator puts the pairwiseTerm in [-1,0] (-ve dists); pairwiseTerm.mean = -0.36, and assume the singletonTerm is already in [0.01, 0.11] with mean=0.03; mean ratio ~ 10 is factored in singletonTerm via normalization factor nf (nf is auto-computed)

	return ((1.0f - alpha) * pairwiseTerm) + (alpha * singletonTerm * nf);
}
float Mesh::evalObjectiveOnVert(Mesh* mesh2, float alpha, int vert, int s)
{
	//same as evalObjective() except this evaluates the objective function on vert \in samples[] only, not all the samples; in other words no samples traversal here 'cos i'll focus only on one sample, namely the vert

	float pairwiseTerm = 0.0f, singletonTerm = 0.0f; //objective function = (1-alpha)*pairwiseTerm + alpha*singletonTerm
	int n = (int) mesh2->samples.size();
//	for (int s = 0; s < n; s++)
//	{
//		int ns = mesh2->samples[s]; using parameter vert instead of ns's

		///////////////// pairwiseTerm ///////////////////////////
		for (int s2 = 0; s2 < n; s2++)
		{
			int ns2 = mesh2->samples[s2]; //next sample different from vert
			if (ns2 == vert)
				continue;
			
			float dist =
#ifdef GEO_INSTEAD_OF_EUC
				fabs(mesh2->verts[vert]->dSpanning[ns2] - verts[ mesh2->verts[vert]->matchIdx ]->dSpanning[ mesh2->verts[ns2]->matchIdx ]);
				//fabs(mesh2->verts[vert]->dSpanning[ns2] - avgLocalRadius[s]);
				//-mesh2->verts[vert]->dSpanning[ns2]; //vert.dSpanning[] ready for sure 'cos vert is an existing sample 
				//mesh2->verts[vert]->dSpanning[ns2]; //minimizing this would collect all samples in center (assuming no feature term and no LOCAL_REPULSION)
#else
				fabs((distanceBetween(mesh2->verts[vert]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist) - (distanceBetween(verts[ mesh2->verts[vert]->matchIdx ]->coords, verts[ mesh2->verts[ns2]->matchIdx ]->coords) / mesh2->maxEucDist)); //mesh1.maxEucDist undefined so ~wrong/close denom after minus
				//fabs((distanceBetween(mesh2->verts[vert]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist) - avgLocalRadius[s]); //avgLocalRadius[] is based on geodesics but euc vs. goe should not be a problem within a small patch
				//-distanceBetween(mesh2->verts[vert]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist;
				//distanceBetween(mesh2->verts[vert]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist; //minimizing this would collect all samples in center (assuming no feature term)
#endif

			pairwiseTerm += dist; //dist is already in [0,1]
			//nAddsR++; save micro time 'cos this loop always iterates n-1 times; n-1 adds in total will be normalized below
		}
		///////////////// pairwiseTerm ends ///////////////////////////

		/////////////// singletonTerm //////////////////////
#ifdef MATCH_LOCAL_RADII
		mesh2->computeAvgLocalRadius(vert, s); //based on the ns.sampleNghbs[]
		singletonTerm = fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]); //added value is in [0,1] as avgLocalRadius[s] is normalized; in practice in [0, 0.05] as radius is just a small distance over the surface (as # samples increases radius decreases)
	/////////////// singletonTerm ends //////////////////////
//	}
#else
	mesh2->computeRadius(false); //mesh2.radiusN & mesh2.minRadiusN updated and ready
	//compare radiusN (maby minRadiusN later too) w/ the fixed radius in mesh1 (that i want to coverge to)
	singletonTerm = fabs(radiusN - mesh2->radiusN); //in [0,1] as radiusN is normalizedl in practice in [0, 0.05] as radius is just a small distance over the surface (as # samples increases radius decreases)
#endif

	return ((1.0f - alpha) * pairwiseTerm / (n-1)) + (alpha * singletonTerm * nf); //denominator n-1 puts the pairwiseTerm in [0,1] (when avgLocalRadius[] not in use); assume singletonTerm is in [0.01, 0.11] with mean=0.03
	//denominator n-1 puts the pairwiseTerm in [-1,0] (-ve dists); pairwiseTerm.mean = -0.36; assume singletonTerm is in [0.01, 0.11] with mean=0.03; mean ratio ~ 10 is factored in feature term via normalization factor nf (nf is auto-computed)
}
float Mesh::evalObjective3(Mesh* mesh2, float alpha, float& iso3, float& grd3)
{
	//same as evalObjective() except this also returns getIsometricDistortion() and getGroundDistortion() for fprints via call-by-ref

	float pairwiseTerm = 0.0f, singletonTerm = 0.0f; //objective function = (1-alpha)*pairwiseTerm + alpha*singletonTerm
	int n = (int) mesh2->samples.size(), nAddsR = 0;
	vector< int > corres;
	for (int s = 0; s < n; s++)
	{
		int ns = mesh2->samples[s]; //ns for next sample

		corres.push_back(mesh2->verts[ns]->matchIdx);
		corres.push_back(ns);			

		///////////////// pairwiseTerm ///////////////////////////
		for (int s2 = 0; s2 < n; s2++)
		{
			int ns2 = mesh2->samples[s2]; //next sample different from ns
			if (s2 <= s) //s2 <= s is to stop ns=ns2 (s=s2) and s2<s case from computing symmetric stuff below (symmetric 'cos s2>s already computed those distances)
				continue;

			float dist = 
#ifdef GEO_INSTEAD_OF_EUC
				fabs(mesh2->verts[ns]->dSpanning[ns2] - verts[ mesh2->verts[ns]->matchIdx ]->dSpanning[ mesh2->verts[ns2]->matchIdx ]);
				//fabs(mesh2->verts[ns]->dSpanning[ns2] - mesh2->avgLocalRadius[s]);
				//-mesh2->verts[ns]->dSpanning[ns2]; //ns/ns2.dSpanning[] ready for sure 'cos ns/ns2 is an existing sample 
				//mesh2->verts[ns]->dSpanning[ns2]; //minimizing this would collect all samples in center (assuming no feature term and no LOCAL_REPULSION)				
#else
				fabs((distanceBetween(mesh2->verts[ns]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist) - (distanceBetween(verts[mesh2->verts[ns]->matchIdx]->coords, verts[ mesh2->verts[ns2]->matchIdx ]->coords) / mesh2->maxEucDist)); //mesh1.maxEucDist undefined so ~wrong/close denom after minus
				//fabs((distanceBetween(mesh2->verts[ns]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist) - mesh2->avgLocalRadius[s]);
				//-distanceBetween(mesh2->verts[ns]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist;
				//distanceBetween(mesh2->verts[ns]->coords, mesh2->verts[ns2]->coords) / mesh2->maxEucDist; //minimizing this would collect all samples in center (assuming no feature term)
#endif
			
			pairwiseTerm += dist; //dist is already in [0,1]
			nAddsR++; //nAddsR addsto pairwiseTerm in total will be normalized below

//if (dist < minConstituentRT) minConstituentRT = dist;if (dist > maxConstituentRT) maxConstituentRT = dist;
		}
		///////////////// pairwiseTerm ends ///////////////////////////

		/////////////// singletonTerm //////////////////////
#ifdef MATCH_LOCAL_RADII
		mesh2->computeAvgLocalRadius(ns, s); //based on the ns.sampleNghbs[]
		singletonTerm += fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]); //added value is in [0,1] as avgLocalRadius[s] is normalized; in practice in [0, 0.05] as radius is just a small distance over the surface (as # samples increases radius decreases)
		//singletonTerm = fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]); //assigned value is in [0,1] as avgLocalRadius[s] is normalized; in practice in [0, 0.05] as radius is just a small distance over the surface (as # samples increases radius decreases)
//if (fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]) < minConstituentFT) minConstituentFT = fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]);if (fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]) > maxConstituentFT) maxConstituentFT = fabs(avgLocalRadius[s] - mesh2->avgLocalRadius[s]);
#endif
		/////////////// singletonTerm ends //////////////////////
	}

	/////////////// singletonTerm //////////////////////
#ifdef MATCH_LOCAL_RADII
	singletonTerm /= n; //n additions in total to singletonTerm (not averaging any more so invalid comment)
#else
	mesh2->computeRadius(false); //mesh2.radiusN/minRadiusN updated
	//compare radiusN (maby later minRadiusN too) w/ the fixed radius in mesh1 (that i want to coverge to)
	singletonTerm = fabs(radiusN - mesh2->radiusN); //in [0,1] as radiusN is normalizedl in practice in [0, 0.05] as radius is just a small distance over the surface (as # samples increases radius decreases)
#endif
	/////////////// singletonTerm ends //////////////////////
	pairwiseTerm /= nAddsR; //denominator puts the pairwiseTerm  in [0,1] (when avgLocalRadius[] not in use), and assume the singletonTerm is already in [0.01, 0.11] with mean=0.03
						    //denominator puts the pairwiseTerm in [-1,0] (-ve dists); pairwiseTerm.mean = -0.36, and assume the singletonTerm is already in [0.01, 0.11] with mean=0.03; mean ratio ~ 10 is factored in singletonTerm via normalization factor nf (nf is auto-computed)

	iso3 = getIsometricDistortion(mesh2, corres);
	grd3 = getGroundDistortion(mesh2, corres);

	return ((1.0f - alpha) * pairwiseTerm) + (alpha * singletonTerm * nf);
}

int Mesh::computeSampleNghbs(int sample, int i)
{
	//computes sampleNghbs[], i.e. surrounding samples of parameter sample

/*		for (int j = 0; j < n; j++)	//w/ radius just-below, times 2.0f makes avg too large (12); 1.5f makes avg ok (5) but some samples have just 1 neighbor; minRadius also sucks; so thresholding sucks do it BFS-style
		if (i != j && verts[ samples[i] ]->dSpanningUnn[ samples[j] ] <= 2.0f * minRadius) //i!=j ensures that sample is not in its neighborhood
			verts[ samples[i] ]->sampleNghbs.push_back(samples[j]);*/

//for (int v = 0; v < (int) verts.size(); v++) if (verts[v]->dBFS != INF) cout << "should have been inf\n";verified that this never prints
	queue< int > q;
	q.push(sample);
	verts[sample]->dBFS = 0.0f; //length of unweighted shortest path from source to source is 0
	while (! q.empty())
	{
		int v = q.front();
		q.pop();

		//look at each adjacent vertices of v
		bool earlyBreak = false;
		set< int, std::less< int > >::const_iterator neighbVertIdx;
		for (neighbVertIdx = verts[v]->vertNeighborsSet.begin(); neighbVertIdx != verts[v]->vertNeighborsSet.end(); neighbVertIdx++)
		{
			int va = (*neighbVertIdx);
			if (verts[va]->dBFS == INF)
			{
				verts[va]->dBFS = verts[v]->dBFS + 1.0f; //update d of this adjacent
				q.push(va); //va.parent = v; could also be done here but not using parents in my applicaiton
				float dist = verts[sample]->dSpanningUnn[va]; //sample.dSpanningUnn[] knwon for sure as it is a sample
				if (verts[va]->sample)
					verts[sample]->sampleNghbs.push_back(va); //thanks to BFS, sample neighbors are added in sorted order; so close samples guaranteed to be in sampleNghbs[] descpite early-break (not the case w/ disabled nested for's above)
				if ((int) verts[sample]->sampleNghbs.size() == 6 || dist > 3.0f * minRadius) //at most 6 samples (for extremties 2nd condition on dist will make less than 6 samples)
				{
					earlyBreak = true;
					break;
				}
			}
		}
		if (earlyBreak)
			break; //from while-loop
	} //end of while
	for (int v = 0; v < (int) verts.size(); v++)
		verts[v]->dBFS = INF; //clear effect(s) of this call(s) for the upcoming call (or for later usage of an irrelevant BFS; leave all the dBFS=INF)

	computeAvgLocalRadius(sample, i); //based on the sample.sampleNghbs[]

	return (int) verts[sample]->sampleNghbs.size();
}
void Mesh::computeAvgLocalRadius(int sample, int i)
{
	//computes sample.avgLocalRadius[]

//bool useMinRadiusInstead = true; //true to use min of all si candidates instead of the avg of all si candidates; avg sucks so trying the min
//float minRad = INF;
	int si = (int) verts[sample]->sampleNghbs.size();
	avgLocalRadius[i] = 0.0f;
	for (int j = 0; j < si; j++)
/*		if (useMinRadiusInstead)
		{
#ifdef GEO_INSTEAD_OF_EUC
			if (verts[sample]->dSpanning[ verts[sample]->sampleNghbs[j] ] < minRad)
				minRad = verts[sample]->dSpanning[ verts[sample]->sampleNghbs[j] ];
#else
			if (distanceBetween(verts[sample]->coords, verts[ verts[sample]->sampleNghbs[j] ]->coords) / maxEucDist < minRad)
				minRad = distanceBetween(verts[sample]->coords, verts[ verts[sample]->sampleNghbs[j] ]->coords) / maxEucDist;
#endif
		}
/*		else*/
			avgLocalRadius[i] +=
#ifdef GEO_INSTEAD_OF_EUC
				verts[sample]->dSpanning[ verts[sample]->sampleNghbs[j] ]; //i'th avg radius to be used in singeltonTerm
#else
				distanceBetween(verts[sample]->coords, verts[ verts[sample]->sampleNghbs[j] ]->coords) / maxEucDist;
#endif*/
//	if (useMinRadiusInstead)
//		avgLocalRadius[i] = minRad;
//	else //need averaging so div by si
		avgLocalRadius[i] /= si;
//if (si != 6) cout << "irregular sample: " << sample << "\t" << si << endl;//no problem to be irregular
}
void Mesh::computeRadius(bool print)
{
	//computes sampling radius of this mesh by averaging the closest-sample distances to each sample; global radiusN and minRadiusN are updated so are radius and minRadius

	radiusN = radius = 0.0f;
	minRadius = INF;
	int n = (int) samples.size();
	for (int i = 0; i < n; i++)
	{
		float minDist = INF, minDistN = INF;
		int closest = -1;
		for (int j = 0; j < n; j++)
			if (i != j && verts[ samples[i] ]->dSpanning[ samples[j] ] < minDistN)
			{
				minDistN = verts[ samples[i] ]->dSpanning[ samples[j] ];
				minDist = verts[ samples[i] ]->dSpanningUnn[ samples[j] ];
				closest = samples[j];
			}
		radius += minDist;
		radiusN += minDistN;
		if (minDist < minRadius)
		{
			minRadius = minDist;
			minRadiusN = minDistN;
			mrIdx = samples[i];
			mrIdx2 = closest;
		}
	}
	radius /= n;
	radiusN /= n;
#ifdef VERBOSE
	if (print)
		cout << "radius: " << radius << " (normalized " << radiusN << ")\t(closest pair dist: " << minRadius << " (normalized " << minRadiusN << ") from " << mrIdx << " to " << mrIdx2 << ")\n";
#endif
}
void Mesh::computeRadiusEff()
{
	//same as computeRadius() but this is more efficient thanks to the existence of sampleNghbs[]

	radiusN = radius = 0.0f;
	minRadius = INF;
	int n = (int) samples.size();
	for (int i = 0; i < n; i++)
	{
		float minDist = INF, minDistN = INF;
		int closest = -1;
		//sampleNghbs[] ready for sure; so check only this small subset since the closest must exist here
		for (int j = 0; j < (int) verts[ samples[i] ]->sampleNghbs.size(); j++)
			if (verts[ samples[i] ]->dSpanning[ verts[ samples[i] ]->sampleNghbs[j] ] < minDistN) //no i!=j 'cos s.sampleNghbs[] excludes s
			{
				minDistN = verts[ samples[i] ]->dSpanning[ samples[j] ];
				minDist = verts[ samples[i] ]->dSpanningUnn[ samples[j] ];
				closest = samples[j];
			}
		radius += minDist;
		radiusN += minDistN;
		if (minDist < minRadius)
		{
			minRadius = minDist;
			minRadiusN = minDistN;
			mrIdx = samples[i];
			mrIdx2 = closest;
		}
	}
	radius /= n;
	radiusN /= n;
}
void Mesh::computeRadiusBasedOn(int va, int s)
{
	//same as computeRadius() except for the s'th sample va is used instead of samples[s]; global radiusN and minRadiusN are updated so are radius and minRadius

	radiusN = radius = 0.0f;
	minRadius = INF;
	int n = (int) samples.size();
	for (int i = 0; i < n; i++)
	{
		float minDist = INF, minDistN = INF;
		int closest = -1, sample = (i != s ? samples[i] : va);
		for (int j = 0; j < n; j++)
			if (i != j && verts[ samples[j] ]->dSpanning[ sample ] < minDistN)
			{
				minDistN = verts[ samples[j] ]->dSpanning[ sample ];
				minDist = verts[ samples[j] ]->dSpanningUnn[ sample ];
				closest = samples[j];
			}
		radiusN += minDistN;
		if (minDist < minRadius)
		{
			minRadius = minDist;
			minRadiusN = minDistN;
		}
	}
	radius /= n;
	radiusN /= n;
}

void Mesh::interpolateCoarseMap(Mesh* mesh2)
{
	//interpolates GA's 100x100 coarse mapping to 12Kx12K full dense mapping and fwrites it

	//GA.go() set matchIdx's and resultToFile() set the corresp[] (and updated matchIdx's after outlier removals); corresp[] and matchIdx's to be used below
	int m = (int) corresp.size() / 2;
#ifdef VERBOSE
	cout << "interpolating coarse GA map of size " << m << " into a dense one (could've been faster if kd-tree was in use while finding the closest vertex)..\n" ;
#endif
	//for each vert v, create the m-size gv[] list (m = # of coarse correspoondence verts) where v.gv[4] gives the geodesic dist from v to 5th coarse corresp matched vert, i.e. 5th sample
	for (int v = 0; v < (int) verts.size(); v++)
	{
		verts[v]->gv.clear(); //in case it has been filled before (for samples)
		for (int s = 0; s < m; s++)
		{
			dijkstraShortestPaths(corresp[2*s]); //returns immediately if dSpanning computed already; so no inefficiency for that; this call mandatory 'cos robustListInUse=true case makes dSpanning ready only for the robustList and now i need it ready for all samples (robustList + euclidean-FPS samples in computeSamples())
			verts[v]->gv.push_back(verts[ corresp[2*s] ]->dSpanning[v]);
		}
	}
	for (int v = 0; v < (int) mesh2->verts.size(); v++)
	{
		mesh2->verts[v]->gv.clear();
		for (int s = 0; s < m; s++)
		{
			mesh2->dijkstraShortestPaths(corresp[2*s+1]);
			mesh2->verts[v]->gv.push_back(mesh2->verts[ corresp[2*s+1] ]->dSpanning[v]);
		}
	}
int stage = 0, //1st stage assigns the closest/best (w.r.t. gv) v1 to current v2; 2nd stage assigns the 2nd best ('cos the 1st best would already been in use when i am in the 2nd stage)
	nMaxStage = 2; //use at most 2 here (3+ not supported)
while (stage++ < nMaxStage)
{
if(nMaxStage>2)exit(0);
	//each mesh2 vert v2 is matched to v1 where v1.gv is closest to v2.gv and v1 is not processed yet
	for (int v1 = 0; v1 < (int) verts.size(); v1++)
		verts[v1]->processed = (verts[v1]->matchIdx != -1 ? true : false); //true means already matched with another mesh vert so not use it to maintain a 1-to-1 bijective map
	int nUnmatched = 0;
	for (int v2 = 0; v2 < (int) mesh2->verts.size(); v2++)
	{
		mesh2->verts[v2]->color = color; //matched verts' colors will be overwritten below
		if (mesh2->verts[v2]->matchIdx != -1) //GA.go() set matchIdx's
			continue; //already placed in corresp w/ its non-interpolated match
		float minDist = INF;
		int bestV1 = -1, prevBestV1 = -1;
		for (int v1 = 0; v1 < (int) verts.size(); v1++)
		{
			float dist = 0.0f; //dist b/w v1.gv & v2.gv
			for (int i = 0; i < m; i++)
				dist += fabs(verts[v1]->gv[i] - mesh2->verts[v2]->gv[i]); //i'th entries are compatible as they're geodesic dists to i'th coarse match
			if (dist < minDist)
			{
				prevBestV1 = bestV1; //useful only for the 2nd stage
				bestV1 = v1;
				minDist = dist;
			}
		}
		if ((stage == 1 && bestV1 == -1) || (stage == 2 && prevBestV1 == -1))
		{
			cout << "no match for mesh2.vert" << v2 << "\t" << minDist << " " << bestV1 << " " << prevBestV1 << endl; //this happens when verts.size() < mesh2.verts.size()
			//exit(0);
		}
		if (stage == 1 && verts[bestV1]->processed)
		{
			nUnmatched++;
			continue;
		}
		else if (stage == 2 && (prevBestV1 == -1 || verts[prevBestV1]->processed))
		{
			nUnmatched++;
			continue;
		}
		
		if (stage == 1)
		{
			corresp.push_back(bestV1);
			corresp.push_back(v2); //note that m not affected for the upcoming iter
			verts[bestV1]->processed = true;
			verts[bestV1]->matchIdx = v2;
			mesh2->verts[v2]->matchIdx = bestV1; //dual operation
		}
		else if (stage == 2)
		{
			corresp.push_back(prevBestV1);
			corresp.push_back(v2); //note that m not affected for the upcoming iter
			verts[prevBestV1]->processed = true;
			verts[prevBestV1]->matchIdx = v2;
			mesh2->verts[v2]->matchIdx = prevBestV1; //dual operation
		}

#ifdef VERBOSE
		if ((stage == 1 && v2 % 2000 == 0) || (stage == 2 && v2 % 2001 == 0))
			cout << v2 << " / " << (int) mesh2->verts.size() << "th match computed via interpolation\n";
#endif
	}
#ifdef VERBOSE
	cout << nUnmatched << " unmatched after stage" << stage << endl;
#endif
}
	//remaining matches will be made using the first unassigned match (can be the 3rd best, 4th best, ..) as long as an outlier is not formed; if outlier then that v2 vert will remain unmtched forever
	int nUnmatched = 0;
	float maxErr = 0.125f;//0.125f; //outlier detection (based on GA.gvCompatible())
	for (int v2 = 0; v2 < (int) mesh2->verts.size(); v2++)
	{
		if (mesh2->verts[v2]->matchIdx != -1)
			continue;
		float minDist = INF;
		int bestV1 = -1;
		for (int v1 = 0; v1 < (int) verts.size(); v1++)
		{
			if (verts[v1]->processed)
				continue; //this could have been the 3rd best, 4th best, .. for v2 but already assigned so i'm skipping it
			float tmp = 0.0f, dist = 0.0f; //dist b/w v1.gv & v2.gv
			bool outlier = false;
/*			for (int i = 0; i < m; i++)
			{
				tmp = fabs(verts[v1]->gv[i] - mesh2->verts[v2]->gv[i]); //i'th entries are compatible as they're geodesic dists to i'th coarse match
				if (tmp > maxErr)
					outlier = true; //even if this v1 passes dist < minDist test below, it should not be assigned to v2 'cos (v1-v2) would make an outlier
				dist += tmp;
			} outliers are also welcome so that there is no unmatched vert in mesh2*/
			if (! outlier && dist < minDist)
			{
				bestV1 = v1;
				minDist = dist;
			}
		}
		if (bestV1 == -1)
		{
			nUnmatched++;
			continue;
		}
		corresp.push_back(bestV1);
		corresp.push_back(v2); //note that m not affected for the upcoming iter
		verts[bestV1]->processed = true;
		verts[bestV1]->matchIdx = v2;
		mesh2->verts[v2]->matchIdx = bestV1; //dual operation
#ifdef VERBOSE
		if (v2 % 2002 == 0)
			cout << v2 << " / " << (int) mesh2->verts.size() << "th match computed via interpolation\n";
#endif
	}
#ifdef VERBOSE
	cout << nUnmatched << " unmatched after final stage\n";
#endif
	//fwrite corresp in the format of *level*.dat files but simpler, i.e. no worst stuff and baseNghbhood stuff, that is nothing after '-2' in levelInfoToFile()
	char fOutName[250];
	sprintf(fOutName, "output\\%d-%ddense.dat", id, mesh2->id);
	FILE* fOutPtr = fopen(fOutName, "w");
	//correspondences to be loaded/fread into nonrobust for a later time; also set the mesh2 coloring using the dense map
	computeVertColors();
	for (int i = 0; i < (int) corresp.size(); i += 2)
	{
		fprintf(fOutPtr, "%d\t%d\n", corresp[i], corresp[i+1]);
		mesh2->verts[ corresp[i+1] ]->color = verts[ corresp[i] ]->color; //corresp[*] cannot be -1 here so no if-control
	}
	fclose(fOutPtr);
	cout << "after interpolation dense map of size " << (int) corresp.size()/2 << " is ready and fprinted\n";
}
void Mesh::computeVertColors()
{
	//determine constant color values of source (mesh1) verts for drawing purposes; the same color will be transferred to the computed match vert in target (mesh2)

	float amaxX, amaxY, amaxZ; //absolute max x-y-z for normalization of color values into [0,1]
	for (int i = 0; i < (int) verts.size(); i++)
		if (i == 0)
		{
			amaxX = fabs(verts[i]->coords[0]);	amaxY = fabs(verts[i]->coords[1]);	amaxZ = fabs(verts[i]->coords[2]);
		}
		else
		{
			if (fabs(verts[i]->coords[0]) > amaxX) amaxX = fabs(verts[i]->coords[0]);
			if (fabs(verts[i]->coords[1]) > amaxY) amaxY = fabs(verts[i]->coords[1]);
			if (fabs(verts[i]->coords[2]) > amaxZ) amaxZ = fabs(verts[i]->coords[2]);
		}
	for (int i = 0; i < (int) verts.size(); i++)
	{
		verts[i]->color = new float[3];
		verts[i]->color[0] = fabs(verts[i]->coords[0])/amaxX;
		verts[i]->color[1] = fabs(verts[i]->coords[1])/amaxY;
		verts[i]->color[2] = fabs(verts[i]->coords[2])/amaxZ; //vert color according to its x-y-z coord
	}
}

inline int factorial(int k)
{
	if (k > 10)
		return -1; //don't bother big numbers
	int res = 1;
	for (int i = k; i > 0; i--)
		res *= i;
	return res;
}
void Mesh::bruteForceMap(Mesh* mesh2, int r)
{
	//find the map b/w mesh1.samples and mesh2.samples by looking at all possible permutations (intractable for CHROM_LEN > 10)
	
#ifdef VERBOSE
	cout << "brute force map computation..";
#endif
	//fread permutations from file to perms10[][] mtrx
	FILE* fPtr;
	char fName[150];
	int factR = factorial(r); //other permutation less than 10 are available in fName folder but here i just need 10!	
	if (r != 10)
	{
		cout << "10- could be supported easily but 10+ is intractable\n";
		exit(0);
	}
	//10! permutations in perms10[][]
	sprintf(fName, "C:\\PhD\\Dropbox_koc\\my codes\\tools\\%d!.txt", r); //matlab code that generated this txt is in that folder
	if (! (fPtr = fopen(fName, "r")))
	{
		cout << fName << " not found\n";
		exit(0);
	}
	//matlab perms(0:10) took 1 second and fprinting the resulting 10! entries to the file above took 2 mins
	int** perms10 = new int*[factR];
	for (int i = 0; i < factR; i++) //r! reorderings/permutations
	{
		perms10[i] = new int[r];
		for (int j = 0; j < r; j++)
			fscanf(fPtr, "%d\t", &perms10[i][j]);
	}
	fclose(fPtr);
	cout << "permut file fread..";

	//use permutations to do the brute-force map computation
	vector< int > optimalMap; //non-empty vector; do optimalMap(2*r, 400) to alloc 2r items whose initial values are all 400
//cout << optimalMap[0] << " - " << optimalMap[1] << endl;//crushes if inited to an empty vector (vector< int > optimalMap;)
	float minDistortion = INF;
	for (int i = 0; i < factR; i++)
	{
		//this exhaustive loop takes 3secs for r=10 (permutations fread above created in matlab in additional 1sec), which is still slower than my GA which takes 0.5 sec to compute the same optimal 10x10 map; becomes intractable as r increases

		//refill tmpSafeCorrespTry according to the current reordering (one of k! permutations) of mesh2 samples
		vector< int > tmpSafeCorrespTry;
		int idx1 = 0, idx2 = 0;
		for (int y = 0; y < 2*r; y++)
			if (y % 2 == 0)
				tmpSafeCorrespTry.push_back(samples[idx1++]); //even idxs from the fixed mesh1 side
			else //odd idxs for permutations from mesh2 side
				tmpSafeCorrespTry.push_back(mesh2->samples[ perms10[i][idx2++] ]);
		float isometryCostTry = getIsometricDistortion(mesh2, tmpSafeCorrespTry);
		if (isometryCostTry < minDistortion)
		{
			minDistortion = isometryCostTry;
			optimalMap = tmpSafeCorrespTry;
			//for (int j = 0; j < (int) tmpSafeCorrespTry.size(); j++) optimalMap[j] = tmpSafeCorrespTry[j]; same as just-above
		}
	}
	//transfer the content of optimalMap into Mesh.verts to see it on screen
	for (int s = 0; s < r; s++)
	{
		verts[ samples[s] ]->matchIdx = optimalMap[2*s + 1];
		mesh2->verts[ optimalMap[2*s + 1] ]->matchIdx = samples[s]; //dual operation
	}
	sprintf(fName, "output\\%d-%d.dat", id, mesh2->id);
	resultToFile(fName, mesh2, false, false);

	//recapture memory
	for (int i = 0; i < factR; i++)
		delete [] perms10[i];
	delete [] perms10;
	cout << "done!\n\n";
}
