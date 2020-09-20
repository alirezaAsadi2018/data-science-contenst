#include<iostream>
#include <string.h> 
#include <cstdio>
#include <algorithm>
#include <queue>
#include <vector>
#include<string>
#include<limits.h>

#define V 1000

#define fore(i, a, b) for(int i = a; i < b; ++i)
#define Trace(x) cout<< #x << " : " << x << endl
#define pb push_back
typedef long long ll;
typedef unsigned long long ull;

using namespace std;
int n,e;
std::vector<int> srcs;
std::vector<int> ts;
std::vector<int> ps;
int graph[V][V] = {{0}};
  
bool bfs(int rGraph[V][V], int s, int t, int parent[]) 
{ 
    bool visited[V]; 
    memset(visited, 0, sizeof(visited)); 

    queue <int> q; 
    q.push(s); 
    visited[s] = true; 
    parent[s] = -1; 
  
    while (!q.empty()) 
    { 
        int u = q.front(); 
        q.pop(); 
  
        for (int v=0; v<V; v++) 
        { 
            if (visited[v]==false && rGraph[u][v] > 0) 
            { 
                q.push(v); 
                parent[v] = u; 
                visited[v] = true; 
            } 
        } 
    } 
    return (visited[t] == true); 
} 
  
int fordFulkerson(int s, int t) 
{ 
    int u, v; 
    int rGraph[V][V];
    for (u = 0; u < V; u++) 
        for (v = 0; v < V; v++) 
             rGraph[u][v] = graph[u][v]; 
  
    int parent[V]; 
  
    int max_flow = 0;
  
    while (bfs(rGraph, s, t, parent)) 
    {  
        int path_flow = INT_MAX; 
        for (v=t; v!=s; v=parent[v]) 
        { 
            u = parent[v]; 
            path_flow = min(path_flow, rGraph[u][v]); 
        } 
  
        for (v=t; v != s; v=parent[v]) 
        { 
            u = parent[v]; 
            rGraph[u][v] -= path_flow; 
            rGraph[v][u] += path_flow; 
        } 
        max_flow += path_flow; 
    } 
    return max_flow; 
} 

int main(){
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	cin>>n;
	fore(i,0,n){
	    int x;
	    cin>>x;
	    if(x == 0){
	        ps.pb(i);
	    }else if(x == 1){
	        srcs.pb(i);
	    }else{
	        ts.pb(i);
	    }
	}
	cin>>e;
	fore(i,0,e){
	    int a,b,c;
	    cin >> a >> b >> c;
	    --a; --b;
	    graph[a][b] = c;
    }
	int dummy_src = n;
	int dummy_sink = n+1;
	int max_cap = INT_MAX;
    fore(i,0,srcs.size()){
		graph[dummy_src][srcs[i]] = max_cap;  
    }
	fore(i, 0, ts.size()){
	   graph[ts[i]][dummy_sink] = max_cap;
	}
   cout<< fordFulkerson(dummy_src, dummy_sink); 
  
    return 0; 
}