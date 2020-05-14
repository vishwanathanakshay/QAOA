# QAOA
This repo contains a QAOA Template that takes in a diagonal Hamiltonian (in the sigma_z basis) and runs the algorithm to output the QAOA cost as well as the success probability or the Ground state overlap. 
## Package and version Information 
```python
scipy==1.2.1
numpy==1.16.2
networkx==2.3
```
## QAOA on MAX-CUT
The current implimentation contains an example to run MAX-CUT on random graph instances. After a successfull compilation the following example can be run: 
```python
#TEST CODE: Runs 1p-QAOA on MAX-CUT Hamiltonian. Instance generated from a random graph with 6 nodes and 10 edges. 
nodes = 6
edges = 10
depth = 1
G = nx.gnm_random_graph(nodes,edges)
G = nx.to_numpy_array(G)
H = Graph_to_Hamiltonian(G,nodes)
Q = QAOA(depth,H)
result = Q.run()
print(f'QAOA cost:{result[0]} , Success probability: {result[1]}')
```
## Contact
The code is straightforward and self-explanitory since only the basic packages are used. For additional information, explantion and comments please contact:\
\
Akshay Vishwanathan\
PhD Student\
Deep Quantum Labs,Skoltech\
Moscow, Russia\
email: akshay.vishwanathan@skoltech.ru

