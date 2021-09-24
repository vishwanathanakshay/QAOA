# QAOA
This repo contains a QAOA Template that takes in a diagonal Hamiltonian (in sigma_z basis) and runs the algorithm to output the QAOA cost as well as the success probability or the ground state overlap. The repo also includes a faster implementation (qaoa_class) and some generators for Graph hamiltonians. qaoa_class only requires basic python packages and can easilly reproduce the ideal-noiseless Google data publshed in [Nature Physics](https://doi.org/10.1038/s41567-020-01105-y).   
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

