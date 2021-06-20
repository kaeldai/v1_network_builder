Network Builder
===============

Uses bmtk.builder module to (re)build the V1/LGN/BKG biophysically detailed network files. 
$ python build_network.py


This will create a ./network/ directory containing all the nodes and edges files required to run the network simulation (eg ../Biophysical_network/network/*).
You can change the directory where the files will be written:
$ python build_network.py --output-dir ~/my_rebuilt_network


If you want to rebuild only the V1 part of the network (V1 nodes and recurrent V1 to V1 edges)
$ python build_network.py v1


You can also just build the LGN and/or BKG files independently, which will take less time. However to create LGN/BKG to V1 connections you need to point to 
a directory containing v1 nodes
$ python build_network.py --v1-nodes-dir ../Biophysical_network/network lgn bkg


Often for testing purposes it may be helpful to build a smaller sampled version of the model. In the following the V1 network will be built using only 2309 
V1 nodes (1% of the original 230924), but with the same proportion of cell types and same connectivity rules:
$ python build_network.py --fraction 0.01

Folder Description
------------------
* node_props/ - contains files with properties on the network nodes
* conn_props/ - contains files for creating connectivity
* build_network.py - main script for building network
* node_funcs.py - functions and parameters used for building individual nodes
* connection_rules.py - functions for defining network connections.# v1_network_builder