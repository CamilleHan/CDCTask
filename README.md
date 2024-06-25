# Communication-Dependent Computing Resource Management for Concurrent Task Orchestration in IoT Systems

This repository contains the codes for our work "Communication-Dependent Computing Resource Management for Concurrent Task Orchestration in IoT Systems".


 
# Requirements
The following versions have been tested: Python 3.7.16 + Pytorch 1.13.1+cu116. But newer versions should also be fine.


## The introduction of each file

`main.py`: run this file to simulate the methods for CDC resource management and concurrent task orchestration.


### Configurations
`options.py`: the configuration file of this work, and the parameters can be adjusted for testing other cases.

### Server and client settings

`TG_server.py`: the TG server simulator.

`TC_client.py`: the TC client simulator.

### Data Processing

`datasets/cifar_mnist.py`: get the mnist and cifar10 datasets, then some split operations on them to distribute on TC clients and TG servers.

`NNmodels`: distributed machine learning models for local training on clients.

