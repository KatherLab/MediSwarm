**Project ODELIA**

# MediSwarm Software Requirement Specification

Document-ID: MediSwarm-SPEC-SRS

### References

| Document-ID        | Title                                |
| :----              | :----                                |
| MediSwarm-SPEC-USR | MediSwarm Use Cases and User Stories |


## 1 Introduction and Overview

MediSwarm is a toolkit for swarm learning, i.e., distributed training of AI models for medical image data with the actual image data kept private at each participating node.

The swarm learning framework (SLF) shall be viewed separately from the application code (ApC), i.e. the code that handles training of a model for a specific purpose. The purpose of the SLF is to orchestrate the process of model updates and to keep track of metrics and performance characteristics of the trained model without interpretation of those characteristics and agnostic with respect to the specific training task. The SLF shall be accompanied by such ApC in order to be used in pre-flight testing and productive use. Testing ApC independent of the productive task is part of the system, ApC for specific tasks is not part of the system. Hence, this document describes generic requirements for ApC, specific requirements for the ApC depending on its specific purpose are beyond the scope of this document.

The framework is based on NVIDIA’s NVFlare ([https://github.com/NVIDIA/NVFlare](https://github.com/NVIDIA/NVFlare)), version 2.4.1. ODELIA, specifically the ODELIA teams at TU Dresden and Fraunhofer MEVIS, implements functionality needed for its purposes on top of NVFlare. This implementation is open source and available via [https://github.com/KatherLab/MediSwarm/](https://github.com/KatherLab/MediSwarm/).

In this prototype use case, a number of limitations/assumptions apply:

* The framework is meant to train a hard-coded convolutional neural network architecture for a binary classification of breast MRI images (preprocessed).
* Node availability for starting a training has to be communicated with the respective node operators outside the framework.
* Being able to initiate a training requires special privileges.
* Initiators of SL experiments are trusted to only run the code previously agreed on

### 1.1 Workflows

#### 1.1.1 Code Versioning

All code is maintained in a repository (git) containing the SLF, ApC, scripts for building docker images and startup kits, and utility scripts (e.g., for VPN setup and obtaining example data). A fork of NVFlare is kept in a separate repository, which is used as a git submodule in the MediSwarm repository. This setup allows identifying all code intended for a specific swarm training experiment via a single commit ID (hash).

#### 1.1.2 Network Communication

Network communication between all nodes is routed through a VPN with state-of-the-art encryption, ensuring connections across institutional firewalls protected against access and interference by non-participants.

#### 1.1.3 Testing Swarm Readiness

Two aspects of swarm readiness need to be tested before a swarm training can be run successfully: The ApC needs to run within the MediSwarm framework and the client nodes need to be able to run the application code with GPU usage in a Docker container being able to communicate with the server node.

##### 1.1.3.1 Verifying Swarm Readiness of Application Code

NVFlare provides a “simulation mode” and a “proof-of-concept mode” for running ApC in separate threads and processes, respectively, on a single node. MediSwarm makes these available in the execution environment of the ApC in the swarm training.
This allows ApC developers to verify that their ApC is compatible with the NVFlare interfaces for swarm training.

##### 1.1.3.2 Verifying Swarm Readiness of Client Nodes

MediSwarm provides a client node preflight check consisting of two steps, Docker+GPU preflight check and local data access preflight check run as local (i.e., non-swarm) trainings.
The Docker+GPU preflight check uses a “minimal training ApC” which runs a training of a small (\<100 parameters) model on data generated on the fly. This allows verifying that a client node is able to run the Docker container and able to run training code with GPU access inside the container in under one minute.
The local data access preflight check uses the productive ApC to train one epoch on the data at the respective node. This allows verifying that the local data is accessible by the productive ApC and that the node is also able to run the productive ApC, providing feedback in a few minutes (depending on the size of the dataset).

##### 1.1.3.3 Swarm Preflight Check

Once the ApC and the client nodes have been verified, and the swarm nodes are operative, a swarm training initiator can deploy the minimal training ApC prior to deploying the productive ApC. This allows verifying that the communication, distributed training, and aggregation within the swarm is actually working.

#### 1.1.4 Usage

##### 1.1.4.1 Swarm Setup

The swarm maintainer

* builds a container image (Docker) containing the SLF (including the execution environment for the application code), the minimal training ApC, and the productive ApC
* uploads the image to an image registry
* builds startup kits for each client node, for the server node and for the admin console (containing auth certificates and scripts for starting nodes; by using the container above)
* connects the SL server host to the VPN
* starts the SL server container using the respective startup kit
  * Note: this “server” has a purely technical role and should not be confused with a central aggregator in non-swarm federated learning
* distributes VPN login data and the client node startup kits to the respective client node maintainers by means outside the system (e.g., e-mail)
* monitors SL server operation via the logged output

##### 1.1.4.2 Node Setup

Each client node maintainer

* connects the client host to the VPN
* runs the Docker+GPU preflight check using  their respective startup kit (which will download the appropriate Docker container)
* runs the local data access preflight check specifying the location of the private data using the startup kit to verify that the private data can be used the application code for a training to work
* debugs the previous steps as needed
* starts the SL server container using the startup kit
* monitors SL client operation via the logged output
* has the opportunity to verify image integrity via its hash
* has the opportunity to run a local training on the private data for comparison to a swarm-trained model using the respective startup kit

##### 1.1.4.3 Running the Training

The swarm training initiator

* Deploys the minimal training ApC using the admin console via their respective startup kit and verifies that this job has succeeded within a few minutes
* Deploys the productive ApC to start the intended swarm training
* Note: For technical reasons, starting the respective ApC currently requires deploying it even though the clients already have it as part of the Docker image.
* Limited information on the status of the swarm (e.g. currently active nodes) is available via the admin console.
* Information on the status of individual nodes and performance of current and individually trained models during training is available only via log files that need to be requested from the respective client node maintainers

##### 1.1.4.4 Results

* Each client node finds the jointly trained model at a specified location when the training has finished

## 2 Abbreviations and Terminology

* **SLF (Swarm Learning Framework)** orchestrates the training, communicates model updates between swarm learning nodes, aggregates model updates in each round, chooses the next aggregating node(s).
* **ApC (Application Code)** Loads the image data and targets, runs the actual training in each round, implements interfaces for the SLF to run the training.

* **Distributed learning**. Training algorithms with data and compute resources located at different sites.
* **Federated learning**. Distributed learning in which data stays private at the participating sites (usually with one central node coordinating the training process and aggregating model updates).
* **Swarm learning**. Federated learning without a central coordinating and aggregating node.
* **Round.** Iteration of local training and one aggregation (including the respective model update transfer)
* **Model.** A model has a fixed architecture/structure and contains parameters (model weights and biases) adapted and aggregated during training.

## 3 Software Requirements

### 3.1 System Requirements

#### 3.1.1 Client Node Hardware

* 64 GB of RAM (32 GB is the absolute minimum)
* 16 CPU cores (8 is the absolute minimum)
* an NVIDIA GPU with 48 GB of RAM (24 is the minimum)
* 8 TB of Storage (4 TB is the absolute minimum)

#### 3.1.2 Node Software (All Nodes)

* Ubuntu 24.04 LTS
* Docker 29.4.1

#### 3.1.3 Network

* VPN
* Gigabit connection recommended, as model updates in the GB range need to be transferred between all nodes per round

### 3.2 Functional and Capability Requirements

#### 3.2.1 Federated/Swarm Learning

* **(A.1)** The system shall enable training of a model on several nodes. (I.e., distributed learning)
* **(A.2)** The system shall communicate only models/checkpoints as well as performance metrics (accuracy) between the nodes
* **(A.3)** The system shall enable one node (admin node) to start a training run.
* **(A.4)** Verifying that the nodes are available and ready to start shall happen via communication outside the system.
* **(A.5)** The system shall enable one node (admin node) to monitor whether a training is running, has succeeded, or has failed.
* **(A.6)** The system shall designate one node for each round of training to collect model updates, aggregate them, and distribute them to all nodes. In particular the system shall work without a fixed central aggregating node. (I.e., swarm learning).
* **(A.7)** The system shall provide means to define a minimum number of responsive nodes to continue the training.
* **(A.8)** The system shall provide means to continue the training if individual nodes do not respond given the defined minimum number of nodes is still available.

Requirements A.9 to A.11 ensure that the system provides means for aggregation of models/checkpoints from different nodes:

* **(A.9)** The aggregator shall be able to accept models/checkpoints from multiple nodes provided by the SLF
* **(A.10)** The aggregator shall be able to compute aggregated models/checkpoints via a defined aggregation strategy
* **(A.11)** The aggregator shall provide the aggregated models/checkpoints to the SLF for distribution to the training clients
* **(A.12)** Communication of metadata of the training data, e.g., summary statistics needed to characterize the overall dataset for a scientific publication, shall happen via communication outside the system.
* **(A.13)** Archival of training data, e.g., for regulatory purposes or to fulfil good scientific practice, shall happen outside the system.

#### 3.2.2 Container Images

* **(B.1)** The system shall be capable of generating container images with all required software installed for generating startup kits, running the server node, the admin console, and the client nodes.
* **(B.2)** The software packages installed for the different purposes shall be the same version.
* **(B.3)** The container image for the admin node shall contain the code needed to start trainings in the swarm
* **(B.4)** The container image for the admin node shall contain the ApC for the swarm preflight check
* **(B.5)** The container image for the admin node shall contain the productive ApC
* **(B.6)** The container image for the server node shall contain the code needed to run a swarm server
* **(B.7)** The container image for the client nodes shall contain the code for running a client node
* **(B.8)** The container image for the client nodes shall contain the ApC for the Docker+GPU preflight check
* **(B.9)** The container image for the client nodes shall contain the ApC for the local data access preflight check
* **(B.10)** The container image for the client nodes shall contain the productive ApC for local training
* **(B.11)** The container image for generating the startup kits shall contain the code necessary to build the startup kits
* **(B.12)** The container image for generating the startup kits shall contain the configuration files (swarm description, description of scripts for the different types of nodes) necessary to build the startup kits
* **(B.13)** The docker image for the client node shall have all required software installed to run the ApCs.
* **(B.14)** The docker image for the client node shall be able to execute the ApCs.

Note: typically, a single container image containing all required components is used for all types of nodes and to build the startup kits to ensure equal versions.

#### 3.2.3 Startup Kits

* **(C.1)** The system shall be capable of generating so-called startup kits for each node containing the required data to join the swarm.
* **(C.2)** All startup kits shall contain certificates for the respective node and a certificate authority certificate for mutual authentication
* **(C.3)** The startup kit for the admin node shall contain a script to be executed on the respective host to obtain the required container image (e.g., download the image from DockerHub)
* **(C.4)** The startup kit for the admin node shall contain a script to be executed on the respective host to start the admin console in the container
* **(C.5)** The startup kit for the server node shall contain a script to be executed on the respective host to obtain the required container image
* **(C.6)** The startup kit for the server node shall contain a script to be executed on the respective host to start the server node in the container
* **(C.7)** The startup kit for the client nodes shall contain a script to be executed on the respective host to obtain the required container image
* **(C.8)** The startup kit for the client nodes shall contain a script to be executed on the respective host to start the container with appropriate access to the GPU, local data and storage for logs and trained models/checkpoints to run the Docker+GPU preflight check
* **(C.9)** The startup kit for the client nodes shall contain a script to be executed on the respective host to start the container with appropriate access to the GPU, local data and storage for logs and trained models/checkpoints to run the local data access preflight check
* **(C.10)** The startup kit for the client nodes shall contain a script to be executed on the respective host to start the container with appropriate access to the GPU, local data and storage for logs and trained models/checkpoints to start the client node in the container
* **(C.11)** The startup kit for the client nodes shall contain a script to be executed on the respective host to start the container with appropriate access to the GPU, local data and storage for logs and trained models/checkpoints to run a local training in the container
* **(C.12)** The startup kits shall be distributed to the respective sites by means outside the system.
* **(C.13)** The startup kits shall enable listing installed system and python packages.

#### 3.2.4 Training Loop

* **(D.1)** The system shall be able to execute ApC (serving the interfaces of the system) performing the actual training rounds on the nodes.
* **(D.2)** The SLF shall provide means to run ApC locally mimicking a swarm training to sufficient extent so that developers can verify that newly developed code is “swarm compatible” in separate threads (“simulation mode”).
* **(D.3)** The SLF shall provide means to run ApC locally mimicking a swarm training to sufficient extent so that developers can verify that newly developed code is “swarm compatible” in separate processes (“proof-of-concept mode”).
* **(D.4)** The SLF shall provide the means to initiate a training ("admin console").
* **(D.5)** The SLF shall, when a training is initiated, distribute the ApC when a training is started
* **(D.6)** The SLF shall, when a training is initiated, designate a starting node
* **(D.7)** The SLF shall receive the initial model/checkpoint from the starting node
* **(D.8)** The SLF shall provide the initial model/checkpoint to all client nodes (send via the server)
* **(D.9)** The SLF shall receive models/checkpoints (including clients' reported performance) after each training round
* **(D.10)** The SLF shall provide models/checkpoints (including clients' reported performance) after each training round to the aggregating node (send via the server)
* **(D.11)** The SLF shall receive the aggregated model/checkpoint from the aggregating node
* **(D.12)** The SLF shall provide the aggregated model/checkpoint to all client nodes (send via the server)
* **(D.13)** The starting node shall provide an initial model/checkpoint to the SLF (send via the server)
* **(D.14)** The aggregating node of the current round shall receive models/checkpoints (including clients' reported performance)
* **(D.15)** The aggregating node of the current round shall compute an aggregated model/checkpoint after a defined number of model updates has been gathered
* **(D.16)** The aggregating node of the current round shall provide the aggregated model/checkpoint to the SLF (send via the server)
* **(D.17)** The server shall forward models/checkpoints from the staring node and aggregating nodes to the client nodes
* **(D.18)** The server shall forward models/checkpoints to the aggregating node
* **(D.19)** The server shall trigger training on the client nodes
* **(D.20)** The aggregator shall keep track of received training results from clients for the current round
* **(D.21)** The aggregator shall aggregate when a configured number of clients has provided results
* **(D.22)** The client nodes shall store the received ApC when a training is started
* **(D.23)** The client nodes shall receive models/checkpoints from the SLF
* **(D.24)** The client nodes shall run one round of training using the ApC when triggered by the server
* **(D.25)** The client nodes shall provide a model/checkpoint at the end of the round (sent via the server)

#### 3.2.5 Application Code

* **(E.1)** The ApC shall decide which models/checkpoints (from which iteration) to keep (e.g., the top k model received so far based on a performance metric; both evaluated on the private validation data) in addition to the global model from the last round
* **(E.2)** The ApC shall run a number of training epochs per round of swarm training
* **(E.3)** The ApC shall respect a pre-defined training/validation/test split when and if loading data
* **(E.4)** The ApC shall provide summary statistics of the local dataset (number of training and validation data) to the SLF for logging
* **(E.5)** The ApC shall provide progress (number of total epochs trained),  to the SLF for logging
* **(E.6)** The ApC shall provide loss values to the SLF for logging
* **(E.7)** The ApC shall provide performance metric values to the SLF for logging
* **(E.8)** The system shall include ApC for the Docker+GPU preflight check.
* **(E.9)** The ApC for the Docker+GPU preflight check shall, in addition to the generic ApC requirements above, require GPU usage
* **(E.10)** The ApC for the Docker+GPU preflight check shall, in addition to the generic ApC requirements above, finish a few epochs in less than one minute
* **(E.11)** The ApC for the Docker+GPU preflight check shall, in addition to the generic ApC requirements above, not load any training data from the host system
* **(E.12)** The system shall include ApC for the swarm preflight check.
* **(E.13)** The ApC for the swarm preflight check shall, in addition to the generic ApC  requirements above, finish each round in less than one minute.
* **(E.14)** The ApC for the swarm preflight check shall, in addition to the generic ApC  requirements above, run for five rounds with two epochs each.
* **(E.15)** The ApC for the swarm preflight check shall, in addition to the generic ApC  requirements above, use checkpoints smaller than 1 MiB in size.
* **(E.16)** The ApC for the swarm preflight check shall, in addition to the generic ApC  requirements above, not load any training data from the host system

Note: typically, the Docker+GPU preflight check and the swarm preflight check use the same ApC.

#### 3.2.6 Application Code Accompanying the System

* **(F.1)** The system shall be accompanied by the ApC for the local data access preflight check.
* **(F.2)** The ApC for the local data access preflight check shall, in addition to the generic ApC requirements above, load all data to be used in the productive training.
* **(F.3)** The ApC for the local data access preflight check shall, in addition to the generic ApC requirements above, run one epoch of training using the code for productive training.
* **(F.4)** The ApC for the local data access preflight check shall, in addition to the generic ApC requirements above, indicate by its return code if an error has occurred.
* **(F.5)** The system shall be accompanied by the productive ApC.
* **(F.6)** The productive ApC shall, in addition to the generic ApC requirements above, load data as needed for the training
* **(F.7)** The productive ApC shall, in addition to the generic ApC requirements above, train a useful model on the local data for the task at hand
* **(F.8)** The productive ApC shall, in addition to the generic ApC requirements above, be executable both for swarm training and for a given number of epochs of local training.

Note: typically, the ApC for the data access preflight check is the same as the productive ApC, configured to run locally for a single epoch. Evaluation of the final model using additional evaluation/test data happens outside this system

#### 3.2.7 Application Code Documentation

* **(G.1)** The ApC for the Docker+GPU preflight check shall be accompanied by documentation for IT security officers and privacy officers to enable code review.
* **(G.2)** The ApC for the local data access preflight check shall be accompanied by documentation for IT security officers and privacy officers to enable code review.
* **(G.3)** The ApC for the swarm preflight check shall be accompanied by documentation for IT security officers and privacy officers to enable code review.
* **(G.4)** The productive ApC shall be accompanied by documentation for data contributors and clinical technicians on what the overall contents of the data is meant to be (e.g., dynamic contrast-enhanced breast MRI data with metadata malignancy yes/no).
* **(G.5)** The productive ApC shall be accompanied by documentation for data contributors and clinical technicians on which aggregated characteristics of the data are to be collected and communicated outside the system
* **(G.6)** The productive ApC shall be accompanied by documentation for data contributors and clinical technicians on documentation of the expected data format (image file format and contents, folder structure, table format and contents)
* **(G.7)** The data mentioned in G.5  shall be requested from data contributors by means outside the system.
* **(G.8)** The productive ApC shall be accompanied by documentation for IT security officers and privacy officers to enable code review.

#### 3.2.8 Preprocessing

* **(H.1)** The ApC for productive training shall be accompanied by instructions (in text form or scripts) how to convert existing clinical data (image data and annotations) to the format (see expected data format) required for the ApC to use the data
* **(H.2)** The ApC for productive training shall be accompanied by instructions (in text form or scripts) how to define a training/validation/test split
* **(H.3)** These steps happen beforehand outside the system.
* **(H.4)** Data contributors/clinical technicians shall ensure that the data has been anonymized prior to entering it in the system by means outside the system, this is not verified by the system.

#### 3.2.9 Logging, Monitoring

* **(J.1)** The server node shall log, as plain text in a file on the host, connections to clients
* **(J.2)** The server node shall log, as plain text in a file on the host, a unique identifier of the training job
* **(J.3)** The server node shall log, as plain text in a file on the host, training rounds started by each client node
* **(J.4)** Each client node shall log, as plain text in a file on the host, connection to the server
* **(J.5)** Each client node shall log, as plain text in a file on the host, tasks (training round) assigned from other nodes
* **(J.6)** Each client node shall log, as plain text in a file on the host, output received from the ApC for logging
* **(J.7)** Each client node shall log, as plain text in a file on the host, in which rounds it acts as the aggregator and who is the subsequent aggregator
* **(J.8)** The system does not need to provide means for displaying this information, since log files and ApC can be read on the host.

#### 3.2.10 Execution Environment, Network Communication

* **(K.1)** The system shall encapsulate the SLF and the ApC in a way that it can be prevented from accessing on-site resources not needed for its operation.
* **(K.2)** This encapsulation shall be achieved by running the system in Docker containers and on a dedicated physical or virtual machine.
* **(K.3)** The system shall be accompanied by hardware specifications and instructions to set up the host system on which containers are to be run.
* **(K.4)** The system shall provide a way to obtain the container images in which the SLF and ApC are executed.
* **(K.5)** Provision of container images shall be achieved by a script to pull the respective Docker image from a container registry to which the images are pushed by means outside the system.
* **(K.6)** The system shall provide read/write access to a location where the SLF can write its logs for the server and client nodes
* **(K.7)** The system shall provide read access to the ApC to specifically selected data for swarm training for client nodes
* **(K.8)** The system shall provide read/write access to a location for saved models/model checkpoints for client nodes
* **(K.9)** Data access shall be achieved by Docker mounts.
* **(K.10)** Each node shall use the authorization system of the host system.
* **(K.11)** The client node maintainer shall make sure that only the data authorized for the training are visible to the execution environment.
* **(K.12)** The system shall run in an environment where nodes can communicate with each other (across potential firewalls etc.).
* **(K.13)** This communication shall be handled via a VPN set up outside the system.
* **(K.14)** The communication between nodes in the system shall be inaccessible by external parties and protected against external interference.
* **(K.15)** A state-of-the-art VPN shall be considered sufficient to ensure this.
* **(K.16)** Additional encryption of traffic between nodes in the swarm is optional (i.e., we currently accept that the VPN server and the swarm server can read what is communicated between nodes)

#### 3.2.11 Authentication

* **(L.1)** The system shall provide means for authentication of connected nodes.
* **(L.2)** This shall be achieved by certificates for all nodes issued by a swarm-specific CA.
* **(L.3)** The server node shall only admit authorized client nodes to the swarm.
* **(L.4)** Client nodes shall connect only to an authorized server.
* **(L.5)** Only authorized swarm training initiators (admin nodes) shall be able to deploy trainings.
* **(L.6)** The system shall issue the required CA and node certificates (to be distributed as part of the startup kits).
* **(L.7)** Authentication between the hosts and the VPN shall be handled outside of the system.

#### 3.2.12 Code Availability

* **(M.1)** The source code of the system shall be available for inspection by all interested stakeholders (in particular IT security and privacy officers, paper reviewers)
* **(M.2)** The ApC shall be available for inspection by all interested stakeholders (in particular IT security and privacy officers, paper reviewers)
* **(M.3)** The code shall be open source (currently not requiring specific licensing)

#### 3.2.13 Swarm Learning Framework Documentation

* **(N.1)** The system shall be accompanied by documentation on how to onboard new nodes to the swarm (i.e., how to build container images and how to build and distribute startup kits)
* **(N.2)** The system shall be accompanied by documentation on how to set up the swarm (i.e., how to obtain the startup kit and start the server)
* **(N.3)** The system shall be accompanied by documentation on how to set up nodes for the swarm (i.e., hardware requirements, VPN connection and what software to install on the SL nodes)
* **(N.4)** The system shall be accompanied by documentation on how to set up a node for a specific training (i.e., how to obtain the startup kit, the required data format,  how to run the client node and local data access preflight checks for a specific training (loading of local data, executing the training step, aggregating updates) without interaction with the other nodes of the swarm, and how to start the ApC configured for the respective site)
* **(N.5)** The system shall be accompanied by documentation on how to read console/log output from the actual training to verify that it has been started successfully and to monitor performance
* **(N.6)** The system shall be accompanied by documentation on an interface definition for the ApC

### 3.3 Input and Output

The SLF shall write the information it logs to files on the SL nodes.
Data for training/validation/testing are preprocessed outside of the framework; since the data format is specific for the application code, the SLF does not provide means to check the data.

1. The system shall enable inclusion of data by providing respective code and configuration in the ApC.
2. Documentation on how to provide data to the swarm shall be provided by the responsible author of the ApC, and thus shall not be part of the SLF.
3. Any contributor of data shall provide aggregated characteristics of the data.

### 3.4 Interfaces to Other Systems

The SLF shall orchestrate the actual training via task-specific ApC via interfaces provided by NVFlare.

### 3.5 Alarms

(None)

### 3.6 Security Requirements

The system shall run via a VPN; for authentication the requirements above.
Other cybersecurity aspects will be addressed in later use cases.

### 3.7 User Interface Requirements

Swarm learning is not an interactive process, there is no dedicated GUI. The SLF shall be started by starting Docker containers, e.g., from a shell. Progress information shall be displayed via the NVFlare Dashboard, accessible by web browsers.

### 3.8 Data Definition and Database Requirements

(None)

### 3.9 Installation and Acceptance Requirements

See requirements B.1 to B.14 and C.1 to C.13

### 3.10 Maintenance Requirements

Not applicable, fresh Docker images are generated per training.

### 3.11 IT-Network Requirements

See System Requirements \> Network.

### 3.12 User Maintenance Requirements

(None)

### 3.13 Regulatory Requirements

(None)
