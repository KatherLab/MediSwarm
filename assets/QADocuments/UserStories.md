**Project ODELIA**

# MediSwarm Use Cases and User Stories

Document-ID: MediSwarm-SPEC-USR

### References

| Document-ID | Title  |
| :----       | :----  |
|             | (none) |

## 1 Introduction

This document describes stakeholder roles interacting with the swarm learning framework and requirements of the different stakeholders phrased as user stories. These serve as the basis for software requirements. User stories are written in the form “As a \<role\>, I can \<capability\>, so that \<I have a benefit\>.”

## 2 Abbreviations

* **SL** swarm learning
* **SLF** swarm learning framework
* **ApC** training/experiment code (application code)

## 3 Use Case: ODELIA Swarm Learning for Publication

Task: Classification wrt. malignant/benign/no with (architecturally) predefined AI model lesion on ODELIA data provided by partners.
Assumptions for Use Case 1:

* Inclusion criteria for data are communicated in advance
* Data are exported from clinical systems, curated, and converted as preprocessing
* Persons involved with running the code are ODELIA consortium partners (plus institutional privacy/ethics/… officers who are not part of ODELIA)
* Initiators of SL experiments are trusted to only run the code previously agreed on, i.e. code is reviewed by swarm participants beforehand

## 4 User Roles/Personas

* **Data Contributor** (clinician) who contributes Breast MRI data
* **Clinical Technician** who collects, exports, converts, imports data
* **Swarm Training Initiator** technician/computer scientist starting an SL experiment
* **Algorithmic Researcher** (computer scientist participating in the swarm) who deals with machine learning tasks, data formats etc. but does not necessarily understand all the medical background
* **IT Security Officer** approving use of SL nodes on site
* **Privacy Officer** approving use of patient data for swarm training
* **Ethics Officer** approving use of patient data for swarm training
* **Client node maintainer** DevOps Engineer maintaining and operating one SL node
* **Swarm maintainer** DevOps Engineer operating the swarm
* **Paper Reviewer** requesting source code

Note: there may be persons in multiple roles.

## 5 User Stories

#### 5.1 Data Contributor

* **(A.1)** As a data contributor, I can contribute MRIs and diagnostic information, so that the resulting tool perspectively helps with my diagnostic work.
* **(A.2)** As a data contributor, I can contribute MRIs and diagnostic information to the model development without the data leaving my institution, so that the data can be used by the swarm participants.
* **(A.3)** As a data contributor, I can define test data to be withheld so that I can evaluate performance of the resulting tool on specific groups of patients.

#### 5.2 Clinical Technician

* **(B.1)** As a clinical technician, I can convert MRIs and diagnostic information from routine data storage to the required format so that it can be used for SL.
* **(B.2)** As a clinical technician, I can verify that the converted data can be used for the SL task so that I can verify I am done with my conversion work.
* **(B.3)** As a clinical technician, I can verify that the converted data satisfies privacy requirements (pseudonymization/anonymization) according to institutional requirements and applicable laws.

#### 5.3 Swarm Training Initiator

* **(C.1)** As a swarm training initiator, I can obtain relevant metadata about the available datasets so that I can properly plan an experiment: number of participating sites/nodes, number of contributed cases, characterization of cases, …
* **(C.2)** As a swarm training initiator, I can find out when the swarm is available so that I know when I can start a swarm learning experiment
* **(C.3)** As a swarm training initiator, I can start an SL experiment, so that a given predefined AI model is trained in the swarm.
* **(C.4)** As a swarm training initiator, I can monitor whether the SL experiment proceeds technically and when it has finished so that I can follow up on the results.

#### 5.4 Algorithmic Researcher

* **(D.1)** As an algorithmic researcher, I can monitor the progress and intermediate results of the training, so that I can check the training process for plausibility, assess model performance, identify outlier sites, and have more confidence in the resulting model.
* **(D.2)** As an algorithmic researcher, I can assess the performance of the individual sites to understand if there are outlier sites.
* **(D.3)** As an algorithmic researcher, I can obtain the swarm-trained model so that I can use it further.
* **(D.4)** As an algorithmic researcher, I can run inference using the swarm-trained model so that I can evaluate and assess it further.
* **(D.5)** As an algorithmic researcher at a site with local data, I can run a non-swarm training on the local data only (in collaboration with the client node maintainer), so that I can compare its performance to the swarm-trained model.
* **(D.6)** As an algorithmic researcher developing own models, I can check in advance that my application code is working properly and is swarm-compatible.
* **(D.7)** As an algorithmic researcher I can provide and submit my own ApC so that I can train a model in the swarm without extending the SLF.
* **(D.8)** As an algorithmic researcher, I can define a training/validation/test split so that I can plan experiments.
* **(D.9)** As an algorithmic researcher, I can obtain experiment metadata and performance results so that I can write a publication.
* **(D.10)** As an algorithmic researcher, I can ensure archival of private data related to publication so that principles of good scientific practice are adhered to.
* **(D.11)** As an algorithmic researcher, I can write an SL publication so that my work gets funded as an SL project.

#### 5.5 IT Security Officer

* **(E.1)** As an IT security officer, I can confirm that the SL framework does not allow successful attacks against itself (assuming that the initiator of a swarm training only runs the code agreed on) so that I can approve usage.
* **(E.2)** As an IT security officer, I can confirm that the SL framework does not allow successful attacks against other infrastructure (assuming that the initiator of a swarm training only runs the code agreed on) so that I can approve usage.
* **(E.3)** as an IT security officer, I have access to the code and documentation so that I can review it.
* **(E.4)** As an IT security officer, I can be certain that no externals can access the swarm communication so that I can rule out interference.

#### 5.6 Privacy Officer

* **(F.1)** As a privacy officer, I can confirm that SL training does not leak data locally supplied for training/test/validation (assuming that the initiator of a swarm training only runs the code agreed on) so that I can legally approve participation.
* **(F.2)** as a privacy officer, I have access to code and documentation so that I can review it.

#### 5.7 Ethics Officer

* **(G.1)** As an ethics officer, I can confirm that all data provided by my node have the respective clearance for the training in question so that I can legally approve participation.

#### 5.8 Client Node Maintainer

* **(H.1)** As a client node maintainer, I can install the SL framework in my network, so that my colleagues can use it.
* **(H.2)** As a client node maintainer, I can check in advance that my node is working and my data makes sense for the training so that I am confident my node will provide a meaningful contribution
* **(H.3)** As a client node maintainer, my SL node can communicate with the other nodes via network so that my colleagues can use it
* **(H.4)** As a client node maintainer, I can make sure I’m connecting to the correct swarm so that I’m collaborating with the right partners.
* **(H.5)** As a client node maintainer, I can start a swarm node so that my colleagues can use it.
* **(H.6)** As a node maintainer, I can check if my node actually works as part of the swarm so that I know it works
* **(H.7)** As a client node maintainer, I can monitor the proper function of my node.

#### 5.9 Swarm Maintainer

* **(J.1)** As a swarm maintainer, I can set up the swarm infrastructure so that I can run a swarm.
* **(J.2)** As the swarm maintainer, I can onboard new nodes, so that I can include more data in the experiment.
* **(J.3)** As a swarm maintainer, I can ensure that only authorized nodes can join the swarm after authenticating themselves so that I know who is participating.
* **(J.4)** As a swarm maintainer, I can provide what is needed for swarm operation to swarm training initiators and client node maintainers.
* **(J.5)** As a swarm maintainer, I can start a swarm server so that the swarm can run training jobs.
* **(J.6)** As a swarm maintainer, I can deploy a test job to verify that the swarm is working before productive training is started.
* **(J.7)** As a swarm maintainer, I can monitor progress of experiments, to ensure that the infrastructure achieves its primary purpose.
* **(J.8)** As a swarm maintainer, I can list the packages contained in the container image I intend to distribute so that I can verify the respective licenses allow me to do so.

#### 5.10 Paper Reviewer

* **(H.1)** As a reviewer of a publication publishing the SL framework, I can also review the source code, so that I can provide a thorough review.
