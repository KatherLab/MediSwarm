**Project ODELIA**

# MediSwarm Software Test Report Template

Document-ID: MediSwarm-TEST-RPT-Template

### References

| Document-ID                 | Title                                 |
| :----                       | :----                                 |
| MediSwarm-SPEC-TST          | MediSwarm Software Test Specification |
| MediSwarm-TEST-RPT-Template | MediSwarm Test Report Template        |

| MediSwarm Version Tested |       |
| :----                    | :---- |
| Date                     |       |
| Tester                   |       |

## 1 Specifications of Test System

| Specification fulfilled? |       |
| :----                    | :---- |

## 2 Code Review

### 2.1 MediSwarm Productive Code

| File(s)                                                                              | To be verified                                                                                                 | Reviewed |
| :----                                                                                | :----                                                                                                          | :----    |
| docker\_config/master\_template.yml                                                  | Startup scripts for all types of nodes use the same container                                                  |          |
| files in application/jobs/minimal\_training\_pytorch\_cnn/app/custom/ and subfolders | Application code for swarm preflight check does not access files in /data inside container                     |          |
| docker\_config/master\_template.yml                                                  | Startup scripts for admin, client, and server nodes do not provide unnecessary resources to containers started |          |
| docker\_config/master\_template.yml                                                  | Startup scripts for admin, client, and server nodes use Docker to start the respective nodes                   |          |
| docker\_config/master\_template.yml                                                  | Startup scripts for admin, client, and server nodes use Docker mount for storage locations                     |          |

### 2.2 MediSwarm Tests

| File                                                                          | Purpose                                                                                                                   | Reviewed |
| :----                                                                         | :----                                                                                                                     | :----    |
| runIntegrationTests.sh                                                        | orchestrates the NVFlare integration tests, allows running NVFlare unit tests                                             |          |
| getVersionNumber.sh                                                           | utility script providing current version string                                                                           |          |
| tests/provision/dummy\_project\_for\_testing.yml                              | definition of the swarm nodes used in the integration tests                                                               |          |
| tests/integration\_tests/\_run\_minimal\_example\_standalone.sh               | runs minimal example application code outside NVFlare when called inside NVFlare docker container                         |          |
| tests/integration\_tests/\_run\_minimal\_example\_simulation\_mode.sh         | runs minimal example application code in NVFlare simulation mode when called inside NVFlare docker container              |          |
| tests/integration\_tests/\_run\_minimal\_example\_proof\_of\_concept\_mode.sh | runs minimal example application code in NVFlare proof-of-concept mode when called inside NVFlare docker container        |          |
| tests/unit\_tests/\_run\_nvflare\_unit\_tests.sh                              | runs NVFlare unit tests when called inside NVFlare docker container                                                       |          |
| tests/integration\_tests/\_run\_3dcnn\_simulation\_mode.sh                    | runs example application code in NVFlare simulation mode when called inside NVFlare docker container                      |          |
| tests/integration\_tests/\_attemptAdminConsoleLogin.exp                       | script (expect) called to interact with admin console when attempting to log in from startup kit with invalid certificate |          |
| tests/integration\_tests/\_submitDummyTraining.exp                            | script (expect) called to interact with admin console to submit minimal example application code as swarm job             |          |
| tests/integration\_tests/\_submit3DCNNTraining.exp                            | script (expect) called to interact with admin console to submit example application code as swarm job                     |          |

## 3 Automatic Tests

### 3.1 Test Output

* [NVFlare unit tests](./TestReport_NVFlareUnitTests.txt)
* [MediSwarm integration tests](./TestReport_MediSwarmIntegrationTests.txt)

### 3.2 Test Results

#### 3.2.1 NVFlare Unit Test

| Test                      | Result | Comment                 |
| :----                     | :----  | :----                   |
| run\_nvflare\_unit\_tests |        | Total line coverage: …% |

#### 3.2.2 MediSwarm Integration Test

| Test                                                | Result | Comment                                                               |
| :----                                               | :----  | :----                                                                 |
| check\_files\_on\_github                            |        |                                                                       |
| run\_dummy\_training\_standalone                    |        |                                                                       |
| run\_dummy\_training\_simulation\_mode              |        |                                                                       |
| run\_dummy\_training\_poc\_mode                     |        |                                                                       |
| create\_synthetic\_data                             | N/A    | required for subsequent steps, does not test functionality            |
| run\_3dcnn\_simulation\_mode                        |        |                                                                       |
| create\_startup\_kits\_and\_check\_contained\_files |        |                                                                       |
| start\_registry\_docker\_and\_push                  |        |                                                                       |
| run\_container\_with\_pulling                       |        |                                                                       |
| kill\_registry\_docker                              | N/A    | resource cleanup before subsequent steps, does not test functionality |
| run\_list\_licenses                                 |        |                                                                       |
| run\_docker\_gpu\_preflight\_check                  |        |                                                                       |
| run\_data\_access\_preflight\_check                 |        |                                                                       |
| run\_data\_access\_preflight\_check\_log\_details   |        |                                                                       |
| cleanup\_synthetic\_data                            | N/A    | resource cleanup before subsequent steps, does not test functionality |
| run\_data\_access\_preflight\_check\_without\_data  |        |                                                                       |
| create\_synthetic\_data                             | N/A    | required for subsequent steps, does not test functionality            |
| run\_3dcnn\_local\_training                         |        |                                                                       |
| verify\_wrong\_certificates\_are\_rejected          |        |                                                                       |
| start\_server\_and\_clients                         | N/A    | required for subsequent steps, does not test functionality            |
| run\_dummy\_training\_in\_swarm                     |        |                                                                       |
| run\_3dcnn\_training\_in\_swarm                     |        |                                                                       |
| kill\_server\_and\_clients                          | N/A    | cleanup, does not test functionality                                  |
| cleanup\_temporary\_data                            | N/A    | cleanup, does not test functionality                                  |
