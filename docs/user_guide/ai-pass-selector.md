<!--------------------------------------------------------------------------------------------------
Copyright 2024 Munich Quantum Software Stack Project

Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://github.com/Munich-Quantum-Software-Stack/MQSS-Quantum-Compilation-Suite/blob/develop/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
----------------------------------------------------------------------------------------------------->

# AI Pass Selector

The `AI/` directory contains a reinforcement-learning (RL) based **pass selector**. Instead of
running a fixed pipeline, an RL agent looks at a Quake circuit, picks one gate-level rewrite pass at
a time, and is rewarded when the pass reduces the circuit's gate count and depth. Training,
evaluation and inference are all done in C++ with
[LibTorch](https://pytorch.org/cppdocs/) on top of the MLIR/CUDA-Q Quake dialect.

This page is about building, running and extending the selector. For the theory (circuit tensor
encoding, network architectures, the RL algorithms and annotated research-paper notes), see
`AI/README.md` in the repository.

## Directory layout

| Path                                   | Contents                                                                                                        |
|----------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `AI/ai_pass_selector.cpp`              | Main executable: dataset/agent info, training, evaluation and single-circuit inference.                         |
| `AI/convert_qasm_dataset_to_quake.cpp` | Converts a dataset from `AI/Datasets/Qasm/<name>` to `AI/Datasets/Quake/<name>`.                                |
| `AI/convert_quake_testsuite_to_tikz.cpp` | Renders the `passtest`, `tensortest` and `randomtest` suites into LaTeX/TikZ (`AI/Datasets/Latex`).           |
| `AI/include`, `AI/src`                 | The `ai_core` library: `Environment/`, `NeuralNetworks/` (including `Agents/{A2C,PPO,SDSAC,ACER}`), `Testsuites/`, `Utils/`. |
| `AI/legacy`                            | The gate-pattern Quake pass library that makes up the agents' action space (see *Legacy pass library* below). |
| `AI/Agents`                            | Trained agent weights (`*.pt`), e.g. `ppo-mq28-hybrid-actor.pt`.                                                |
| `AI/Datasets`                          | `Qasm/`, `Quake/`, rendered `Latex/` test suites and `Evaluations/` (JSON results per agent).                    |
| `AI/Circuits`                          | Small example circuits and their optimized counterparts.                                                        |
| `AI/Scripts`                           | Python helpers, e.g. `create_chemistry_dataset.py`. `legacy-build/` holds the pre-merge build scripts.         |
| `AI/ResearchPapers`                    | The papers referenced in `AI/README.md`.                                                                        |

## Building

The selector is **off by default** and is enabled with the CMake option `BUILD_MQSS_AI`. It needs
the following on top of the regular suite dependencies (see [Build](build.md)):

- **LibTorch**, unpacked into `AI/external/libtorch` (this directory is git-ignored). Pick the CPU
  or CUDA build that matches your machine.
- **yaml-cpp** (`find_package(yaml-cpp CONFIG)`), used to read the random-circuit-generator
  statistics.
- The **CUDA-Q MLIR runtime** library (`cudaq-mlir-runtime`). CMake warns if it can't find it.

```bash
# 1. Configure and build the suite as usual.
make mqss-opt
make target

# 2. Re-configure with the AI selector enabled and build its targets.
cmake -B build -DBUILD_MQSS_AI=ON
ninja -C build ai_pass_selector convert_qasm_dataset_to_quake convert_quake_testsuite_to_tikz
```

The executables end up in `build/AI/`. Dataset, agent and circuit directories are compiled in as
absolute paths (`AI_DATASET_DIR`, `AI_AGENTS_DIR`, `AI_CIRCUITS_DIR`), so you can run the binaries
from any directory.

```{note}
In `Release` builds, `ai_core` is compiled with `-Ofast -march=native`. Binaries built this way may
not run on a different CPU.
```

## Running

All options are passed as `key=value` pairs. `./ai_pass_selector --help` lists every key along with
its type and default value.

```bash
cd build/AI

# Print information about a dataset (including the statistics used by the random circuit generator)
./ai_pass_selector dataset=Chemistry info=true

# Optimize a single circuit with a trained agent
./ai_pass_selector agent=ppo-mq28-hybrid circuit=/path/to/circuit.qke output=/path/to/optimized.qke

# Train an agent (random circuits drawn from the dataset's statistics)
./ai_pass_selector agent=a2c-mq28-tcn dataset=Chemistry

# Evaluate an agent on a dataset (optionally on a random sample of N circuits)
./ai_pass_selector agent=a2c-mq28-tcn dataset=Chemistry evaluate=true evaluation_sample=200
```

What the selector does depends on which keys you set:

| Keys set                           | Mode                                                                                                  |
|------------------------------------|-------------------------------------------------------------------------------------------------------|
| `info=true`                        | Prints parameters plus circuit, dataset and agent info, then exits.                                   |
| `agent` and `circuit`              | Inference: applies the agent's chosen passes to `circuit` and writes the result to `output`.          |
| `agent` and `dataset`, no circuit  | Training, or evaluation when `evaluate=true`.                                                         |

### Agent naming

Agents are named `<algorithm>-mq<max-qubits>-<architecture>`:

- `<algorithm>`: `a2c` (Advantage Actor-Critic), `ppo` (Proximal Policy Optimization), `sdsac`
  (Stable Discrete Soft Actor-Critic) or `acer` (Actor-Critic with Experience Replay).
- `<max-qubits>`: the largest circuit width the agent's observation tensor can represent.
- `<architecture>`: `tcn` (Temporal Convolutional Network), `lstm` (bidirectional LSTM) or
  `hybrid` (TCN followed by LSTM).

Weights are stored in `AI/Agents` as `<agent>-actor.pt` plus the critic files for the algorithm:
`-critic.pt` for A2C, PPO and ACER, and `-critic_Q1.pt`, `-critic_Q2.pt` and `-alpha.pt` for
SD-SAC.

### Main parameters

| Key                                                    | Default        | Meaning                                                                        |
|--------------------------------------------------------|----------------|--------------------------------------------------------------------------------|
| `agent`                                                | `a2c-mq28-tcn` | Agent to train, evaluate or run.                                               |
| `dataset`                                              | `Chemistry`    | Dataset in `AI/Datasets/Quake` (training statistics and evaluation circuits).  |
| `circuit` / `output`                                   | empty          | Input and output `.qke` files for inference.                                   |
| `evaluate`, `evaluation_sample`                        | `false`, `0`   | Evaluate instead of train; `0` means the whole dataset.                        |
| `device`                                               | CUDA if available | Torch device.                                                               |
| `nr_episodes`                                          | `1000001`      | Number of training episodes.                                                   |
| `max_steps_per_episode`                                | `256`          | Episode truncation limit.                                                       |
| `max_steps_no_change`, `max_steps_same_action`         | `32`, `8`      | Termination criteria (see *Environment and reward*).                            |
| `discount_factor`, `gae_hyperparameter`                | `0.995`, `0.96`| Discount factor and GAE lambda.                                                 |
| `actor_learning_rate`, `critic_learning_rate`          | `1e-3`, `5e-3` | Learning rates. Optimizers are chosen with `actor_optimizer_idx` and `critic_optimizer_idx`. |
| `entropy_coefficient`                                  | `0.01`         | Entropy bonus.                                                                 |
| `probability_max_qubits`, `training_max_nr_gates`      | `0.4`, `5000`  | Shape of the randomly generated training circuits.                             |
| `save_agent_every_ith_episode`                         | `10`           | Checkpoint frequency.                                                          |

Algorithm-specific keys are prefixed with `ppo_`, `sdsac_` or `acer_`. See `--help` for the full
list.

## Environment and reward

`QuantumCircuitEnvironment` (`AI/include/Environment/quantum_circuit_environment.hpp`) wraps a
Quake module:

- **Observation:** an `[N, MAX_QUBITS + 20]` tensor with one row per gate. Each row holds
  the qubit roles (control -1, target +1), a signed one-hot gate identity (adjoint -1) and up
  to three rotation parameters.
- **Action:** the index of one pass in `PASS_FUNCTIONS` (`AI/include/Utils/passes_utils.hpp`).
- **Reward:** after a pass that changed the circuit, the relative improvement in gate count and
  depth, normalized by the original circuit:

```{math}
r_t = \frac{G_{t-1}-G_t}{G_0} + \frac{D_{t-1}-D_t}{D_0}
```

Here G is the gate count, D is the depth, and the subscript is the step index.

- **Termination:** the episode ends when no pass has changed the circuit for more than
  `max_steps_no_change` steps, or when the same non-changing pass has been picked more than
  `max_steps_same_action` times in a row. It is truncated after `max_steps_per_episode` steps.

Training circuits come from a random quantum circuit generator. It is parameterized by
per-dataset statistics in `AI/include/Environment/StatisticsForRQCG/<Dataset>Statistics.yaml`.
Run `info=true` on a new dataset to generate these numbers, so you don't have to train directly
on (and overfit to) the dataset itself.

## Action space

The action space has 65 passes, grouped here by kind:

| Group             | Passes                                                                                                                                                  |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cancellations     | `ZeroRxToId`, `ZeroRyToId`, `ZeroRzToId`, `CxCxToId`, `CyCyToId`, `CzCzToId`, `XXToId`, `YYToId`, `ZZToId`, `SSdgToId`, `SdgSToId`, `TTdgToId`, `TdgTToId`, `HHToId` |
| Merges            | `RxRxToRx`, `RyRyToRy`, `RzRzToRz`, `SSToZ`, `SdgSdgToZ`, `SSSToSdg`, `SdgSdgSdgToS`, `SZToSdg`, `ZSToSdg`, `SdgZToS`, `ZSdgToS`, `TTToS`, `CxCxCxToSwap` |
| Hadamard rewrites | `HXHToZ`, `HZHToX`, `XHZToH`, `ZHXToH`, `HRxHToRz`, `HRzHToRx`, `HCxHToCz`, `HCzHToCx`, `HCrxHToCrz`, `HCrzHToCrx`                                       |
| Commutations      | `XHToHZ`, `HXToZH`, `YHToHY`, `HYToYH`, `ZHToHX`, `HZToXH`, `CxZToZCx`, `ZCxToCxZ`, `CxXToXCx`, `XCxToCxX`, `CxRxToRxCx`, `RxCxToCxRx`                    |
| Decompositions    | `ReverseCx`, `XToHZH`, `ZToHXH`, `RxToHRzH`, `RzToHRxH`, `CxToUpperHCzH`, `CxToLowerHCzH`, `CzToUpperHCxH`, `CzToLowerHCxH`, `CrxToHCrzH`, `CrzToHCrxH`, `SdgToSSS`, `SToSdgSdgSdg`, `SToTT`, `SwapToLowerCxCxCx`, `SwapToUpperCxCxCx` |

Decompositions make the circuit bigger on purpose. They let the agent expose cancellations that no
single reducing pass can reach.

```{important}
The trained agents' output layer has exactly one entry per pass, in `PASS_FUNCTIONS` order. Adding, removing or
reordering entries in `PASS_FUNCTIONS` invalidates every agent in `AI/Agents`.
```

### Legacy pass library

The upstream suite implements its transforms as dialect-agnostic `Common*` passes (see
[Passes](passes.md)). The agents were trained on the finer-grained, one-rewrite-per-pass Quake
passes listed above, so those passes are kept self-contained in `AI/legacy/`:

- `AI/legacy/include/Passes/{Transforms,Decompositions,Cancellations}.td` hold the TableGen pass
  declarations (namespace `mqss::opt`).
- `AI/legacy/lib/Passes/...` holds the pass implementations. `lib/Support` contains the shared
  rewrite helpers (`mlir_utils`, Quake DAG). `lib/Interfaces` and `lib/Optimizer` hold the QASM
  import and optimization pipelines.
- `AI/legacy/CMakeLists.txt` builds the `SupportQuake`, `Interfaces`, `MLIRPasses` and `Optimizer`
  libraries. They are built only when `BUILD_MQSS_AI=ON`.

## Datasets and test suites

| Dataset      | Quake circuits | Notes                                                                                           |
|--------------|---------------:|-------------------------------------------------------------------------------------------------|
| `Chemistry`  | 1548           | Quantum-chemistry circuits (qiskit-nature + PySCF, several molecules, bases and mappers), up to 28 qubits. Generated by `AI/Scripts/create_chemistry_dataset.py`. |
| `MQTBench`   | 1943           | MQT Bench circuits, up to 130 qubits.                                                           |
| `Passtest`   | 65             | One targeted circuit per pass, for checking that every action rewrites what it should.          |
| `Tensortest` | 20             | Circuits that test the tensor encoding.                                                         |
| `Randomtest` | 20             | Circuits sampled from the random circuit generator.                                             |

To add a dataset, put `.qasm` files in `AI/Datasets/Qasm/<Name>` and run
`./convert_qasm_dataset_to_quake <Name>`. To render a test suite to TikZ, run
`./convert_quake_testsuite_to_tikz passtest|tensortest|randomtest`.

## Results

Twelve agents with `MAX_QUBITS = 28` have been trained on the Chemistry statistics. Each was then
evaluated on all 1548 Chemistry circuits. The full per-circuit results, including the selected
pass sequences, are in `AI/Datasets/Evaluations/Evaluation_Chemistry_<agent>_full.json`.

| Agent               | Avg. gate reduction | Avg. depth reduction | Circuits improved |
|---------------------|--------------------:|---------------------:|------------------:|
| `ppo-mq28-hybrid`   | 46.24               | 1.66                 | 954               |
| `sdsac-mq28-hybrid` | 46.24               | 1.66                 | 954               |
| `a2c-mq28-tcn`      | 45.89               | 1.60                 | 886               |
| `a2c-mq28-hybrid`   | 42.64               | 1.46                 | 609               |
| `a2c-mq28-lstm`     | 42.03               | 1.39                 | 524               |
| `acer-mq28-tcn`     | 17.76               | 0.51                 | 122               |
| `ppo-mq28-tcn`      | 11.78               | 0.11                 | 112               |
| `ppo-mq28-lstm`     | 11.78               | 0.11                 | 112               |
| `acer-mq28-hybrid`  | 0                   | 0                    | 0                 |
| `acer-mq28-lstm`    | 0                   | 0                    | 0                 |
| `sdsac-mq28-lstm`   | 0                   | 0                    | 0                 |
| `sdsac-mq28-tcn`    | 0                   | 0                    | 0                 |

Reductions are absolute gate and depth reductions averaged per circuit. Matching rows (for example
`ppo-mq28-hybrid` and `sdsac-mq28-hybrid`) aren't duplicates: the agents chose different pass
sequences on some circuits but reached the same final circuit every time. Training on the MQTBench
statistics is still an open problem: its large circuits produce observation tensors too big to fit
in memory.
