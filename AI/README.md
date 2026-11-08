# AI Pass Selector

Reinforcement-learning agents that optimize Quake circuits by picking a sequence of gate-level
rewrite passes. The user-facing documentation (building, CLI options, action space, reward, datasets
and evaluation results) is in the suite docs at
[docs/user_guide/ai-pass-selector.md](../docs/user_guide/ai-pass-selector.md). This README covers
the concepts behind the selector and collects notes on the research papers it builds on.

## Build & Usage

### Building the project

The selector is part of the main CMake project but is off by default. Build the suite the usual
way (see [docs/user_guide/build.md](../docs/user_guide/build.md)), then enable the AI targets:

```bash
make mqss-opt && make target
cmake -B build -DBUILD_MQSS_AI=ON
ninja -C build ai_pass_selector convert_qasm_dataset_to_quake convert_quake_testsuite_to_tikz
```

Extra dependencies:

- LibTorch, unpacked into `AI/external/libtorch` (git-ignored).
- `yaml-cpp`.
- The CUDA-Q MLIR runtime (`cudaq-mlir-runtime`).

The agents' action space is defined by the one-rewrite-per-pass Quake passes in [legacy](legacy).
Upstream replaced these with dialect-agnostic `Common*` passes, so they are kept here and built by
[legacy/CMakeLists.txt](legacy/CMakeLists.txt). Changing the pass list or its order in
[passes_utils.hpp](include/Utils/passes_utils.hpp) invalidates every trained agent in [Agents](Agents).

The scripts in [Scripts/legacy-build](Scripts/legacy-build) (`build_initial.sh`, `rebuild.sh`,
`start_docker.sh`) are from the pre-merge build system. They are kept for reference only, mainly
for building on the LRZ GPU nodes, where Docker isn't available.

### Running AI Pass Selector

The `ai_pass_selector` executable is built to `build/AI/ai_pass_selector`. Run it from any
directory, since the dataset and agent paths are compiled in:

```bash
cd build/AI
./ai_pass_selector --help                                        # list all key=value parameters
./ai_pass_selector dataset=Chemistry info=true                   # dataset info + RQCG statistics
./ai_pass_selector agent=ppo-mq28-hybrid circuit=in.qke output=out.qke   # optimize one circuit
./ai_pass_selector agent=a2c-mq28-tcn dataset=Chemistry          # train
./ai_pass_selector agent=a2c-mq28-tcn dataset=Chemistry evaluate=true    # evaluate
```

Agents are referenced by a three-part, dash-separated name of the form
`<rl-algorithm>-mq<max-qubits>-<nn-architecture>`.

- `<rl-algorithm>` is the reinforcement learning algorithm the agent was trained using.
  Currently the supported tokens for that value are: `a2c` for Advantage Actor-Critic, `ppo` for Proximal Policy
  Optimization, `sdsac` for Stable Discrete Soft Actor-Critic and `acer` for Actor-Critic with Experience Replay.
- `<max-qubits>` is an $\mathrm{int}$ which captures the maximum number of qubits in a givne circuit the model was
  trained to handle.
- `<nn-architecture>` selects the neural-network architecture of the agent with the currently available ones being:
  `tcn` for Temporal Convolutional Network, "lstm" for Long Short-Term Memory, and "hybrid" which is a TCN-LSTM
  combination.

## Project Concepts

### Quantum Circuits for Training

#### Tensor Encoding of a Quantum Circuit

Every quantum circuit can be viewed as a list of instructions, that means quantum gates acting on qubits.
An instruction is exactly defined by:

1. __Gate identity__ f.e. _X_, _H_, _Sdg_, etc.
2. __Qubits it acts on__ meaning both control and target qubits:
3. __Optional parameters__ such as the angles of rotation and/or unitary gates.

The tensor encoding the quantum circuit considers one such instruction as an element
with _Instruction Representation Size_ (IRS) number of values.
Here, $\mathrm{MAX\_QUBITS}$ is the maximum number of qubits a specific tensor can represent, the number of distinct
usable gate identities in quake is $17$ and the maximum number of parameters a gate may have is $3$.
An instruction representation element is constructed like this:

1. First $\mathrm{MAX\_QUBITS}$ values encode the qubits the instruction acts on.
    - If the instruction acts on qubit at index $i$ as a control qubits, the value at $i$ in the element is $-1$.
    - If the instruction acts on qubit at index $i$ as a target qubits, the value at $i$ in the element is $+1$.
    - If the instruction does not acto on qubit at index $i$, the value at $i$ in the element is $0$.
2. The next 17 values are a signed-one-hot encoding of the gate.
    - If the gate occurs as its usual type f.e. _X_, its corresponding index is set to $+1$.
    - If the gate occurs as its adjoint type f.e. _Sdg_ as the adjoint of _S_, its corresponding index is set to $-1$.
    - All other values are set to $0$.
3. The final 3 values are set as the angles of the rotation/unitary gate in order of occurence in the gate construction
   and left as $0.0$ if unused.

Therefore IRS is always: $\mathrm{IRS}=\mathrm{MAX\_QUBITS}+20$.

Finally the full quantum circuit is length-$N$ list of the above described instructions.
The full tensor is just the concatenation of the instruction representation elements and has therefore the
shape: $[N,\,\mathrm{IRS}]$.

#### Random Quantum Circuit Generator

The random quantum circuit generator, used for training to circumvent the necessity of having a training dataset and the
associated risk of overfitting, requires 2 sets of numerical attributes: a test dataset's cholesky params and gates
weights.
To see an example of how these sets of attribues look, you can view the `yaml` files
in [StatisticsForRQCG](include/Environment/StatisticsForRQCG)
like [ChemistryStatistics.yaml](include/Environment/StatisticsForRQCG/ChemistryStatistics.yaml)
or [MQTBenchStatistics.yaml](include/Environment/StatisticsForRQCG/MQTBenchStatistics.yaml).
These attribute sets can be computed using existing functions.
To do that, create a dataset with quake `.qke` circuits and place it in [AI/Datasets/Quake](Datasets/Quake).
Then run `./ai_pass_selector dataset=<your-dataset-name> info=true` from the [build/AI](../build/AI) folder.
The console should print the dataset's statistics which you can copy-paste into a corresponding `yaml` file in
the [StatisticsForRQCG](include/Environment/StatisticsForRQCG) folder.
This will create a `<your-dataset-name>Statistics.yaml` file in
the [StatisticsForRQCG](include/Environment/StatisticsForRQCG) folder and henceforth will allow you to train an agent
with the statistics of your dataset.

#### Quantum Circuit Environment

The environment, here called the `QuantumCircuitEnvironment` declared
in [quantum_circuit_environment.hpp](include/Environment/quantum_circuit_environment.hpp) and defined
in [quantum_circuit_environment.cpp](src/Environment/quantum_circuit_environment.cpp), aquires
an initial quantum circuit and defines Reinforcement-Learning specific operations on that quantum circuit.
Most importantly:

1. The environment provides a function `get_observation` to parse the underlying quantum circuit to create a tensor
   according to the encoding described above, that can be fed into an AI agent as the input.
2. The environment provides a `step` function which takes as argument an index of the action from the discrete action
   space $\mathcal{A}$, and executes the optimization pass at the corresponding index in `PASS_FUNCTIONS`
   in [passes_utils.hpp](include/Utils/passes_utils.hpp) on the underlying quantum circuit. It then computes a
   reward $r_t$ for the chosen action and whether the training episode should terminate.

The environment can terminate an episode according to a predefined termination condition, in this case if a certain pass
has been selected to often in a row without making changes, or truncate an episode if some predefined maximum allowed
number of steps has been reached.
An episode can have at most some $T_\mathrm{max}$ number of steps and the reward at step $t$ is indexed as $r_t$.

### Neural Network Architectures

#### Temporal Convolutional Networks

The original design of the Temporal Convolutional Network (TCN) architecture is described
in [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](ResearchPapers/05_NeuralNetworkArchitecturesForSequencesOfElementsTCNandLSTM.pdf).
In this project the TCN architecture is slightly modified.

1. The convolutions in the original TCN architecture are causal, meaning that there is no information leakage from
   future to past. To achieve this, the TCN uses causal convolutions, where an output at time $t$ is convolved only with
   elements from time $t$ and earlier in the previous layer. Here causality is not necessary, since the agent should see
   the entire quantum circuit at the same time. Therefore the chomping layer procedure making the convolutions causal is
   not added.
2. The TCN employs a generic residual block in place of a convolutional layer at each TCN layer. Given the
   input $\vec{x}$, the residual block performs a series of transformations $\mathcal{F}$ on the input, adds them onto
   the identity mapping and applies the activation on the sum. Using $\mathrm{ReLU}$ as the activation, the output of a
   residual block is therefore:
   $$
   \vec{o}=\mathrm{ReLU}\left(\vec{x}+\mathcal{F}\left(\vec{x}\right)\right)
   $$
   Unfortunately leaving it this way lead to exploding gradients in the actor network despite weight normalization in
   the transformation $\mathcal{F}$. This resulted in exploding values and finally $\mathrm{NaNs}$ in the policy
   function after as few as $3$ episodes. Additionally applying $\mathrm{ReLU}$ activation on the very first input
   sequence, being the encoded quantum circuit observation, would drop all negative values and make the network loose
   informatiopn about control qubits or adjoint gates, unless the very first transformation sequence $\mathcal{F}$ can
   remap that information to positive values. To adress both issues the $\mathrm{ReLU}$ activation layer was replaced
   with a layer normalization layer, leading to the modified resdiual block output function:
   $$
   \vec{o}=\mathrm{LayerNorm}\left(\vec{x}+\mathcal{F}\left(\vec{x}\right)\right)
   $$

The other aspects of the TCN architecture were left exactly as in the original paper.

#### Long Short-Term Memory Cells

General design and function of Long Short-Term Memory (LSTM) cells are described
in [Long Short-Term Memory](ResearchPapers/04_RNNandLSTMforSequencesVariantsEvaluationsAndApplications.pdf)
and [Recurrent Neural Networks and Long Short-Term Memory Networks: Tutorial and Survey](ResearchPapers/04_RNNandLSTMforSequencesVariantsEvaluationsAndApplications.pdf).
The important attribute of LSTM cells that makes them attractive for processing the quantum circuit observation is that
they can process a variable-length sequence of elements and retain information about previously seen elements while
viewing the next one.

LSTM cells can be made to process the input sequence bidirectionally, that means front to back and back to front and
concatenate both outputs to produce a bidirectionally processed output.
In this project LSTM cells will be used only with the `bidirectional` setting set to `true` because just as described
for the TCN above, we do not require the processing of the sequence to be causal.
The LSTM cells are plug-and-play layers in the PyTorch library, relieving the user from having to implement them, or
even understand in detail how they work.
How to use the PyTorch implementation of LSTM cells in one's neural network is described in
the [PyTorch LSTM Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).

Examples of how LSTM are used in Reinforcement Learning projects where a 1-dimensional sequence of elements must be
mapped to a static-size output are described
in [Language Understanding for Text-based Games using Deep Reinforcement Learning](ResearchPapers/05_NeuralNetworkArchitecturesForSequencesOfElementsTCNandLSTM.pdf)
and [Counting to Explore and Generalize in Text-based Games](ResearchPapers/05_NeuralNetworkArchitecturesForSequencesOfElementsTCNandLSTM.pdf).
In this project an additional layer normalization layer over the output element size is applied after the LSTM layer and
before the adaptive average pooling layer.

#### Hybrid Architecture

The authors
of [Deep learning coupled model based on TCN-LSTM for particulate matter concentration prediction](ResearchPapers/05_NeuralNetworkArchitecturesForSequencesOfElementsTCNandLSTM.pdf)
suggested a combined TCN-LSTM architecture for processing 1-dimensional sequences of elements and reported slightly
better performance compared to other models in their study.
This TCN-LSTM concatenation is called "Hybrid" in this work and sequentially combines the 2 neural network architectures
described above.

Libtorch implementations of all used neural network architectures can be viewed in the
file [agent_architectures.cpp](src/NeuralNetworks/agent_architectures.cpp).

### Values and Functions in Training

#### Actor Policy $\pi(a|s)$

All used agents follow an Actor-Critic structure.
This means each agent contains ar least 1 actor and at least 1 critic network, all of which are independant of one
another, that means they share no trainable parameters.
The actor network outputs a policy $\vec{\pi}(s)$ over actions.
The available actions and their count are given by `PASS_FUNCTIONS` and `NR_PASSES`
in [passes_utils.hpp](include/Utils/passes_utils.hpp).
The policy $\vec{\pi}(s)$ is a `NR_PASSES`-dimensional vector containing the actor's belief about the probabilities
which action is the best to execute next.
Because it's a probability distribution all values are in the interval $[0,1]$ and sum up to $1$.

$$
\begin{aligned}
\mathcal{S}&\widehat{=}\text{State/Observation Space} \\
\mathcal{A}&\widehat{=}\text{Action Space} \\
|\mathcal{A}|&=\mathrm{NR\_PASSES} \\
\forall s\in\mathcal{S}:
&\left\{
\forall a\in\mathcal{A}: 0\leq\pi(a|s)\leq 1
\,\land\,
\sum_{a\in\mathcal{A}}\pi(a|s)=1
\right\}
\end{aligned}
$$

#### Q-Value Estimation $Q(s,a)$

A Q-estimating critic in the discrete action setting is a critic which learns to output a Q-value vector $\vec{Q}(s)$ of
dimension $|\mathcal{A}|$.
Each element $Q(s,a)$ of this critic's output vector is a scalar describing how desirable it is to perform the
action $a$ in the state $s$.

#### State Value Function $V(s)$

The normal critic in an Actor-Critic agent learns to output a state-value function $V(s)$.
Given some observation $s$, the state-value function's output $V(s)$ is a scalar describing how desirable it is to be in
the state $s$.
Alternatively to having a critic network learning to approximate the state-value function, it can be computed as the
dot-product of the policy $\vec{\pi}(s)$ and the Q-value estimation $\vec{Q}(s)$:

$$
V(s)=\vec{\pi}(s)\cdot\vec{Q}(s)=\sum_{a\in\mathcal{A}}\pi(a|s)Q(s,a)
$$

#### Reward-to-go $R_t$

Assume an episode of training occured over $T\leq T_\mathrm{max}$ amount of time steps.
Thus the agent achieved some trajectory of $T$ rewards $r_t$ for $t\in\mathbb{N},\,t\in[1,T]$.
Given a real-numbered training hyperparameter called **discount factor** $\gamma\in(0,1]$, a training step's
Reward-to-go is defined as:

$$
R_t=\sum_{t'=t}^{T}\gamma^{t'-t}r_t
$$

#### Advantage $A(s,a)$

The advantage of actions $a$ in state $s$ is officially defined as:

$$
A(s,a)=Q(s,a)-V(s)
$$

However if the Q-value and state-value of a state and action are unknown, the advantage must be estimated using network
outputs and rewards.
The most prominent algorithm for estimating the advantage is the Generalized Advantage Estimation (GAE) proposed
in [High-Dimensional Continuous Control Using Generalized Advantage Estimation](ResearchPapers/03_ReinforcementLearningAlgorithmsSet1.pdf).
With an additional real-valued hyperparameter called **GAE hyperparameter** $\lambda\in[0,1]$, and the state-value
estimation of the standard critic, the advantage at time step $t$ computed use GAE is:

$$
A(s_t,a_t)=\sum_{t'=t}^{T}(\gamma\lambda)^{t'-t}\left(
r_{t'}+\gamma V\left(s_{t'+1}\right)-V\left(s_{t'}\right)
\right)
$$

As a result, two interesting edge-cases emerge:

$$
\begin{aligned}
\lambda=0:\quad
&A(s_t,a_t)=r_{t}+\gamma V\left(s_{t+1}\right)-V\left(s_{t}\right) \\
\lambda=1:\quad
&A(s_t,a_t)=-V\left(s_{t}\right)+\sum_{t'=t}^{T}\gamma^{t'-t}r_{t'}
\end{aligned}
$$

### Reinforcement Learning Algorithms

#### Advantage Actor-Critic

The Advantage Actor-Critic (A2C) is a reinforcement learning algorithm, in other words, an algorithm which defines how
to construct the losses of the involved neural networks and/or how to update their trainable parameters using the reward
that an action on the environment achieves and the output of the neural networks themselves.
An reinforcement learning algorithm is not to be confused with the neural network model architecture.
A2C was first proposed as its asynchronous version: the Asynchronous Advantage Actor-Critic (A3C), with multiple
actors all training independantly to reduce bias, in the
paper [Asynchronous Methods for Deep Reinforcement Learning](ResearchPapers/03_ReinforcementLearningAlgorithmsSet1.pdf).
The A3C algorithm can also be used with only 1 actor resulting in A2C which still works and sometimes performs even
better than A3C.
The Generalized Advantage Estimation (GAE) can be used as an alternative to the unmodified rewards-to-go to compute the
losses of the actor and the critic in the A2C algorithm.
A2C is one of the most basic and easiest to implement algorithms in reinforcement learning.
Its implementation is in: [a2c_trainer.cpp](src/NeuralNetworks/Agents/A2C/a2c_trainer.cpp).

#### Proximal Policy Optimization

Proximal Policy Optimization (PPO) came out in 2017 in the
paper [Proximal Policy Optimization Algorithms](ResearchPapers/03_ReinforcementLearningAlgorithmsSet1.pdf) and builds on
prior work done on A2C and the Trust Region Policy Optimization (TRPO) algorithm.
Just like A2C, PPO uses the concept of the advantage $A(s,a)$ to compute the losses of the actor and the critic, albait
using a different loss function.
The novel procedure in the TRPO and PPO algorithms is to use the old policy from a previously played
episode $\pi_\mathrm{old}(a|s)$ to update the current policy $\pi(a|s)$ in such a way that the current policy does not
diverge too far from the previous policy during the update.
This is supposed to reduce the magnitude of the updates to the policy function to prevent it from suddenly jumping to
some very undesirable policy.
To not confuse it with ACER, it is important to note that this algorithm only keeps a trajectory of the 1 previously
seen episode, then uses the policy from that previous episode to compute the loss for the current episode.
This algorithm does not keep a buffer of multiple previously played episodes.
Its implementation is in: [ppo_trainer.cpp](src/NeuralNetworks/Agents/PPO/ppo_trainer.cpp).

#### Stable Discrete Soft Actor-Critic

The Soft-Actor Critic (SAC) algorithm was proposed originally exclusively for continuous action spaces in the
paper [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](ResearchPapers/03_ReinforcementLearningAlgorithmsSet1.pdf).
Being only for continuous action spaces makes it inapropriate for this project, since the action space here is a
discrete one.
However, multiple attempts were performed to adapt the SAC to discrete action spaces, the most recent one of which:
Stable
Discrete Soft-Actor Critic (SD-SAC) from the
paper [Revisiting Discrete Soft Actor-Critic](ResearchPapers/08_ReinforcementLearningAlgorithmsSet3.pdf),
has been implemented here.
This actor-critic reinforcement learning algorithm uses a Q-value estimating critic unlike A2C and PPO which use
standard state-value function estimating critic.
The difference between standard critics and Q-estimating critics is described in the section on values and functions in
training.
The implementation of SD-SAC is in: [sdsac_trainer.cpp](src/NeuralNetworks/Agents/SDSAC/sdsac_trainer.cpp).

#### Actor-Critic with Experience Replay

The final algorithm implemented in this project is Actor-Critic with Experience Replay (ACER).
ACER was first proposed
in [Sample Efficient Actor-Critic with Experience Replay](ResearchPapers/03_ReinforcementLearningAlgorithmsSet1.pdf).
The main idea is too keep a replay memory $\mathcal{D}$ of previously played episodes and the performed action
trajectories, then replay those episodes with the current policy, resulting in possibly new action trajectories.
The old and new episode trajectories are then used to perform trust-region updates to the actor.
Just like the SD-SAC, the ACER uses Q-value estimating critics instead of standard critics.
ACER is the most complicated reinforcement learning algorithm of the 4 implemented here.
Its implementation is in: [acer_trainer.cpp](src/NeuralNetworks/Agents/ACER/acer_trainer.cpp).

### Evaluation Datasets

#### Chemistry

The Chemistry dataset is this project's unique dataset which utilizes python packages `qiskit`, `qiskit_nature`, and
`PySCF` to construct small quantum circuits for quantum chemistry simulations of various molecules in different bases
using different mappers.
This dataset can be created using the script [create_chemistry_dataset.py](Scripts/create_chemistry_dataset.py).
The quantum circuits willl initially by written in QASM, but can be translated into QUAKE by moving the constructed
dataset into [AI/Datasets/Qasm](Datasets/Qasm) and running `./convert_qasm_dataset_to_quake Chemistry` from
the [build/AI](../build/AI) folder.
The Chemistry dataset contains a grand total of $1548$ circuits with $\mathrm{MAX\_QUBITS}=28$, a maximum gate count
of $2188$ and maximum depth of $1234$.

#### MQTBench

The MQTBench dataset is described
in [MQT Bench: Benchmarking Software and Design Automation Tools for Quantum Computing](ResearchPapers/01_OptimizingQuantumComputing.pdf).
It has been downloaded from
the [previous version of the official MQTBench library](https://www.cda.cit.tum.de/mqtbench/) and baked into this
project.
The MQTBench dataset is available in its original QASM for in [AI/Datasets/Qasm/MQTBench](Datasets/Qasm/MQTBench) as
well as transformed into QUAKE in [AI/Datasets/Quake/MQTBench](Datasets/Quake/MQTBench).
The MQTBench dataset contains a grand total of $1943$ circuits with $\mathrm{MAX\_QUBITS}=130$, a maximum gate count
of $98338$ and maximum depth of $89139$.

### Current Status

With $3$ neural networks implemented to process 1-dimensional sequences of elements and $4$ reinforcement learning
algorithms implemented to train them, gives $3\cdot4=12$ trainable agents per dataset statistics.
Currently 12 agents with `MAX_QUBITS` set to $28$ have been trained using the Chemistry dataset statistics to select
passes.
MQTBench dataset statistics still cause difficulties in training due to the large circuits and corresponding extremely
large observation tensors consuming huge chunks of memory.

## Research Paper Notes

### 1. [Optimizing Quantum Computing](ResearchPapers/01_OptimizingQuantumComputing.pdf)

- __[Page 2]__ Machine learning scales to large corpora of training examples, which we expect to increase the likelihood
  of obtaining policies that generalize well. This is important because, we do not want to retrain policies too
  frequently. It is an adoption blocker.
- __[Page 37]__ Markov chain oracle outperforms the Independent Identically Distributed probability distribution oracle,
  followed by the Random Iterative Compilation methodology using a uniform probability distribution.
- __[Page 42]__ Near optimal pass sequence can be directly predicted with high accuracy via Graph Edge Attention
  Network (GEAN), a graph neural network (GNN) architecture. RL-based approaches like PPO operating on the original
  compiler pass space often suffer from unstable training due to inaccurate value estimation and sparse reward space and
  fail to generalize to unseen programs at inference. As a contender to GNN, use transformers to process graphs. If we
  effectively encode positional and local sub-structures of graphs and feed them to the transformer, then the
  transformer can outperform the classical GNN models.
- __[Page 62]__ On average, the agent manages to slightly reduce the gate count, although the depth is slightly
  increased. Even though this agent is, thus, not able to reliably optimize the circuit, it is still able to sometimes
  achieve an improvement. If we, however, train an agent via RL directly on this specific circuit, it does not only
  learn to reliably improve the circuit, but it also finds two further optimizations, each reducing the depth by 1.
- __[Page 68]__ The observations used to guide the reinforcement learning agent are based on seven features—namely the
  number of qubits, the depth of the circuit, and the five composite features of program communication, critical-depth,
  entanglement-ratio, parallelism, and liveness.
- __[Page 71]__ The Munich Quantum Compiler provides a subset tailored for each submitted quantum circuit by conducting
  a two-fold Design Space Exploration (DSE) to derive a Pareto-optimal solution. The two key stages of this process are
  the Target-Agnostic Optimization Stage and the Target-Specific Optimization Stage.
- __[Page 93]__ The manifold hypothesis is a conjecture of deep learning that posits that neural networks operate by
  learning low dimensional manifolds in high dimensional spaces. A digit that is rotated, translated, or stretched still
  lies along the same manifold.
- __[Page 123]__ Various characteristics are used to describe a quantum circuit for both models: the number of qubits,
  the depth of the circuit, and the five composite features of program communication, critical-depth,
  entanglement-ratio, parallelism, and liveness.
- __[Page 132]__ Propose the MQT Bench benchmark suite as part of the Munich Quantum Toolkit MQT. MQT Bench presents a
  first step towards benchmarking different abstraction levels with a single benchmark suite to increase comparability,
  reproducibility, and transparency.
- __[Page 136]__ Several approaches to provide benchmark suites for some of the levels within the quantum circuit
  compilation flow have already been proposed and a non-exhaustive overview is given in: pplication-Oriented Performance
  Benchmarks for Quantum Computing, SupermarQ, QASMbench and RevLib.

### 2. [Reinforcement Learning Basics](ResearchPapers/02_ReinforcementLearningBasics.pdf)

- __[Page 10]__ The additional concept that we need is discounting. The agent tries to select actions so that the sum of
  the discounted rewards it receives over the future is maximized. $\gamma$ is a parameter, $0\leq\gamma\leq1$, called
  the discount rate.
- __[Page 13]__ The value function of a state $s$ under a policy $\pi$, denoted $v_\pi(s)$, is the expected return when
  starting in $s$ and following $\pi$ thereafter. For Markov Decision Processes, we can define $v_\pi$ formally.
  Similarly, we define the value of taking action $a$ in state $s$ under a policy $\pi$, denoted $q_\pi(s,a)$, as the
  expected return starting from $s$, taking the action $a$, and thereafter following policy $\pi$. We call $q_\pi$ the
  action-value function for policy $\pi$. _
- __[Page 27]__ The quantity in brackets in the Temporal-Difference update is a sort of error, measuring the difference
  between the estimated value of $S_t$ and the better estimate $R_{t+1}+\gamma V(S_{t+1})$. This quantity, called the
  Temporal-Difference error $\delta_t$, arises in various forms throughout reinforcement learning.
- __[Page 75]__ Illustrate the architecture of a deep convolutional network. This instance was designed to recognize
  hand-written characters. It consists of alternating convolutional and subsampling layers, followed by several fully
  connected final layers. Each convolutional layer produces a number of feature maps.
- __[Page 87]__ The $\mathrm{TD}(\lambda)$ algorithm can be understood as one particular way of averaging $n$-step
  updates. This average contains all the $n$-step updates, each weighted proportionally to $\lambda^{n-1}$,
  where $\lambda\in[0,1]$, and is normalized by a factor of $\lambda-1$ to ensure that the weights sum to $1$. The
  resulting update is toward a return, called the $\lambda$-return.
- __[Page 132]__ Asynchronous Advantage Actor-Critic is a classic policy gradient method with the special focus on
  parallel training. In $\mathrm{A3C}$, the critics learn the state-value function $V_w(s)$, while multiple actors are
  trained in parallel and get synced with global parameters from time to time.
- __[Page 141]__ Each policy $\pi$ is represented by a neural network that maps a given state $s$ and goal $g$ to a
  distribution over action $\pi(a|s,g)$.
- __[Page 142]__ Our policies are trained with PPO using the clipped surrogate objective. We maintain two networks, one
  for the policy $\pi_\theta(a|s,g)$ and another for the value function $V_\psi(s,g)$.
- __[Page 152]__ Algorithm 1: Proximal Policy Optimization summarizes the common learning procedure used to train all
  policies. Policy updates are performed after a batch of $m=4096$ samples has been collected. Minibatches of
  size $n=256$ are then sampled from the data for each gradient step. A discount factor $\gamma=0.95$ is used for all
  motions. $\lambda=0.95$ is used for both $\mathrm{TD}(\lambda)$ and $\mathrm{GAE}(\gamma,\lambda)$.

### 3. [Reinforcement Learning Algorithms Set 1](ResearchPapers/03_ReinforcementLearningAlgorithmsSet1.pdf)

- __[Page 2]__ Methods that learn approximations to both policy and value functions are often called actor–critic
  methods, where actor is a reference to the learned policy, and critic refers to the learned value function, usually a
  state-value function.
- __[Page 12]__ Only through bootstrapping do we introduce bias and an asymptotic dependence on the quality of the
  function approximation. As we have seen, the bias introduced through bootstrapping and reliance on the state
  representation is often beneficial because it reduces variance and accelerates learning.
- __[Page 20]__ The objective function for the Actor-Critic algorithm is a combination of the policy gradient for the
  actor and the value function for the critic. $A(s,a)$ is the advantage function representing the advantage of taking
  the action $a$ in state $s$.
- __[Page 30]__ The Generalized Advantage Estimator $\mathrm{GAE}(\gamma,\lambda)$ is defined as the
  exponentially-weighted average of $k$-step estimators. The advantage estimator has a remarkably simple formula
  involving a discounted sum of Bellman residual terms. There are two notable special cases of this formula, obtained by
  setting $\lambda=0$ and $\lambda=1$.
- __[Page 41]__ We now present multi-threaded asynchronous variants of advantage actor-critic. Multiple actor-learners
  running in parallel are likely to be exploring different parts of the environment. Moreover, one can explic itly use
  different exploration policies in each actor-learner to maximize this diversity.
- __[Page 42]__ The algorithm, which we call asynchronous advantage actor-critic (A3C), maintains a
  policy $\pi(a_t|s_t,\theta$ and an estimate of the value function $V(s_t|\theta_v)$. As with the value-based methods
  we rely on parallel actor-learners and accumulated updates for improving training stability.
- __[Page 103]__ Compare PPO to several previous algorithms. On continuous control tasks, it performs better than the
  algorithms we compare against. On Atari, it performs significantly better in terms of sample complexity than A2C and
  similarly to ACER.
- __[Page 115]__ SAC avoids the complexity and potential instability associated with approximate inference in prior
  off-policy maximum entropy algorithms based on soft Q-learning. Empirical results show that soft actor-critic attains
  a substantial improvement in both performance and sample efficiency over both off-policy and on-policy prior methods.
- __[Page 131]__ Choosing the optimal temperature is non-trivial, and the temperature needs to be tuned for each task.
  Formulate different maximum entropy reinforcement learning objective, where the entropy is treated as a constraint.
  The magnitude of the reward differs not only across tasks, but it also depends on the policy, which improves over time
  during training.
- __[Page 141]__ Critic in SAC may be underfitted, as only a single gradient update step on the network parameters is
  performed for each environment step. Randomized Ensembled Double Q-Learning was proposed, which increased this number
  of gradient steps, termed update-to-data (UTD) ratio. In addition, Dropout Q functions improved the computational
  efficiency of REDQ while maintaining the same sample efficiency by replacing its ensemble of critics with dropout.
  REDQ and DroQ represent the state-of-the-art in terms of sample efficiency in Deep RL for continuous control.
- __[Page 159]__ <span style="font-variant: small-caps; font-size:15px;">Stable-Baselines3</span> contains the following
  state-of-the-art on-policy and off-policy algorithms, commonly used as experimental baselines: A2C, PPO, DDPG, SAC,
  TD3, HER, and DQN.

### 4. [RNN & LSTM for Sequences: Variants, Evaluations and Applications](ResearchPapers/04_RNNandLSTMforSequencesVariantsEvaluationsAndApplications.pdf)

- __[Page 12]__ With the efficient, truncated update rule, error flows only through connections to output unit, and
  through fixed self-connections within cell blocks. Error flow is truncated once it wants to leave memory cells or gate
  units. Therefore, no connection shown above serves to propagate error back to the unit from which the connection
  originates, although the connections themselves are modifiable. That is why the truncated LSTM algorithm is so
  efficient, despite its ability to bridge very long time lags.
- __[Page 15]__ We always use online learning, as opposed to batch learning, and logistic sigmoids as activation
  functions. Initial weights are chosen in the range $[−0.2,0.2]$, for the other experiments in $[−0.1,0.1]$. Training
  sequences are generated randomly according to the various task descriptions.
- __[Page 46]__ LSTM based RNN architectures can obtain state of the art performance in a large vocabulary speech
  recognition system with thousands of context dependent states. The proposed architectures modify the standard
  architecture of the LSTM networks to make better use of the model parameters while addressing the computational
  efficiency problems of large networks.
- __[Page 51]__ RNN and LSTM networks are causal models which condition every sequence element on the previous elements
  in the sequence. Later researches showed that processing the sequence in both directions can perform better for the
  sequences which can be processed offline.
- __[Page 52]__ One of the methods for training RNN is Backpropagation Through Time (BPTT), which is very similar to the
  backpropagation algorithm because it is based on gradient descent and chain rule, but it has also chain rule through
  time. BPTT was developed by several works. This algorithm is very solid in theory, however, it does not show the best
  performance in practice. In BPTT, the loss is considered as a summation of loss functions at the previous time steps
  until now.
- __[Page 62]__ The bidirectional LSTM includes two LSTM networks each of which processes the sequence from one
  direction. In other words, there are two LSTM networks which are fed with the sequence in opposite orders. Experiments
  have shown that the bidirectional LSTM outperforms the unidirectional LSTM.
- __[Page 68]__ Sequence modeling aims at learning a probability distribution over sequences, by maximizing the
  log-likelihood of a model given a set of training sequences.
- __[Page 75]__ Finite-sized RNNs with nonlinear activations are a rich family of models, capable of nearly arbitrary
  computation. With sigmoidal activation functions they can simulate a universal Turing machine.
- __[Page 90]__ For the intuition of the peephole connection consider a network which must learn to count objects and
  emit some desired output when n objects have been seen. The network might learn to let some fixed amount of activation
  into the internal state after each object is seen. This activation is trapped in the internal state by the constant
  error carousel, and is incremented iteratively each time another object is seen. When the nth object is seen, the
  network needs to know to let out content from the internal state so that it can affect the output.
- __[Page 116]__ Unitary/Orthogonal matrices keep the norm of vectors. By enforcing hidden to hidden transition matrix
  to be unitary/orthogonal, no matter how many time steps are propagated, the norm of the gradient will stay the same.
- __[Page 130]__ Use nonlinearity $\mathrm{modReLU}(z_i,b_i)=\mathrm{sign}(z_i)\cdot\mathrm{ReLU}(|z_i|+b_i)$. This
  nonlinearity function performs the best. This function possibly also serves as a forgetting filter that removes the
  noise using the bias threshold.
- __[Page 133]__ Efficient Unitary Neural Network (EUNN) whose computational cost is merely $\mathcal{O}(1)$ per
  parameter, which is  $\mathcal{O}(\log(N)))$  more efficient than the other methods discussed. It significantly
  outperforms existing RNN architectures on the standard Copying Task, and the pixel-permuted MNIST Task using a
  comparable parameter count, demonstrating the highest recorded ability to memorize sequential information over long
  time periods.

### 5. [Neural Network Architectures for Sequences of Elements: TCN & LSTM](ResearchPapers/05_NeuralNetworkArchitecturesForSequencesOfElementsTCNandLSTM.pdf)

- __[Page 2]__ Each layer can have a small kernel, for example `size=3`, but with dilation factors doubling at each
  layer $(1,2,4,8)$. This way, a relatively deep network, up to 12 layers, can capture long-range dependencies, hundreds
  of time-steps, without needing an impractically large kernel.
- __[Page 3]__ The TCN formulation distilled many of these best practices into a simple architecture can learn complex
  sequence patterns, even something as unusual as quantum gate sequences, given sufficient depth and training.
- __[Page 8]__ Results indicate that a simple convolutional architecture outperforms canonical recurrent networks such
  as LSTMs across a diverse range of tasks and datasets, while demonstrating longer effective memory. To represent
  convolutional networks, we describe a generic Temporal Convolutional Network (TCN) architecture that is applied across
  all tasks. This architecture is informed by recent research, but is deliberately kept simple, combining some of the
  best practices of modern convolutional architectures.
- __[Page 9]__ The TCN architecture appears not only more accurate than canonical recurrent networks such as LSTMs and
  GRUs, but also simpler and clearer. It may therefore be a more appropriate starting point in the application of deep
  networks to sequences. Basic RNN architectures are notoriously difficult to train and more elaborate architectures are
  commonly used instead, such as the LSTM and the GRU.
- __[Page 15]__ The copy memory task is perfectly set up to examine a model's ability to retain information for
  different lengths of time. The requisite retention time can be controlled by varying the sequence length $T$. TCN
  outperforms LSTMs and vanilla RNNs by a significant margin in perplexity on LAMBADA, with a substantially smaller
  network and virtually no tuning.
- __[Page 39]__ Convolutional networks do not depend on the computations of the previoustime step and therefore allow
  parallelization over every ele ment in a sequence. This contrasts with RNNs which main tain a hidden state of the
  entire past that prevents parallel computation within a sequence.
- __[Page 54]__ A dilated convolution is a convolution where the filter is applied over an area larger than its length
  by skipping input values with a certain step. It is equivalent to a convolution with a larger filter derived from the
  original filter by dilating it with zeros, but is significantly more efficient. A dilated convolution effectively
  allows the network to operate on a coarser scale than with a normal convolution.
- __[Page 72]__ Architecture of LSTM-DQN: The Representation Generator $\phi_R$ takes as input a stream of words
  observed in state s and produces a vector representation $v_s$, which is fed into the action scorer $\phi_A$ to
  produce scores for all actions and argument objects.
- __[Page 80]__ To further enhance the agent's capacity to remember previous states, replace the shared MLP in $\phi_A$
  by an LSTM cell. LSTM-DRQN processes textual observations word-by-word to generate a fixed-length vector
  representation. This representation is used by the recurrent policy to estimate Q-values for all verbs $Q(s,v)$ and
  objects $Q(s,o)$.
- __[Page 94]__ Bi-directional LSTMs extend the idea of LSTMs by having two LSTMs in each layer. One LSTM processes the
  sequence from left to right, and the other from right to left. The outputs of both LSTMs are then concatenated. This
  allows the network to have access to past and future contexts at the same time.
- __[Page 112]__ For both $PM_{2.5}$ and $PM_{10}$ concentrations, the TCN-LSTM model produced the highest $R^2$ values
  of all the tested models, indicating that the TCN-LSTM model achieved the closest agreement between the predicted and
  observed value. These results indicate that the TCN-LSTM model had the highest prediction accuracy among the four deep
  learning models considered: TCN-LSTM, CNN-LSTM, LSTM, and TCN.
- __[Page 116]__ The TCN-LSTM model predicted $PM_{2.5}$ and $PM_{10}$ concentrations with satisfactory $R^2$ values
  of $0.95$ and $0.88$, respectively higher than those achieved by any other model considered in this study. The Monte
  Carlo cross-validation of the time series tests the robustness of the model, and the results showed the high stability
  of the TCN-LSTM model.

### 6. [Normalization & Optimization Schemes](ResearchPapers/06_NormalizationAndOptimizationSchemes.pdf)

- __[Page 3]__ Batch Normalization takes a step towards reducing internal covariate shift, and in doing so dramatically
  accelerates the training of deep neural nets. It accomplishes this via a normalization step that fixes the means and
  variances of layer inputs. Batch Normalization also has a beneficial effect on the gradient flow through the network,
  by reducing the dependence of gradients on the scale of the parameters or of their initial values. This allows us to
  use much higher learning rates without the risk of divergence.
- __[Page 12]__ Under layer normalization, all the hidden units in a layer share the same normalization terms $\mu$
  and $\sigma$, but different training cases have different normalization terms. Layer normaliztion does not impose any
  constraint on the size of a mini-batch and it can be used in the pure online regime with batch size 1.
- __[Page 13]__ In a layer normalized RNN, the normalization terms make it invariant to re-scaling all of the summed
  inputs to a layer, which results in much more stable hidden-to-hidden dynamics.
- __[Page 49]__ Weight normalization allows CrossQ to scale effectively. Through the addition of Weight Normalization,
  CrossQ+WN shows stable training and can stably scale with increasing UTD ratios. CrossQ benefits from the addition of
  WN, which results in stable training and scales well with higher UTD ratios.
- __[Page 65]__ Deep Neural Networks (DNN) and Recurrent Neural Networks (RNN) are powerful models that were considered
  to be almost impossible to train using stochastic gradient descent with momentum. When stochastic gradient descent
  with momentum uses a well-designed random initialization and a particular type of slowly increasing schedule for the
  momentum parameter, it can train both DNNs and RNNs on datasets with long-term dependencies to levels of performance
  that were previously achievable only with Hessian-Free optimization.
- __[Page 99]__ Adam is an algorithm for first-order gradient-based optimization of stochastic objective functions,
  based on adaptive estimates of lower-order moments. The method is computationally efficient, has little memory
  requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in
  terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very
  noisy and/or sparse gradients.
- __[Page 95]__ Adagrad outperforms SGD with Nesterov momentum by a large margin both with and without dropout noise.
  Adam converges as fast as Adagrad. Similar to Adagrad, Adam can take advantage of sparse features and obtain faster
  convergence rate than normal SGD with momentum. Although Adam convergence analysis does not apply to non-convex
  problems, Adam often outperforms other methods in such cases.
- __[Page 96]__ Adam and Adagrad make rapid progress lowering the cost in the initial stage of the training. Adam and
  SGD eventually converge considerably faster than Adagrad for CNNs. The second moment estimate $\hat{v}_t$ vanishes to
  zeros after a few epochs and is dominated by the $\epsilon$ in the Adam algorithm. The second moment estimate is
  therefore a poor approximation to the geometry of the cost function in CNNs comparing to fully connected networks.
  Reducing the minibatch variance through the first moment is more important in CNNs and contributes to the speed-up. As
  a result, Adagrad converges much slower than others.

### 7. [Reinforcement Learning Algorithms Set 2](ResearchPapers/07_ReinforcementLearningAlgorithmsSet2.pdf)

- __[Page 53]__ Use dropout for model uncertainty injection instead of the large ensemble. Specifically, the dropout
  Q-function that is a Q-function equipped with dropout and layer normalization, and DroQ, a variant of REDQ that uses a
  small ensemble of dropout Q-functions.
- __[Page 54]__ Dropout Q-function implementation: Dropout function is implemented by modifying REDQ. The modification
  is adding dropout and layer normalization. "Weight" is a weight layer and "ReLU" is the activation layer of rectified
  linear units.
- __[Page 82]__ By adjusting the number of randomly selected Q-functions for in-target minimization, REDQ can control
  the average Q-function bias. In comparison with standard ensemble averaging and with SAC with a higher
  Update-To-Data (UTD) ratio, REDQ has much lower std of Q-function bias while maintaining an average bias that is
  negative but close to zero throughout most of training, resulting in significantly better learning performance.

### 8. [Reinforcement Learning Algorithms Set 3](ResearchPapers/08_ReinforcementLearningAlgorithmsSet3.pdf)

- __[Page 4]__ Algorithm 2. Actor–Critic with Experience Replay. Estimators mentioned in Steps 6 and 7 are based on the
  samples in a database. Due to more exhaustive exploitation of information experience replay leads to faster learning
  at the cost of additional computation.
- __[Page 13]__ Prioritized Level Replay is a general framework for selectively sampling the next training level by
  prioritizing those with higher estimated learning potential when revisited in the future. TD-errors effectively
  estimate a level's future learning potential. Variation across levels implies that at each point of training, each
  level likely holds different potential for an agent to learn about the structure shared across levels to improve
  generalization.
- __[Page 14]__ The only requirements are satisfied by almost any problem that can be framed as Procedural Content
  Generation (PCG), including RL environments implemented as seeded simulator. PCG environment is any computational
  process that, given a level identifier like a a random seed, generates a level, defined as an environment instance
  exhibiting a unique configuration of its underlying factors of variation.
