## Abstract
This paper introduces a novel and computationally efficient policy representation, termed a banded controller, for Partially Observable Markov Decision Processes (POMDPs). The structure of a banded controller is obtained by restricting the number of successor nodes for each node in a finite state controller (FSC) policy representation; this is formally defined as the restriction of the controller’s node transition matrices to the space of banded matrices. A gradient ascent based algorithm which leverages banded matrices is presented and we show that the policy structure results in a computational structure that can be exploited when performing policy evaluation. We then show that policy evaluation is asymptotically superior to a general FSC and that the degrees of freedom can be reduced while maintaining a large amount of expressivity in the policy. Specifically, we show that banded controller policy representations are equivalent to any FSC policy which is permutation similar to a banded controller. Meaning that banded controllers are computationally efficient policy representations for a class of FSC policies. Lastly, experiments are conducted which show that banded controllers outperform state-of-the-art FSC algorithms on many of the standard benchmark problems.

## Notes
##### Intro
* solving pomdp is pscape complete
* online vs offline methods
* fsc approaches: nonlinear programming formulations (NLPs) , gradient based methods, and policy iteration methods
* fsc are ideal for energy constrained systems (e.g. smartphones) application area

##### Background
* two ways of representing policy, the belief based approach and the fsc approach using the action selection model $\psi$ and the observation dependent node transitions $\eta$

##### Banded controllers
* banded controllers are computationally efficient representations for a class of controllers with no a priori structure
* $\eta_B$  = $P^{-1}  \eta  P$  where $\eta_B$ is a banded matrix, following that we have $\psi_B = P \psi$ 
* A controller (ψ,η) is permutation similar to a banded controller, if there exists a permutation matrix P satisfying P 2 =I such that P −1ηP ∈B
* policy evaluation can be done much more efficiently because of banded structure

##### Banded controller gradient ascent algorithm
* reduced form formulation of the problem using the banded structure without a projection into banded controller space
* projected gradient ascent algorithm

##### Experiments
* banded gradient ascent is typically much faster than both NLP and gradient ascent baselines


 