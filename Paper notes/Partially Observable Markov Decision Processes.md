## Abstract
For reinforcement learning in environments in which an agent has access to a reliable state signal, methods based on the Markov decision process (MDP) have had many successes. In many problem domains, however, an agent suffers from limited sensing capabilities that preclude it from recovering a Markovian state signal from its perceptions. Extending the MDP framework, partially observable Markov decision processes (POMDPs) allow for principled decision making under conditions of uncertain sensing. In this chapter we present the POMDP model by focusing on the differences with fully observable MDPs, and we show how optimal policies for POMDPs can be represented. Next, we give a review of model-based techniques for policy computation, followed by an overview of the available modelfree methods for POMDPs. We conclude by highlighting recent trends in POMDP reinforcement learning.

## Notes
##### Intro
* good and thorough argument for the need for POMDPs by contrasting with fully observable MDPs, useful for my thesis introduction

##### Decision making in partially observable environments
* continuous and structured pomdp representations
* belief updating
* value function over beliefs and its convex shape

##### Model-based techniques
* Even when the full model is known to the agent, solving the POMDP optimally is typically only computationally feasible for small problems, hence the interest in methods that compromise optimality for reasons of efficiency. 
* All the methods presented in this section exploit a belief state representation, as it provides a compact representation of the complete history of the process.
* Solving the underlying MDP is of much lower complexity than solving the POMDP (P-complete vs. PSPACE-complete)

	###### MDP solution based heuristics:
	
		- most likely state (MLS) heuristic: ignores uncertainty in state
		- Qmdp heuristic: assumes that any uncertainty regarding the state             will disappear after taking one action. will fail in domains where            repeated information gathering is necessary.
		- Expansion of the MDP setting to model some form of sensing                   uncertainty without considering full-blown POMDP beliefs, instead             the mean and entropy of the belief distribution which can fail if             it's not uni-modal. 
	
    Approximate methods:
		
		- POLICY SEARCH: search for a good policy within a restricted class            of controllers
		- policy iteration and bounded policy iteration search through the             space of (bounded-size) stochastic finite-state controllers by                performing policy-iteration steps.
		- Other options for searching the policy space include gradient                ascent and heuristic methods like stochastic local search
		- PEGASUS method
		- Policy search methods have demonstrated success in several cases,             but searching in the policy space can often be difficult and                  prone to local optima
		- heuristic search. Defining an initial belief b0 as the root node,             these methods build a tree that branches over (a,o) pairs, each               of which recursively induces a new belief node. Branch-and-bound              techniques are used to maintain upper and lower bounds to the                 expected return at fringe nodes in the search tree. 
		- Hansen (1998a) proposes a policy-iteration method that represents             a policy as a finite-state controller, and which uses the belief             tree to focus the search on areas of the belief space where the               controller can most likely be improved, its applicability to                  large problems is limited by its use of full dynamic-programming              updates. 


##### Decision making without a-priori models
    
* direct (truly model free) and indirect (reconstruct pomdp) RL
* memoryless techniques: deterministic and stochastic policies
* Storing the complete history is not practical (model-free case the agent can't compute a belief state, such a representation does not allow for easy generalization)
* models with internal memory: 
	* finite history window (can't capture long term dependencies)
	* utile suffix memory (short term memory rep as a suffix tree)
	* LSTM RNNs as internal state rep
	* extension of VAPS to finite state automata (finite policy graph) with stochastic gradient descent to converge to local optimal controller (the optimal POMDP policy can require an infinite policy graph to be properly represented)
	* predictive state representations as an alternative to pomdps (advantageous for model free)
	
    

	    
    
    
	

##### Recent trends
Most of the model-based methods discussed in this chapter are offline techniques that determine a priori what action to take in each situation the agent might encounter. Online approaches, on the other hand, only compute what action to take at the current moment (Ross et al, 2008b). Focusing exclusively on the current decision can provide significant computational savings in certain domains, as the agent does not have to plan for areas of the state space which it never encounters. However, the need to choose actions every time step implies severe constraints on the online search time. Offline point-based methods can be used to compute a rough value function, serving as the online search heuristic. In a similar manner, Monte Carlo approaches are also appealing for large POMDPs, as they only require a generative model (black box simulator) to be available and they have the potential to mitigate the curse of dimensionality (Thrun, 2000; Kearns et al, 2000; Silver and Veness, 2010). As discussed in detail in the chapter on Bayesian reinforcement learning, Bayesian RL techniques are promising for POMDPs, as they provide an integrated way of exploring and exploiting models. Put otherwise, they do not require interleaving the model-learning phases (e.g., using Baum-Welch (Koenig and Simmons, 1996) or other methods (Shani et al, 2005)) with model-exploitation phases, which could be a naive approach to apply model-based methods to unknown POMDPs. Poupart and Vlassis (2008) extended the BEETLE algorithm (Poupart et al, 2006), a Bayesian RL method for MDPs, to partially observable settings. As other Bayesian RL methods, the models are represented by Dirichlet distributions, and learning involves updating the Dirichlet hyper-parameters. The work is more general than the earlier work by Jaulmes et al (2005), which required the existence of an oracle that the agent could query to reveal the true state. Ross et al (2008a) proposed the BayesAdaptive POMDP model, an alternative model for Bayesian reinforcement learning which extends Bayes-Adaptive MDPs (Duff, 2002). All these methods assume that the size of the state, observation and action spaces are known. Policy gradient methods search in a space of parameterized policies, optimizing the policy by performing gradient ascent in the parameter space (Peters and Bagnell, 2010). As these methods do not require to estimate a belief state (Aberdeen and Baxter, 2002), they have been readily applied in POMDPs, with impressive results (Peters and Schaal, 2008). Finally, a recent trend has been to cast the model-based RL problem as one of probabilistic inference, for instance using Expectation Maximization for computing optimal policies in MDPs. Vlassis and Toussaint (2009) showed how such methods can also be extended to the model-free POMDP case. In general, inference methods can provide fresh insights in well-known RL algorithms.