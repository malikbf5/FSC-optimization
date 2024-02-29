## Abstract
Developing scalable algorithms for solving partially observable Markov decision processes (POMDPs) is an important challenge. One approach that effectively addresses the intractable memory requirements of POMDP algorithms is based on representing POMDP policies as finitestate controllers. In this paper, we illustrate some fundamental disadvantages of existing techniques that use controllers. We then propose a new approach that formulates the problem as a quadratically constrained linear program (QCLP), which defines an optimal controller of a desired size. This representation allows a wide range of powerful nonlinear programming algorithms to be used to solve POMDPs. Although QCLP optimization techniques guarantee only local optimality, the results we obtain using an existing optimization method show significant solution improvement over the state-of-the-art techniques. The results open up promising research directions for solving large POMDPs using nonlinear programming methods.


## Notes
##### Intro
* pomdp applications; cassandra 1998b survey
* effective algorithms for solving MDPs and POMDPs
* current POMDP exact techniques are limited by high memory requirements to toy problems.
##### Background
* We also allow for stochastic transitions and action selection, as this can help to make up for limited memory [Singh et al., 1994]
* previous work on optimizing fsc: bpi, biased bpi, Meuleau et al. fsc gradient ascent and their disadvantages
##### Optimal fixed-size controllers
* QCLP formulation and theorem
##### Methods for solving QCLP
* problem in non convex, global optimality can't be guaranteed
* non linear programming methods for such problems can guarantee locally optimal solutions but also globally optimal ones can be found sometimes
* merit functions, which evaluate a current solution based on fitness criteria, can be used to improve convergence
* the problem space can be made convex by approximation or domain information
* the quadratic constraints and linear objective of QCLP often permits better approximations and the representation is more likely to be convex than problems with a higher degree objective and constraints
* solver used is snopt which uses SQP, a merit function is also used to guarantee convergence from any initial point
##### Experiments
* QCLP compared with BPI and biased BPI
* lower and upper bounds were added which represent the value of taking the highest and lowest valued action (of the underlying mdp) respectively for an infinite number of steps
* domains: hallway benchmark, machine maintenance problem