# FSC optimization
Where I keep my files related to my master's thesis in the optimization of Finite-State Controllers with applications to navigation tasks.

FSC are versatile and memory-efficient policy parametrizations for the control of partially observable systems. They can be optimized both in a model-based and in a model-free setting. Since the problem is non convex, because of partial observability, the optimization is nontrivial. In this project we will study different approaches to FSC optimization that improve over gradient descent. The resulting methods will be applied to several navigation tasks with applications ranging from the modeling of animal behavior to robotics. 

# Requirements
In order to run some files, it is needed to install pyomo: <br>
https://pyomo.readthedocs.io/en/stable/installation.html

The Ipopt solver is also needed, it can be downloaded throught the following link: <br>
https://www.coin-or.org/download/binary/Ipopt/ <br>
Then the Ipopt\bin folder must be added to the PATH

Here's a helpful tutorial to perform the aforementioned steps: <br>
https://www.youtube.com/watch?v=EB_qVoM74Fg&ab_channel=BostanjiDevelopers
