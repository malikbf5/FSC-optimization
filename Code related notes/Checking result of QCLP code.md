# Checking code
for the following values:

number of states: 2

number of actions: 2

number of observations: 2

$\theta = 0.8$, $\gamma = 0.5$, $b_0 = (0.5, 0.5)$

Reward model:

$R(s = A, a = 1) = P(r = 1 \,|\, s= A, a=1) = \theta$  

$R(s = A, a = 2) = P(r = 1 \,|\, s= A, a=2) = 1 - \theta$ 

$R(s = B, a = 1) = P(r = 1 \,|\, s= B, a=1) = 1 - \theta$ 

$R(s = B, a = 2) = P(r = 1 \,|\, s= B, a=2) = \theta$ 

State transition model:

$P(s'|s, a) = \mathbb{1}_{s=s'}$ 

Observation model:

$O(o = 1|s' = A,a = 1) = \theta$

$O(o = 1|s' = A,a = 2) = 1 - \theta$

$O(o = 1|s' = B,a = 1) = 1 - \theta$

$O(o = 1|s' = B,a = 2) = \theta$

#### Objective function given by the code:

obj = $0.5 y(0,0) + 0.5 y(0,1)$

#### Constraints:

##### Bellman constraint:

for $q = 0, s = 0$: 

$y[0,0] = (0.8 x[0,0,0,0] + 0.5 (0.2 x[0,0,0,0] y[0,0] + 0.0 x[0,0,0,0] y[0,1] + 0.8 x[0,0,0,1] y[0,0] + 0.0 x[0,0,0,1] y[0,1])  
+ 0.2 x[0,1,0,0] + 0.5 (0.8 x[0,1,0,0] y[0,0] + 0.0 x[0,1,0,0] y[0,1] + 0.2 x[0,1,0,1] y[0,0] + 0.0 x[0,1,0,1] y[0,1]))$

 for $q = 0, s = 1:$
 
 $y[0,1] = (0.2 x[0,0,0,0] + 0.5 (0.0 x[0,0,0,0] y[0,0] + 0.8 x[0,0,0,0] y[0,1] + 0.0 x[0,0,0,1] y[0,0] + 0.2 x[0,0,0,1] y[0,1])  
 + 0.8 x[0,1,0,0] + 0.5 (0.0 x[0,1,0,0] y[0,0] + 0.2 x[0,1,0,0] y[0,1] + 0.0 x[0,1,0,1] y[0,0] + 0.8 x[0,1,0,1] y[0,1]))$

 ##### Sum over action and resulting node:

 for $q = 0, o = 0:$

 $x[0,0,0,0] + x[0,1,0,0] == 1$

 for $q = 0, o = 1:$

 $x[0,0,0,1] + x[0,1,0,1] == 1$

 ##### Independence of observation:

 $o_k = 0$
 
for $q = 0, a = 0, o = 0$  
$x[0,0,0,0] == x[0,0,0,0]$

for $q = 0, a = 1, o = 0$  
$x[0,1,0,0] == x[0,1,0,0]$ 

for $q = 0, a = 0, o = 1$  
$x[0,0,0,1] ==  x[0,0,0,0]$ 

for $q = 0, a = 1, o = 1$  
$x[0,1,0,1] == x[0,1,0,0]$ 
