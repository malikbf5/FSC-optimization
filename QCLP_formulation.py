from pyomo.environ import *

# define the qclp model and optimize it
def qclp_formulation(num_states, num_actions, num_observations, num_nodes, b_0, gamma, state_transition_model, reward_model, observation_model):
    # Check if the inputs are valid
    if num_nodes > 0 and num_actions > 0 and num_observations > 0 and num_states > 0:
        # Define model
        model = ConcreteModel()

        # Define the indices of the variables and constraints
        model.q = range(num_nodes)
        model.a = range(num_actions)
        model.o = range(num_observations)
        model.s = range(num_states)

        # Variables
        # P(q', a | q, o)
        model.x = Var(model.q,model.a,model.q,model.o, bounds = (0.0,1.0))
        # V(q , s)
        model.y = Var(model.q,model.s )
        
        # Objective
        # Maximize sum over s_ of b_0(s)*V(0, s)
        model.obj = Objective(expr = sum([b_0[s_]*model.y[0,s_] for s_ in model.s]) , sense = maximize)
        
        # Constraints
        # Probability constraints
        model.sum_over_action_and_qp = ConstraintList()
        for q_ in model.q:
            for o_ in model.o:
                model.sum_over_action_and_qp.add(
                    sum([model.x[qp, a_, q_ , o_] for qp in model.q for a_ in model.a]) == 1)
                
        model.sum_independence_on_o = ConstraintList()
        for q_ in model.q:
            for o_ in model.o:
                for a_ in model.a:
                    if o_ != 0:
                        model.sum_independence_on_o.add(
                            sum([model.x[qp, a_, q_, o_] for qp in model.q]) 
                            == sum([model.x[qp, a_, q_, 0] for qp in model.q]))
        
        # Bellman equation constraints
        model.bellman_equation = ConstraintList()
        for q_ in model.q:
            for s_ in model.s:
                model.bellman_equation.add(model.y[q_,s_] ==
                    sum([
                        sum([model.x[qp_, a_, q_, 0] for qp_ in model.q]) 
                        * reward_model[s_, a_] 
                        + gamma 
                        * sum([state_transition_model[s_, a_, sp_] 
                                * observation_model[sp_, a_, o_] 
                                * model.x[qp_, a_, q_, o_] 
                                * model.y[qp_, sp_] for qp_ in model.q for o_ in model.o for sp_ in model.s ]) for a_ in model.a]))
                
        return model
    




# initialize the model variable x with win_stay lose_shift strategy for two nodes controller
def win_stay_lose_shift(num_nodes,newmodel):
    # initial values for the variables with Win-Stay Lose-Shift policy
    if num_nodes == 2:
        for qnode in newmodel.q:
            for actionn in newmodel.a:
                for qnodeprime in newmodel.q:
                    for obs in newmodel.o:
                        if qnode == qnodeprime and qnode == actionn and obs == 1:
                            newmodel.x[qnodeprime,actionn,qnode,obs] = 1
                            # print(f"being in {qnode} having chosen {actionn} and observed {obs} we stay in {qnodeprime}", newmodel.x[qnodeprime,actionn,qnode,obs].value)
                        elif qnode != qnodeprime and qnode == actionn and obs == 0:
                            newmodel.x[qnodeprime,actionn,qnode,obs] = 1 
                            # print(f"being in {qnode} having chosen {actionn} and observed {obs} we shift to {qnodeprime}",newmodel.x[qnodeprime,actionn,qnode,obs].value)
                        else:
                            newmodel.x[qnodeprime,actionn,qnode,obs] = 0
                            # print(f"being in {qnode} having chosen {actionn} and observed {obs} we don't go to {qnodeprime}",newmodel.x[qnodeprime,actionn,qnode,obs].value)
        newmodel.x.pprint() 





# initialize the model variable x with random strategy
def generate_randomx(model,num_nodes,num_actions):
    for qnode in model.q:
        # left in P(a | q)
        left_in_a = 1
        for actionn in model.a:
            # if last a
            if actionn == num_actions - 1:
                prob_a = left_in_a
            # else generate random number between 0 and what's left
            else:
                prob_a = random.uniform(0,left_in_a)
                left_in_a -= prob_a
            for obs in model.o:
            # left in P(q' | q, a, o)
                left_in_qp = 1
                for qnodeprime in model.q:
                    # if last q'
                    if qnodeprime == num_nodes - 1:
                        prob_qp = left_in_qp
                    # else generate random number between 0 and what's left
                    else:
                        prob_qp = random.uniform(0,left_in_qp)
                        left_in_qp -= prob_qp
                    # P(q',a | q,o) = P(q' | q, a, o) * P(a | q)
                    model.x[qnodeprime,actionn,qnode,obs] = prob_qp * prob_a




