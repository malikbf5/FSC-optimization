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
