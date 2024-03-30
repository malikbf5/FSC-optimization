from pyomo.environ import *
import random
import pandas as pd
from itertools import product
import os
os.environ['NEOS_EMAIL'] = 'malikbf5@gmail.com' 

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
    




# initialize the model variable x with win_stay lose_shift strategy for 2 nodes controller
def win_stay_lose_shift_2_init(newmodel,solvername = "snopt"):
    # initial values for the variables with Win-Stay Lose-Shift policy
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
    # newmodel.x.pprint() 
    # solve the model
    opt = SolverManagerFactory("neos")
    opt.solve(newmodel, solver = solvername)
    # results dataframe
    vdf = value_dataframe(newmodel,name ="V_WSWS82_init(q,s)")
    adf, ndf = actionselect_nodetrans(newmodel.x, horiz_action=True,horiz_trans=False)
    return newmodel, vdf, adf, ndf





# Win Stay Lose Shift strategy for two node controller dataframe
value_WSLS2 = [{"(q,s)":(qi,si), "V_WSLS_2(q,s)": 1.47 if qi == si else 0.87}
            for (qi,si) in product(range(2),range(2))]
value_WSLS2df = pd.DataFrame(value_WSLS2).T
value_WSLS2df.columns = value_WSLS2df.iloc[0]
value_WSLS2df.drop(value_WSLS2df.index[0], inplace=True)
value_WSLS2df.insert(len(value_WSLS2df.columns),"mean value",float(value_WSLS2df.mean(axis=1)[0]))
value_WSLS2df.insert(len(value_WSLS2df.columns),"objective funct",0.5* (1.47 + 0.87))





# initialize the model variable x with win_stay lose_shift strategy for 4 nodes controller





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





# create value dataframe
def value_dataframe(newmodel,horiz=True, name ="V(q,s)"):
    optimal_y = {"(q,s)": [key for key in newmodel.y.get_values().keys()],
                  name: [newmodel.y.get_values()[key] for key in newmodel.y.get_values().keys()]}
    if horiz:
        optimal_y_df = pd.DataFrame(optimal_y).T   
        # specify column names
        optimal_y_df.columns = optimal_y_df.iloc[0]
        # drop extra row
        optimal_y_df = optimal_y_df.drop(optimal_y_df.index[0])
        # add objective function value
        optimal_y_df.insert(0,"objective funct",value(newmodel.obj))
        # add mean value
        optimal_y_df.insert(1,"mean value",float(optimal_y_df.mean(axis=1)[0]))
        # add value over nodes
        optimal_y_df.insert(2,
                            "value for nodes V(q)",
                            str([round(sum(newmodel.y.get_values()[key] for key in newmodel.y.get_values().keys() if key[0] == qnode)
                                   /len(newmodel.s),3) 
                                   for qnode in newmodel.q]))
        # add value over states
        optimal_y_df.insert(3,"value for states V(s)",
                            str([round(sum(newmodel.y.get_values()[key] for key in newmodel.y.get_values().keys() if key[1] == state)
                                   /len(newmodel.q),3) 
                             for state in newmodel.s]))
    else:
        optimal_y_df = pd.DataFrame(optimal_y)
        # set index
        optimal_y_df = optimal_y_df.set_index("(q,s)")
        # add objective function value
        optimal_y_df.loc["objective funct"] = value(newmodel.obj)
        # add mean value
        optimal_y_df.loc["mean value"] = float(optimal_y_df.mean(axis=1)[0])
        # add value over nodes
        optimal_y_df.loc["value for nodes V(q)"] = str([round(sum(newmodel.y.get_values()[key] for key in newmodel.y.get_values().keys() 
                                                              if key[0]==qnode)/len(newmodel.s),3) 
                                                              for qnode in newmodel.q])
        # add value over states
        optimal_y_df.loc["value for states V(s)"] = str([round(sum(newmodel.y.get_values()[key] for key in newmodel.y.get_values().keys() 
                                                               if key[1]==si)/len(newmodel.q),3) 
                                                               for si in newmodel.s])
        
    return optimal_y_df





# create action selection and node transition dataframes
def actionselect_nodetrans(newmodelx, horiz_action = True, horiz_trans = False):
    nodetrans = {}
    actionselect = {}

    for key in newmodelx.get_values().keys():
        index = key[1:]
        actionselect[key[1:3]] = sum(newmodelx.get_values()[key] for key in newmodelx.get_values().keys() if key[1:] == index)
        nodetrans[key] = 0 
        if actionselect[key[1:3]] != 0:
            nodetrans[key] = newmodelx.get_values()[key] / actionselect[key[1:3]]

    actionselectdict = [{"(a,q)": key, "P(a | q)": actionselect[key]}for key in actionselect.keys()]
    if horiz_action:
        actionselectdf = pd.DataFrame(actionselectdict).T
        actionselectdf.columns = actionselectdf.iloc[0]
        actionselectdf.drop(actionselectdf.index[0], inplace=True)
    else:
        actionselectdf = pd.DataFrame(actionselectdict)
        actionselectdf = actionselectdf.set_index("(a,q)")
    
    nodetransdict = [{"(q',a,q,o)": key, "P(q' | q, a, o)": round(nodetrans[key],3), "P(q',a | q,o)": round(value(newmodelx[key]),3)}for key in nodetrans.keys()]
    nodetransdf = pd.DataFrame(nodetransdict)
    nodetransdf.set_index("(q',a,q,o)", inplace=True)
    if horiz_trans:
        nodetransdf = pd.DataFrame(nodetransdict).T
        nodetransdf.columns = nodetransdf.iloc[0]
        nodetransdf.drop(nodetransdf.index[0], inplace=True)
    else:
        nodetransdf = pd.DataFrame(nodetransdict)
        nodetransdf.set_index("(q',a,q,o)", inplace=True)
    
    return actionselectdf, nodetransdf
