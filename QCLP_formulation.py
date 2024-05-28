from pyomo.environ import *
import random
import pandas as pd
from itertools import product
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
os.environ['NEOS_EMAIL'] = 'malikbf5@gmail.com' 


# define the qclp model and optimize it
def qclp_formulation(num_states, num_actions, num_observations, num_nodes, b_0, gamma, state_transition_model, reward_model, observation_model, obj = "first node"):
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
        if obj == "first node":
            # Maximize sum over s_ of b_0(s)*V(0, s)
            model.obj = Objective(expr = sum([b_0[s_]*model.y[0,s_] for s_ in model.s]) , sense = maximize)
        elif obj == "all nodes":
            # Maximize sum over q_ of sum over s_ of b_0(q,s)*V(q_, s)
            model.obj = Objective(expr = sum([b_0[s_]*model.y[q_,s_]*(1/len(model.q)) for q_ in model.q for s_ in model.s]) , sense = maximize)            
        
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
                        * sum([state_transition_model[sp_, a_,s_ ] 
                                * observation_model[sp_, a_, o_] 
                                * model.x[qp_, a_, q_, o_] 
                                * model.y[qp_, sp_] for qp_ in model.q for o_ in model.o for sp_ in model.s ]) for a_ in model.a]))
                
    return model





# function that generates FSCs of different sizes and optimizes them
def opt_instances(num_nodes_list, num_instances_for_each_numnodes,num_states, num_actions, num_observations, b0, gamma, state_transition_model, reward_model, observation_model, obj = "first node"):
    sol = {}
    for num_node in num_nodes_list:
        sol[num_node] = {}
        for instance in range(num_instances_for_each_numnodes):
            sol[num_node][instance] = {}
            # call qclp funct
            newmodel = qclp_formulation(num_states, num_actions, num_observations, num_node, b0, gamma, state_transition_model, reward_model, observation_model, obj = obj)
            # generate x values randomly
            generate_randomx(newmodel, num_node, num_actions)
            # newmodel.x.pprint()
            # print([sum(newmodel.x[qnodeprime,actionn,qnode,obs].value for qnodeprime in newmodel.q for actionn in newmodel.a) for qnode in newmodel.q for obs in newmodel.o])
            # print([sum(newmodel.x[qnodeprime,actionn,qnode,obs].value for qnodeprime in newmodel.q) == sum(newmodel.x[qnodeprime,actionn,qnode,0].value for qnodeprime in newmodel.q)  for qnode in newmodel.q for actionn in newmodel.a for obs in newmodel.o])
            # call opt
            opt = SolverManagerFactory('neos')
            opt.solve(newmodel, solver = "snopt")
            # save results
            # dataframe for y
            vdf = value_dataframe(newmodel)
            # dataframe for action selection and node transition
            adf, ndf = actionselect_nodetrans(newmodel.x, horiz_action = True, horiz_trans = True)
            sol[num_node][instance] = {"model": newmodel, "value df": vdf, 
                                       "action selection df": adf, "node transition df": ndf,
                                       "objective": value(newmodel.obj),
                                       "mean value": sum(newmodel.y.get_values().values()) / len(newmodel.y.get_values()),
                                       "mean value for nodes": [round(sum(newmodel.y.get_values()[key] for key in newmodel.y.get_values().keys() if key[0] == qnode)/len(newmodel.s),2) for qnode in newmodel.q],
                                       "mean value for states": [round(sum(newmodel.y.get_values()[key] for key in newmodel.y.get_values().keys() if key[1] == state)/len(newmodel.q),2) for state in newmodel.s]}
        # mean value for a given controller size over instances
        sol[num_node]["mean value"] = round(sum(sol[num_node][instance]["mean value"] 
                                                for instance in range(num_instances_for_each_numnodes)) 
                                                / num_instances_for_each_numnodes,3)
        # max objective function for a given controller size over instances
        sol[num_node]["max obj"] = max([sol[num_node][instance]["objective"] 
                                                 for instance in range(num_instances_for_each_numnodes)])
        # mean objective function for a given controller size over instances
        sol[num_node]["mean obj"] = sum(sol[num_node][instance]["objective"] 
                                                 for instance in range(num_instances_for_each_numnodes)) / num_instances_for_each_numnodes
        # mean value for nodes for a given controller size over instances
        sol[num_node]["mean value for nodes"] = np.round(sum(np.array(
            sol[num_node][instance]["mean value for nodes"]) 
            for instance in range(num_instances_for_each_numnodes)) 
            / num_instances_for_each_numnodes,3)
        # mean value for states a given controller size over instances
        sol[num_node]["mean value for states"] = np.round(sum(np.array(
            sol[num_node][instance]["mean value for states"]) 
            for instance in range(num_instances_for_each_numnodes)) 
            / num_instances_for_each_numnodes,3)
        # value df for a given controller size    
        sol[num_node]["value df"] = pd.concat([sol[num_node][instance]["value df"] for instance in range(num_instances_for_each_numnodes)])
        # action selection df for a given controller size
        sol[num_node]["action select df"] = pd.concat([sol[num_node][instance]["action selection df"] for instance in range(num_instances_for_each_numnodes)])
        # node transition df for a given controller size
        sol[num_node]["node trans df"] = pd.concat([sol[num_node][instance]["node transition df"] for instance in range(num_instances_for_each_numnodes)])
    
    # solution dataframe
    sol["dataframe"] = pd.DataFrame({ "controller size": num_nodes_list, 
                                     "max obj": [sol[num_node]["max obj"] for num_node in num_nodes_list],
                                     "mean obj": [sol[num_node]["mean obj"] for num_node in num_nodes_list],
                                     "mean value": [sol[num_node]["mean value"] for num_node in num_nodes_list], 
                                     "mean value for nodes V(q)": [sol[num_node]["mean value for nodes"] for num_node in num_nodes_list], 
                                     "mean value for states V(s)": [sol[num_node]["mean value for states"] for num_node in num_nodes_list]})
    sol["dataframe"].set_index("controller size", inplace=True)
    return sol





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
        optimal_y_df.insert(len(optimal_y_df.columns),"objective funct",value(newmodel.obj))
        # add mean value
        optimal_y_df.insert(len(optimal_y_df.columns),"mean value",float(optimal_y_df.mean(axis=1)[0]))
        # add value over nodes
        optimal_y_df.insert(len(optimal_y_df.columns),
                            "value for nodes V(q)",
                            str([round(sum(newmodel.y.get_values()[key] for key in newmodel.y.get_values().keys() if key[0] == qnode)
                                   /len(newmodel.s),3) 
                                   for qnode in newmodel.q]))
        # add value over states
        optimal_y_df.insert(len(optimal_y_df.columns),"value for states V(s)",
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




# function that calculates the value (variables y) of a fixed strategy (fixed x variables)
def calculate_value_list(model, state_transition_model, observation_model, reward_model, gamma):
    # matrix of coefficients
    a = np.zeros((len(model.q)*len(model.s),len(model.q)*len(model.s)))
    # vector of constants
    b = np.zeros(len(model.q)*len(model.s))
    # loop the same way in qclp_formulation
    i = 0
    for q_ in model.q:
        for s_ in model.s:
            # sum_a( sum_q'(x[q',a,q,o_k]) * R[s, a] )
            b[i] = -sum(
                sum([value(model.x[qp_, a_, q_, 0]) for qp_ in model.q])
                * reward_model[s_, a_]   for a_ in model.a)
            j = 0
            for qp_ in model.q:
                for sp_ in model.s:
                    # sum_a( sum_o( gammma * P(s'|s,a) * O(o|s',a) * x[q',a,q,o] ) )
                    a[i,j] = sum(
                            gamma * state_transition_model[sp_,a_,s_] *
                            observation_model[sp_, a_, o_] *
                            value(model.x[qp_, a_, q_, o_]) 
                        for a_ in model.a for o_ in model.o) - int(qp_ == q_ and sp_ == s_)
                    j += 1
            i += 1
    # solve the linear system and get values for this strategy
    v = list(np.linalg.solve(a, b))
    return v            





# function that returns action selection and node transition dict
def actionselect_nodetrans_dict(newmodelx, output = "all"):
    nodetrans = {}
    actionselect = {}
    for key in newmodelx.get_values().keys():
        # fixing (a,q,o)
        index = key[1:]
        # summing over x values (sum on q') where (a,q,o) is fixed
        actionselect[key[1:3]] = sum(newmodelx.get_values()[key] for key in newmodelx.get_values().keys() if key[1:] == index)
        # calculating P(q' | q, a, o) = x[q',a,q,o] / P(a | q)
        nodetrans[key] = 0 
        if actionselect[key[1:3]] != 0:
            nodetrans[key] = newmodelx.get_values()[key] / actionselect[key[1:3]]
    if output == "actionselect":
        return actionselect
    elif output == "nodetrans":
        return nodetrans
    else:
        return actionselect, nodetrans





# create action selection and node transition dataframes
def actionselect_nodetrans(newmodelx, horiz_action = True, horiz_trans = False):
    # get dicts
    actionselect, nodetrans = actionselect_nodetrans_dict(newmodelx)

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




# function that returns positions of action nodes circling a controller node
def action_node_pos(number_actions,origin, radius = 1, angle = 0):
    angle_increment = np.divide(np.multiply(2 , np.pi) , number_actions)
    action_pos = []
    for i in range(number_actions):
        action_pos.append((np.multiply(np.cos(angle),radius)+origin[0], np.multiply(np.sin(angle),radius) + origin[1]))
        angle += angle_increment
    return action_pos




# function that returns positions of nodes in a horizontal line
def horiz_pos(number_nodes, space = 1, origin = (0,0)):
    qnode_pos_list = []
    if number_nodes % 2 == 0:
        qnode_pos_list.append((origin[0] + space * 0.5, origin[1]))
        qnode_pos_list.append((origin[0] - space * 0.5, origin[1]))
        for i in range(1,number_nodes //2):
            qnode_pos_list.append((qnode_pos_list[0][0] + (i)* space, qnode_pos_list[0][1] ))
            qnode_pos_list.append((qnode_pos_list[1][0] - (i)*space, qnode_pos_list[1][1] ))
    else:
        qnode_pos_list.append(origin)
        for i in range(number_nodes // 2):
            qnode_pos_list.append((origin[0] + (i+1) * space, origin[1] ))
            qnode_pos_list.append((origin[0] - (i+1) * space, origin[1] ))
    return qnode_pos_list





# function that draws fsc graph
def fsc_graph(model,colors_dict):
    # get action select and node transitions
    actionselect, nodetrans = actionselect_nodetrans_dict(model.x)
    # print("xv",model.x.get_values())
    # print("as", actionselect)
    # print("nt", nodetrans)
    # create names for everything
    nodes = [ "q" + str(i) for i in model.q]
    states = ["s" + str(i) for i in model.s]
    actions = ["a" + str(i) for i in model.a]
    observations = ["o" + str(i) for i in model.o]
    # print(observations)
    # colors for observations
    obs_color = {obs: colors_dict[list(colors_dict.keys())[i]] for i, obs in enumerate(observations)}
    # print(obs_color)
    # Create graph
    G = nx.MultiDiGraph() # multi directed graph
    # action node list
    actionnodelist = []
    # dictionary for labels
    labels_dict = {}
    # adding nodes and edges
    for qnode in nodes:
        # add node and its label
        G.add_node(qnode)
        labels_dict[qnode] = qnode
        # add action node associated to qnode and edge between them
        for actionnode in actions:
            # if the actionnode contributes
            if round(actionselect[(int(actionnode[1]), int(qnode[1]))], 1) > 0:
                # print(f"as{(int(actionnode[1]), int(qnode[1]))}: {actionselect[(int(actionnode[1]), int(qnode[1]))]}")
                # add the node to the list for later drawing
                actionnodelist.append((actionnode, qnode))
                labels_dict[(actionnode, qnode)] = actionnode
                # edge where you have P(a | q)
                G.add_edge(qnode, (actionnode, qnode), 
                probability = actionselect[(int(actionnode[1]), int(qnode[1]))],
                color = '#000000'
                )
                for qprime, obs in product(nodes,observations):
                    # (q',a,q,o)
                    index = (int(qprime[1]), int(actionnode[1]), int(qnode[1]), int(obs[1]))
                    if round(nodetrans[index], 1) > 0:
                        # add P(q' | q, a, o) edge
                        # print(f"nt{index}{nodetrans[index]}")
                        G.add_edge((actionnode, qnode), qprime, observation = obs, 
                        probability = nodetrans[index],
                        color = obs_color[obs])
    # radius surrounding origin for controller nodes
    radius1 = 1
    # radius surrounding controller node for action nodes
    radius2 = 0.5
    # get position of controller node surrounding the origin
    qnodepos = action_node_pos(len(nodes),origin = (0,0), radius = radius1)
    pos = {}
    for qnode in nodes:
        # add position of qnode 
        pos[qnode] = qnodepos.pop(0)
        # position action nodes surrounding the controller node
        actionnodepos = action_node_pos(len(actionnodelist), pos[qnode], radius = radius2)
        for actionnode in actions:
            if (actionnode, qnode) in actionnodelist:
                pos[(actionnode, qnode)] = actionnodepos.pop(0)

    # Draw nodes
    # controller nodes
    nx.draw_networkx_nodes(G, pos,nodes, node_size=500, node_color='skyblue')
    # action nodes
    nx.draw_networkx_nodes(G, pos,actionnodelist, node_size=500, node_color='red', node_shape= "s")
    # Draw edges
    for node1, node2 in product(G.nodes, G.nodes):
        if G.has_edge(node1, node2):
            for key in G[node1][node2].keys():
                if 'observation' not in G[node1][node2][key].keys(): 
                    nx.draw_networkx_edges(G, pos, [(node1, node2, key)], width= G[node1][node2][key]['probability'],
                                       arrows=True,  edge_color = G[node1][node2][key]['color'],
                                       connectionstyle='arc3, rad = 0.3', arrowsize = 20)
                else:
                    obsindex = int(G[node1][node2][key]['observation'][1])
                    nx.draw_networkx_edges(G, pos, [(node1, node2, key)], width= G[node1][node2][key]['probability'], 
                                        arrows=True,  edge_color = G[node1][node2][key]['color'],
                                        connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*(1+obsindex))), 
                                        arrowsize = 20)
    # Draw labels
    # actionnode labels
    nx.draw_networkx_labels(G, pos,
                            labels = {key: labels_dict[key] 
                            for key in labels_dict.keys() if type(key) == tuple}, 
                            font_size=10, font_family="sans-serif",
                            font_color = "w")
    # qnode labels
    nx.draw_networkx_labels(G, pos,
                            labels = {key: labels_dict[key] 
                            for key in labels_dict.keys() if type(key) == str}, 
                            font_size=10, font_family="sans-serif",
                            font_color = "k")
    # legend
    # Get unique edge colors
    legend_elements = [Line2D([0], [0], color=color, lw=2,
                      label=f'{[key for key in obs_color.keys() if obs_color[key] == color][0]}') 
                      for color in obs_color.values()]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title('Finite State Controller')
    plt.show()    





# function that returns Transition dict
def capital_transition(model, state_transition_model, observation_model):
    T = {}
    for (sprime,qprime,state,qnode) in product(model.s,model.q,model.s,model.q):
        # T(s',m' | s,m) = sum_a,y P(a | m) P(y | s',a) P(s' | s,a) P(m' | m,a,y)
        T[sprime,qprime,state,qnode] = sum(value(model.x[qprime,actionn,qnode,obs]) *
                                           state_transition_model[sprime,actionn,state]*
                                           observation_model[sprime,actionn,obs]
                                           for actionn,obs in product(model.a,model.o)
                                           )
    return T    





# function that returns occupancy dict
def eta_calculation(model,state_transition_model, observation_model, gamma):
    # get T
    T = capital_transition(model, state_transition_model, observation_model)
    # for all s',m' rho(s',m') = eta(s',m') - sum_s,m gamma* T(s',m' | s,m) eta(s,m)
    # solve system of linear equations
    # matrix of coefficients
    a = np.zeros((len(model.s)*len(model.q),len(model.s)*len(model.q)))
    # vector of constants which is rho(s',m')
    b = np.ones(len(model.s)*len(model.q)) / (len(model.s) * len(model.q))
    # fill a
    for i, index in enumerate(product(model.s,model.q)):
        for j, secindex in enumerate(product(model.s,model.q)):
            if i == j:
                # coefficient of eta(s',m') = 1 - gamma * T(s',m' | s,m)
                a[i,j] = 1 - gamma * T[index[0], index[1], secindex[0], secindex[1]]
            else:
                # coefficient of eta(s,m)  = - gamma * T(s',m' | s,m)
                a[i,j] = - gamma * T[index[0], index[1], secindex[0], secindex[1]]
    # solve the system of linear equations
    sol = list(np.linalg.solve(a, b))
    # get eta with index
    eta = {key: sol[index] for index, key in enumerate(product(model.s,model.q))}
    return eta


