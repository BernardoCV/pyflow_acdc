# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:24:05 2024

@author: BernardoCastro
"""
import numpy as np
import pyomo.environ as pyo


weights_def = {
    'Ext_Gen'         : {'w': 0},
    'Market_sale'     : {'w': 0},
    'AC_losses'       : {'w': 1},
    'DC_losses'       : {'w': 1},
    'Converter_Losses': {'w': 1}
}
def add_extGrid(Grid, node_name, price=1):
    for node in Grid.nodes_AC:
        if node_name == node.name:
            node.extGrid = True
            node.price = price
            if node.type!= 'PV':
                node.PGi = 0



def OPF_ACDC(grid,ObjRule=weights_def,kap=1e-3,PV_set=False):
    model= OPF_createModel(grid,PV_set)
      
    """
    """
    obj_rule= OPF_obj(model,grid,ObjRule,kap)
    
    """
    """

    
    if any(conv.OPF_fx for conv in grid.Converters_ACDC):
            fx_conv(model, grid)
                
                
    """
    """
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    opt = pyo.SolverFactory("ipopt")
    results = opt.solve(model)

    ExportACDC_model_toPyflowACDC(grid,model)   
    s=1   
    return model, results

def fx_conv(model,grid):
    def fx_PDC(model,conv):
        if grid.Converters_ACDC[conv].OPF_fx==True and grid.Converters_ACDC[conv].OPF_fx_type=='PDC':
            return model.P_conv_DC[conv]==grid.Converters_ACDC[conv].P_DC
        else:
            return pyo.Constraint.Skip
    def fx_PAC(model,conv):   
        if grid.Converters_ACDC[conv].OPF_fx==True and (grid.Converters_ACDC[conv].OPF_fx_type=='PQ' or grid.Converters_ACDC[conv].OPF_fx_type=='PV'):
            return model.P_conv_s_AC[conv]==grid.Converters_ACDC[conv].P_AC
        else:
            return pyo.Constraint.Skip
    def fx_QAC(model,conv):    
        if grid.Converters_ACDC[conv].OPF_fx==True and grid.Converters_ACDC[conv].OPF_fx_type=='PQ':
            return model.Q_conv_s_AC[conv]==grid.Converters_ACDC[conv].Q_AC
        else:
            return pyo.Constraint.Skip
        
    model.Conv_fx_pdc=pyo.Constraint(model.conv,rule=fx_PDC)
    model.Conv_fx_pac=pyo.Constraint(model.conv,rule=fx_PAC)
    model.Conv_fx_qac =pyo.Constraint(model.conv,rule=fx_QAC)

def OPF_obj(model,grid,ObjRule,kap):
    
               
    obj_expr=0
    
    for node in  model.nodes_AC:
        nAC=grid.nodes_AC[node]
        if nAC.Num_conv_connected >= 2:
            obj_expr += sum(model.Q_conv_s_AC[conv]**2 for conv in nAC.connected_conv)
    
    
    
    
    def formula_Min_Ext_Gen():
        return sum((model.P_Gen[node]**2 + (model.Q_Gen[node]**2)) for node in model.nodes_AC)

    def formula_Market_sale():
        return sum((model.P_Gen[node] * model.price[node] + (model.Q_Gen[node]**2)) for node in model.nodes_AC)

    def formula_AC_losses():
        return sum(model.PAC_line_loss[line] for line in model.lines_AC)

    def formula_DC_losses():
        return sum(model.PDC_line_loss[line] for line in model.lines_DC)

    def formula_Converter_Losses():
        return sum(model.P_conv_loss[conv] for conv in model.conv)
               
    for key, entry in ObjRule.items():
        if key == 'Ext_Gen':
            entry['f'] = formula_Min_Ext_Gen()
        elif key == 'Market_sale':
            entry['f'] = formula_Market_sale()
        elif key == 'AC_losses':
            entry['f'] = formula_AC_losses()
        elif key == 'DC_losses':
            entry['f'] = formula_DC_losses()
        elif key == 'Converter_Losses':
            entry['f'] = formula_Converter_Losses()
    s=1
    total_weight = sum(entry['w'] for entry in ObjRule.values())
    weighted_sum = sum(entry['w'] / total_weight * entry['f'] for entry in ObjRule.values())+ kap*obj_expr
    return weighted_sum



def OPF_createModel(grid,PV_set):
    
    """Translation of element wise to internal numbering
    """

    lista_nodos_AC = list(range(0, grid.nn_AC))
    lista_lineas_AC = list(range(0,grid.nl_AC))
    lista_nodos_DC = list(range(0, grid.nn_DC))
    lista_lineas_DC = list(range(0,grid.nl_DC))
    lista_conv = list(range(0, grid.nconv))
    price = {}
    V_ini_AC = {}
    Theta_ini = {}
    P_known_AC = {}
    Q_know = {}
    P_conv_AC = {}
    Q_conv_AC = {}
    S_lineAC_limit ={}
    P_lineDC_limit ={}
    S_limit = {}
    S_limit_conv={}
    V_ini_DC = {}
    P_known_DC = {}
    P_conv_DC = {}
    P_conv_limit = {}
    u_min_ac=list(range(0, grid.nn_AC))
    u_min_dc=list(range(0, grid.nn_DC))
    u_max_ac=list(range(0, grid.nn_AC))
    u_max_dc=list(range(0, grid.nn_DC))
    u_c_min=list(range(0, grid.nconv))
    u_c_max=list(range(0, grid.nconv))
    

    P_conv_loss = {}

    AC_nodes_connected_conv = []
    DC_nodes_connected_conv = []

    AC_nodes_extGrid = []
    AC_slack = []
    AC_PV = []

    DC_slack = []

    for n in grid.nodes_AC:
        if n.type == 'Slack':
            AC_slack.append(n.nodeNumber)
        if n.type == 'PV':
            AC_PV.append(n.nodeNumber)
        if n.extGrid == True:
            AC_nodes_extGrid.append(n.nodeNumber)
    for n in grid.nodes_DC:
        if n.type == 'Slack':
            DC_slack.append(n.nodeNumber)
    s = 1


    for n in grid.nodes_AC:
        V_ini_AC[n.nodeNumber] = n.V_ini
        Theta_ini[n.nodeNumber] = n.theta_ini
        P_known_AC[n.nodeNumber] = n.PGi-n.PLi
        Q_know[n.nodeNumber] = n.QGi-n.QLi
        u_min_ac[n.nodeNumber] = n.Umin
        u_max_ac[n.nodeNumber] = n.Umax
        P_conv_AC[n.nodeNumber] = 0
        Q_conv_AC[n.nodeNumber] = 0
        S_limit[n.nodeNumber] = 0
        price[n.nodeNumber] = n.price

    for l in grid.lines_AC:
        S_lineAC_limit[l.lineNumber]= l.MVA_rating/grid.S_base
    
    for n in grid.nodes_DC:
        V_ini_DC[n.nodeNumber] = n.V_ini
        P_known_DC[n.nodeNumber] = n.P_DC
        P_conv_DC[n.nodeNumber] = 0
        P_conv_limit[n.nodeNumber] = 0
        u_min_dc[n.nodeNumber] = n.Umin
        u_max_dc[n.nodeNumber] = n.Umax
    for l in grid.lines_DC:
        P_lineDC_limit[l.lineNumber]= l.MW_rating/grid.S_base
        
    for conv in grid.Converters_ACDC:
        AC_nodes_connected_conv.append(conv.Node_AC.nodeNumber)
        DC_nodes_connected_conv.append(conv.Node_DC.nodeNumber)

        P_conv_AC[conv.Node_AC.nodeNumber] = conv.P_AC
        Q_conv_AC[conv.Node_AC.nodeNumber] = conv.Q_AC
        S_limit[conv.Node_AC.nodeNumber] += conv.MVA_max
        S_limit_conv[conv.ConvNumber] = conv.MVA_max

        P_conv_DC[conv.Node_DC.nodeNumber] = conv.P_DC
        P_conv_limit[conv.Node_DC.nodeNumber] = conv.MVA_max
        
        u_c_min[conv.ConvNumber] = conv.Ucmin
        u_c_max[conv.ConvNumber] = conv.Ucmax
        

        P_conv_loss[conv.ConvNumber] = conv.P_loss

    AC_nodes_no_extGrid = [node for node in lista_nodos_AC if node not in AC_nodes_extGrid]


    """
    MODEL INITIATION
    """



    model = pyo.ConcreteModel()
    
    model.name="AC included Converter model"
    
    "Model Sets"
    model.nodes_AC   = pyo.Set(initialize=lista_nodos_AC)
    model.lines_AC   = pyo.Set(initialize=lista_lineas_AC)
    model.AC_no_extG = pyo.Set(initialize=AC_nodes_no_extGrid)
    model.AC_slacks  = pyo.Set(initialize=AC_slack)
    model.AC_PVs     = pyo.Set(initialize=AC_PV)

    model.nodes_DC   = pyo.Set(initialize=lista_nodos_DC)
    model.lines_DC   = pyo.Set(initialize=lista_lineas_DC)
    model.DC_slacks  = pyo.Set(initialize=DC_slack)

    model.conv       = pyo.Set(initialize=lista_conv)
     
    """Variables and limits
    """
    ### AC Variables
    #AC nodes variables
    model.V_AC       = pyo.Var(model.nodes_AC, bounds=lambda model, node: (u_min_ac[node], u_max_ac[node]), initialize=V_ini_AC)
    model.thetha_AC  = pyo.Var(model.nodes_AC, bounds=(-1.6, 1.6), initialize=Theta_ini)

    model.P_known_AC = pyo.Param(model.nodes_AC, initialize=P_known_AC)
    model.Q_known_AC = pyo.Param(model.nodes_AC, initialize=Q_know)
    model.price      = pyo.Param(model.nodes_AC, initialize=price)
    
       
    def P_Gen_bounds(model, node):
        if node in AC_nodes_no_extGrid or grid.nodes_AC[node].type=='PV':
            return (0, 0)
    def Q_Gen_bounds(model, node):
        if node in AC_nodes_no_extGrid:
            return (0, 0)
         
    model.P_Gen = pyo.Var(model.nodes_AC, bounds=P_Gen_bounds, initialize=0)
    model.Q_Gen = pyo.Var(model.nodes_AC, bounds=Q_Gen_bounds, initialize=0)
    
    def AC_V_slack_rule(model, node):
        return model.V_AC[node] == V_ini_AC[node]

    def AC_theta_slack_rule(model, node):
        return model.thetha_AC[node] == Theta_ini[node]

    def AC_V_PV_rule(model, node):
        return model.V_AC[node] == V_ini_AC[node]

    
    model.AC_theta_slack_constraint = pyo.Constraint(model.AC_slacks, rule=AC_theta_slack_rule)
    if PV_set == True:
        model.AC_V_slack_constraint = pyo.Constraint(model.AC_slacks, rule=AC_V_slack_rule)
        model.AC_V_PV_constraint = pyo.Constraint(model.AC_PVs, rule=AC_V_PV_rule)
    
    #AC Lines variables
    def Sbounds_lines(model, line):
        return (-S_lineAC_limit[line], S_lineAC_limit[line])
    
    model.PAC_to       = pyo.Var(model.lines_AC, bounds=Sbounds_lines, initialize=0)
    model.PAC_from     = pyo.Var(model.lines_AC, bounds=Sbounds_lines, initialize=0)
    model.QAC_to       = pyo.Var(model.lines_AC, bounds=Sbounds_lines, initialize=0)
    model.QAC_from     = pyo.Var(model.lines_AC, bounds=Sbounds_lines, initialize=0)
    model.PAC_line_loss= pyo.Var(model.lines_AC, initialize=0)
    
    ### DC variables
    #DC nodes variables
    model.V_DC = pyo.Var(model.nodes_DC, bounds=lambda model, node: (u_min_dc[node], u_max_dc[node]), initialize=V_ini_DC)
    model.P_known_DC = pyo.Param(model.nodes_DC, initialize=P_known_DC)
    
    def DC_V_slack_rule(model, node):
        return model.V_DC[node] == V_ini_DC[node]
    
    model.DC_V_slack_constraint = pyo.Constraint(model.DC_slacks, rule=DC_V_slack_rule)
    
    #DC Lines variables
    def Pbounds_lines(model, line):
        return (-P_lineDC_limit[line], P_lineDC_limit[line])
    
    model.PDC_to       = pyo.Var(model.lines_DC,bounds=Pbounds_lines ,  initialize=0)
    model.PDC_from     = pyo.Var(model.lines_DC,bounds=Pbounds_lines , initialize=0)
    model.PDC_line_loss= pyo.Var(model.lines_DC,bounds=Pbounds_lines , initialize=0)
    
    ### Converter Variables
    model.Uc   = pyo.Var(model.conv, bounds=lambda model, conv: (u_c_min[conv], u_c_max[conv]), initialize=1) 
    model.Uf   = pyo.Var(model.conv, bounds=lambda model, conv: (u_c_min[conv], u_c_max[conv]), initialize=1) 
    model.th_c   = pyo.Var(model.conv, bounds=(-1.6, 1.6), initialize=0) 
    model.th_f   = pyo.Var(model.conv, bounds=(-1.6, 1.6), initialize=0) 
    
    def P_conv_bounds(model, node):
        return (-P_conv_limit[node], P_conv_limit[node])
    
    def Sbounds(model, node):
        return (-S_limit[node], S_limit[node])

    def Sbounds_conv(model, conv):
         return (-S_limit_conv[conv], S_limit_conv[conv])
    
    model.P_conv_loss = pyo.Var(model.conv, initialize=P_conv_loss)
    
    model.P_conv_DC = pyo.Var(model.nodes_DC, bounds=P_conv_bounds, initialize=0)
    
    model.P_conv_AC = pyo.Var(model.nodes_AC, bounds=Sbounds, initialize=0)
    model.Q_conv_AC = pyo.Var(model.nodes_AC, bounds=Sbounds, initialize=0)
    
    model.P_conv_s_AC  = pyo.Var(model.conv, bounds=Sbounds_conv, initialize=0)   
    model.Q_conv_s_AC = pyo.Var(model.conv, bounds=Sbounds_conv, initialize=0)

    model.P_conv_c_AC  = pyo.Var(model.conv, bounds=Sbounds_conv, initialize=0)   
    model.Q_conv_c_AC = pyo.Var(model.conv, bounds=Sbounds_conv, initialize=0)
    
    """EQUALITY CONSTRAINTS
    """
    
    ### AC constraints
    # AC node constraints
    def P_AC_node_rule(model, node):
        P_sum = sum(
                model.V_AC[node] * model.V_AC[k] *
                (np.real(grid.Ybus_AC[node, k]) * pyo.cos(model.thetha_AC[node] - model.thetha_AC[k]) +
                 np.imag(grid.Ybus_AC[node, k]) * pyo.sin(model.thetha_AC[node] - model.thetha_AC[k]))
                for k in model.nodes_AC if grid.Ybus_AC[node, k] != 0   )   

        return P_sum == model.P_known_AC[node] + model.P_conv_AC[node] + model.P_Gen[node]

    def Q_AC_node_rule(model, node):

        Q_sum = sum(
            model.V_AC[node] * model.V_AC[k] *
            (np.real(grid.Ybus_AC[node, k]) * pyo.sin(model.thetha_AC[node] - model.thetha_AC[k]) -
             np.imag(grid.Ybus_AC[node, k]) * pyo.cos(model.thetha_AC[node] - model.thetha_AC[k]))
            for k in model.nodes_AC if grid.Ybus_AC[node, k] != 0)

        return Q_sum == model.Q_known_AC[node] + model.Q_conv_AC[node] + model.Q_Gen[node]

    model.P_AC_node_constraint = pyo.Constraint(model.nodes_AC, rule=P_AC_node_rule)
    model.Q_AC_node_constraint = pyo.Constraint(model.nodes_AC, rule=Q_AC_node_rule)
    
    # AC line equality constraints
    
    def P_to_AC_line(model,line):   
        #        Iij_cart[j, i] = (V_cart[i]-V_cart[j]) * Ybus[j, i]+V_cart[j]*l.Y/2
        #        Sij[j, i] = V_cart[j]*np.conj(Iij_cart[j, i])

        l = grid.lines_AC[line]
        f = l.fromNode.nodeNumber
        t = l.toNode.nodeNumber
        G_bus=np.real(grid.Ybus_AC[t,f])
        B_bus=np.imag(grid.Ybus_AC[t,f])
        G= np.real(l.Y)
        B= np.imag(l.Y)
        Pto = (model.V_AC[t]*(G*model.V_AC[t] - 2*G_bus*model.V_AC[t] + 2*G_bus*model.V_AC[f]*pyo.cos(model.thetha_AC[f] - model.thetha_AC[t]) -\
                        2*B_bus*model.V_AC[f]*pyo.sin(model.thetha_AC[f] - model.thetha_AC[t])))/2
        
        return model.PAC_to[line] == Pto
    
    def P_from_AC_line(model,line):       
        #        Iij_cart[i, j] = (V_cart[j]-V_cart[i]) * Ybus[i, j]+V_cart[i]*l.Y/2
        #        Sij[i, j] = V_cart[i]*np.conj(Iij_cart[i, j])
      
        l = grid.lines_AC[line]
        f = l.fromNode.nodeNumber
        t = l.toNode.nodeNumber
        G_bus=np.real(grid.Ybus_AC[f,t])
        B_bus=np.imag(grid.Ybus_AC[f,t])
        G= np.real(l.Y)
        B= np.imag(l.Y)
        Pfrom = (model.V_AC[f]*(G*model.V_AC[f] - 2*G_bus*model.V_AC[f] + 2*G_bus*model.V_AC[t]*pyo.cos(model.thetha_AC[f] - model.thetha_AC[t]) +\
                        2*B_bus*model.V_AC[t]*pyo.sin(model.thetha_AC[f] - model.thetha_AC[t])))/2
 

        return model.PAC_from[line] == Pfrom
    
    def Q_to_AC_line(model,line):   
        #        Iij_cart[j, i] = (V_cart[i]-V_cart[j]) * Ybus[j, i]+V_cart[j]*l.Y/2
        #        Sij[j, i] = V_cart[j]*np.conj(Iij_cart[j, i])

        l = grid.lines_AC[line]
        f = l.fromNode.nodeNumber
        t = l.toNode.nodeNumber
        G_bus=np.real(grid.Ybus_AC[t,f])
        B_bus=np.imag(grid.Ybus_AC[t,f])
        G= np.real(l.Y)
        B= np.imag(l.Y)
        Qto = -(model.V_AC[t]*(B*model.V_AC[t] - 2*B_bus*model.V_AC[t] + 2*B_bus*model.V_AC[f]*pyo.cos(model.thetha_AC[f] - model.thetha_AC[t]) +\
                         2*G_bus*model.V_AC[f]*pyo.sin(model.thetha_AC[f] - model.thetha_AC[t])))/2
         
        
        return model.QAC_to[line] == Qto
    
    def Q_from_AC_line(model,line):       
        #        Iij_cart[i, j] = (V_cart[j]-V_cart[i]) * Ybus[i, j]+V_cart[i]*l.Y/2
        #        Sij[i, j] = V_cart[i]*np.conj(Iij_cart[i, j])
        # B_bus*(Va*cos(theta_a) - Vb*cos(theta_b)) + G_bus*(Va*sin(theta_a) - Vb*sin(theta_b)) + (B*Vb*cos(theta_b))/2 + (G*Vb*sin(theta_b))/2
 
        l = grid.lines_AC[line]
        f = l.fromNode.nodeNumber
        t = l.toNode.nodeNumber
        G_bus=np.real(grid.Ybus_AC[f,t])
        B_bus=np.imag(grid.Ybus_AC[f,t])
        G= np.real(l.Y)
        B= np.imag(l.Y)
        Qfrom = -(model.V_AC[f]*(B*model.V_AC[f] - 2*B_bus*model.V_AC[f] + 2*B_bus*model.V_AC[t]*pyo.cos(model.thetha_AC[f] - model.thetha_AC[t]) -\
                         2*G_bus*model.V_AC[t]*pyo.sin(model.thetha_AC[f] - model.thetha_AC[t])))/2

        return model.QAC_from[line] == Qfrom
    
    def P_loss_AC_rule(model,line):
        return model.PAC_line_loss[line]== model.PAC_to[line]+model.PAC_from[line]
    
    
    model.Pto_AC_line_constraint   = pyo.Constraint(model.lines_AC, rule=P_to_AC_line)
    model.Pfrom_AC_line_constraint = pyo.Constraint(model.lines_AC, rule=P_from_AC_line)
    model.Qto_AC_line_constraint   = pyo.Constraint(model.lines_AC, rule=Q_to_AC_line)
    model.Qfrom_AC_line_constraint = pyo.Constraint(model.lines_AC, rule=Q_from_AC_line)
    model.P_AC_loss_constraint     =pyo.Constraint(model.lines_AC, rule=P_loss_AC_rule)
    
    
    
    
    ### DC constraints
    #DC node constraints
    def P_DC_node_rule(model, node):
        i = node
        P_sum = 0
        for k in range(grid.nn_DC):
            Y = grid.Ybus_DC[i, k]

            if k != i:
                if Y != 0:
                    line = grid.get_lineDC_by_nodes(i, k)
                    pol = line.pol
                    Y = -Y
                    P_sum += pol*model.V_DC[i] * \
                        (model.V_DC[i]-model.V_DC[k])*Y

        return P_sum == model.P_known_DC[node] + model.P_conv_DC[node]

    def P_DC_noconv_rule(model, node):
        return model.P_conv_DC[node] == 0

    model.P_DC_node_constraint = pyo.Constraint(model.nodes_DC, rule=P_DC_node_rule)
    
    #DC lines equality constraints
    
    def P_from_DC_line(model,line):       
        l = grid.lines_DC[line]
        f = l.fromNode.nodeNumber
        t = l.toNode.nodeNumber
        pol = l.pol
        
        Pfrom= (model.V_DC[t]-model.V_DC[f])*grid.Ybus_DC[f,t]*model.V_DC[f]*pol
        
        return model.PDC_from[line] == Pfrom
    
    def P_to_DC_line(model,line):   
        l = grid.lines_DC[line]
        f = l.fromNode.nodeNumber
        t = l.toNode.nodeNumber
        pol = l.pol

         
        Pto= (model.V_DC[f]-model.V_DC[t])*grid.Ybus_DC[t,f]*model.V_DC[t]*pol 
        
        
        return model.PDC_to[line] == Pto
    
    def P_loss_DC_line_rule(model,line):
        
        return model.PDC_line_loss[line]==model.PDC_from[line]+ model.PDC_to[line]
    
    model.Pfrom_DC_line_constraint   = pyo.Constraint(model.lines_DC, rule=P_from_DC_line)
    model.Pto_DC_line_constraint     = pyo.Constraint(model.lines_DC, rule=P_to_DC_line)
    model.Ploss_DC_line_constraint   = pyo.Constraint(model.lines_DC, rule=P_loss_DC_line_rule)    
    
    
    ### Converter Constraints    
    
    def Conv_Ps_rule(model,conv):
       element=grid.Converters_ACDC[conv]
       nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
       nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
        
       Gc = element.Gc
       Bc = element.Bc
       Gtf = element.Gtf
       Btf = element.Btf
       Bf = element.Bf
       
       if element.Bf == 0:
           Ztf = element.Ztf
           Zc = element.Zc
           Zeq = Ztf+Zc
           Yeq = 1/Zeq

           Gc = np.real(Yeq)
           Bc = np.imag(Yeq)
           
           Ps = -model.V_AC[nAC]**2*Gc+model.V_AC[nAC]*model.Uc[conv] * \
               (Gc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv])+Bc*pyo.sin(model.thetha_AC[nAC]-model.th_c[conv]))
          

       elif element.Gtf == 0:
   
           Bcf = Bc+Bf

           Ps = -model.V_AC[nAC]**2*Gc+model.V_AC[nAC]*model.Uc[conv] * \
               (Gc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv])+Bc*pyo.sin(model.thetha_AC[nAC]-model.th_c[conv]))
          
           
       else:

           Ps = -model.V_AC[nAC]**2*Gtf+model.V_AC[nAC]*model.Uf[conv] * \
               (Gtf*pyo.cos(model.thetha_AC[nAC]-model.th_f[conv])+Btf*pyo.sin(model.thetha_AC[nAC]-model.th_f[conv]))
           
       return model.P_conv_s_AC[conv]-Ps==0
           
    def Conv_Qs_rule(model,conv):
       element=grid.Converters_ACDC[conv]
       nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
       nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
       
       Gc = element.Gc
       Bc = element.Bc
       Gtf = element.Gtf
       Btf = element.Btf
       Bf = element.Bf
        
       if element.Bf == 0:
           Ztf = element.Ztf
           Zc = element.Zc
           Zeq = Ztf+Zc
           Yeq = 1/Zeq

           Gc = np.real(Yeq)
           Bc = np.imag(Yeq)
           
           Qs = model.V_AC[nAC]**2*Bc+model.V_AC[nAC]*model.Uc[conv]*(Gc*pyo.sin(model.thetha_AC[nAC]-model.th_c[conv])-Bc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv]))

       elif element.Gtf == 0:
  
           Bcf = Bc+Bf

           Qs = model.V_AC[nAC]**2*Bcf+model.V_AC[nAC]*model.Uc[conv] * \
                (Gc*pyo.sin(model.thetha_AC[nAC]-model.th_f[conv])-Bc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv]))
         
       else:
                         
           Qs = model.V_AC[nAC]**2*Btf+model.V_AC[nAC]*model.Uf[conv] * \
               (Gtf*pyo.sin(model.thetha_AC[nAC]-model.th_f[conv])-Btf*pyo.cos(model.thetha_AC[nAC]-model.th_f[conv]))

       return model.Q_conv_s_AC[conv]-Qs==0
       
   
    

    def Conv_Pc_rule(model,conv):
       element=grid.Converters_ACDC[conv]
       nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
       nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
       
       Gc = element.Gc
       Bc = element.Bc
       Gtf = element.Gtf
       Btf = element.Btf
       Bf = element.Bf
       
       if element.Bf == 0:
           Ztf = element.Ztf
           Zc = element.Zc
           Zeq = Ztf+Zc
           Yeq = 1/Zeq

           Gc = np.real(Yeq)
           Bc = np.imag(Yeq)
           
           Pc = model.Uc[conv]**2*Gc-model.V_AC[nAC]*model.Uc[conv] * \
               (Gc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv])-Bc*pyo.sin(model.thetha_AC[nAC]-model.th_c[conv]))
          

       elif element.Gtf == 0:
                    
           Bcf = Bc+Bf
        
           Pc = model.Uc[conv]**2*Gc-model.V_AC[nAC]*model.Uc[conv]*(Gc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv])-Bc*pyo.sin(model.thetha_AC[nAC]-model.th_c[conv]))
           
           
       else:
           
           Pc = model.Uc[conv]**2*Gc-model.Uf[conv]*model.Uc[conv]*(Gc*pyo.cos(model.th_f[conv]-model.th_c[conv])-Bc*pyo.sin(model.th_f[conv]-model.th_c[conv]))
           
           
       return -Pc+model.P_conv_c_AC[conv]==0
           
    def Conv_Qc_rule(model,conv):
       element=grid.Converters_ACDC[conv]
       nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
       nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
        
       Gc = element.Gc
       Bc = element.Bc
       Gtf = element.Gtf
       Btf = element.Btf
       Bf = element.Bf
       
       if element.Bf == 0:
           Ztf = element.Ztf
           Zc = element.Zc
           Zeq = Ztf+Zc
           Yeq = 1/Zeq

           Gc = np.real(Yeq)
           Bc = np.imag(Yeq)
           
           
           Qc = -model.Uc[conv]**2*Bc+model.V_AC[nAC]*model.Uc[conv] * \
               (Gc*pyo.sin(model.thetha_AC[nAC]-model.th_c[conv])+Bc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv]))
          

       elif element.Gtf == 0:
           
           Bcf = Bc+Bf

           Qc = -model.Uc[conv]*model.Uc[conv]*Bc+model.V_AC[nAC]*model.Uc[conv] * \
               (Gc*pyo.sin(model.thetha_AC[nAC]-model.th_c[conv])+Bc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv]))

          
           
       else:
           
           Qc = -model.Uc[conv]*model.Uc[conv]*Bc+model.Uf[conv]*model.Uc[conv] * \
               (Gc*pyo.sin(model.th_f[conv]-model.th_c[conv])+Bc*pyo.cos(model.th_f[conv]-model.th_c[conv]))

       return -Qc+model.Q_conv_c_AC[conv]==0




    def Conv_F1_rule(model,conv):
       element=grid.Converters_ACDC[conv]
       nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
       nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
        
       Gc = element.Gc
       Bc = element.Bc
       Gtf = element.Gtf
       Btf = element.Btf
       Bf = element.Bf
       
       if element.Bf == 0 or element.Gtf == 0:
        return pyo.Constraint.Skip
           
       else:
                
           Psf = model.Uf[conv]*model.Uf[conv]*Gtf-model.Uf[conv]*model.V_AC[nAC] * \
               (Gtf*pyo.cos(model.thetha_AC[nAC]-model.th_f[conv])-Btf*pyo.sin(model.thetha_AC[nAC]-model.th_f[conv]))
      
           Pcf = -model.Uf[conv]*model.Uf[conv]*Gc+model.Uf[conv]*model.Uc[conv] * \
               (Gc*pyo.cos(model.th_f[conv]-model.th_c[conv])+Bc*pyo.sin(model.th_f[conv]-model.th_c[conv]))
        

           F1 = Pcf-Psf
         
           
            
       return F1==0

    def Conv_F2_rule(model,conv):
       element=grid.Converters_ACDC[conv]
       nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
       nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
       constraints = pyo.ConstraintList()
       
       if element.Bf == 0 or element.Gtf == 0:
        return pyo.Constraint.Skip
           
       else:
           
           Gc = element.Gc
           Bc = element.Bc
           Gtf = element.Gtf
           Btf = element.Btf
           Bf = element.Bf

         
           Qsf = -model.Uf[conv]**2*Btf+model.Uf[conv]*model.V_AC[nAC] * \
               (Gtf*pyo.sin(model.thetha_AC[nAC]-model.th_f[conv])+Btf*pyo.cos(model.thetha_AC[nAC]-model.th_f[conv]))

         
           Qcf = model.Uf[conv]**2*Bc+model.Uf[conv]*model.Uc[conv] * \
               (Gc*pyo.sin(model.th_f[conv]-model.th_c[conv])-Bc*pyo.cos(model.th_f[conv]-model.th_c[conv]))

           Qf = -model.Uf[conv]*model.Uf[conv]*Bf

           

           F2 = Qcf-Qsf-Qf
           
           
            
       return F2==0
    
    model.Conv_Ps_constraint = pyo.Constraint(model.conv,rule=Conv_Ps_rule)
    model.Conv_Qs_constraint = pyo.Constraint(model.conv,rule=Conv_Qs_rule)
    model.Conv_Pc_constraint = pyo.Constraint(model.conv,rule=Conv_Pc_rule)
    model.Conv_Qc_constraint = pyo.Constraint(model.conv,rule=Conv_Qc_rule)
    model.Conv_F1_constraint = pyo.Constraint(model.conv,rule=Conv_F1_rule)
    model.Conv_F2_constraint = pyo.Constraint(model.conv,rule=Conv_F2_rule)
    
    # Adds all converters in the AC nodes they are connected to
    def Conv_PAC_rule(model,node):
       nAC = grid.nodes_AC[node]
       P_conv=0
       for conv in nAC.connected_conv:
           P_conv += model.P_conv_s_AC[conv]
           
       return  model.P_conv_AC[node] ==   P_conv
           
    def Conv_Q_rule(model,node):
       nAC = grid.nodes_AC[node]
       Q_conv = sum(model.Q_conv_s_AC[conv] for conv in nAC.connected_conv)
    
      
       return   model.Q_conv_AC[node] ==   Q_conv       
         

    # IGBTs losses
    def Conv_DC_rule(model, conv):
        nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
        nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber

        return model.P_conv_c_AC[conv]+model.P_conv_DC[nDC] + model.P_conv_loss[conv] == 0

    def Conv_loss_rule(model, conv):
        nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
        nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
        a = grid.Converters_ACDC[conv].a_conv
        b = grid.Converters_ACDC[conv].b_conv
      

        current = (model.P_conv_c_AC[conv]+model.Q_conv_c_AC[conv])/(2*model.Uc[conv])
        currentsqr = (model.P_conv_c_AC[conv]**2+model.Q_conv_c_AC[conv]**2)/(model.Uc[conv]**2)


        c_inver=grid.Converters_ACDC[conv].c_inver
        c_rect=grid.Converters_ACDC[conv].c_rect
    
       
        if model.P_conv_DC[nDC].value < 0:
            P_loss = a +  c_inver * currentsqr
        else:
            P_loss = a +  c_rect * currentsqr
    
        
        return model.P_conv_loss[conv] == P_loss

    
    model.Conv_DC_constraint = pyo.Constraint(model.conv, rule=Conv_DC_rule)
    model.Conv_PAC_constraint = pyo.Constraint(model.nodes_AC, rule=Conv_PAC_rule)
    model.Conv_QAC_constraint = pyo.Constraint(model.nodes_AC, rule=Conv_Q_rule)
    model.Conv_loss_constraint = pyo.Constraint(model.conv, rule=Conv_loss_rule)
    
    
    """
    INEQUALITY CONSTRAINTS
    """
    #AC lines inequality
    def S_to_AC_limit_rule(model,line):
        
        return model.PAC_to[line]**2+model.QAC_to[line]**2 <= S_lineAC_limit[line]**2
    def S_from_AC_limit_rule(model,line):
        
        return model.PAC_from[line]**2+model.QAC_from[line]**2 <= S_lineAC_limit[line]**2
    
    
    model.S_to_AC_limit_constraint   = pyo.Constraint(model.lines_AC, rule=S_to_AC_limit_rule)
    model.S_from_AC_limit_constraint = pyo.Constraint(model.lines_AC, rule=S_from_AC_limit_rule)
    
    
    #DC lines inequality
    
    #they set in the variables themselves
    
    #Converters inequality 
    
    def Conv_AC_Limit_rule(model, conv):
        nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
        return model.P_conv_c_AC[conv]**2+model.Q_conv_c_AC[conv]**2 <= S_limit_conv[conv]**2*0.99
    
    model.Conv_AC_Limit_constraint = pyo.Constraint(model.conv, rule=Conv_AC_Limit_rule)
    
    return model




def OPF_conv_results(grid,model,Dict=False):
    lista_conv = list(range(0, grid.nconv))
    AC_nodes_extGrid = []
    AC_slack = []
    AC_PV = []

    DC_slack = []

    for n in grid.nodes_AC:
        if n.type == 'Slack':
            AC_slack.append(n.nodeNumber)
        if n.type == 'PV':
            AC_PV.append(n.nodeNumber)
        if n.extGrid == True:
            AC_nodes_extGrid.append(n.nodeNumber)
    for n in grid.nodes_DC:
        if n.type == 'Slack':
            DC_slack.append(n.nodeNumber)
    s = 1
   
    
    opt_res_P_conv_DC = np.zeros(grid.nconv)
    opt_res_P_conv_DC_dict = {}
    opt_res_P_conv_AC = np.zeros(grid.nconv)
    opt_res_P_conv_AC_dict = {}
    opt_res_Q_conv_AC = np.zeros(grid.nconv)
    opt_res_Q_conv_AC_dict = {}
    opt_res_P_loss_conv = np.zeros(grid.nconv)
    for conv in lista_conv:
        element= grid.Converters_ACDC[conv]
        name = element.name
        opt_res_P_conv_DC_dict[name] = pyo.value(model.P_conv_DC[element.Node_DC.nodeNumber])
        opt_res_P_conv_DC[conv]      = pyo.value(model.P_conv_DC[element.Node_DC.nodeNumber])
        
        opt_res_P_conv_AC_dict[name] = pyo.value(model.P_conv_s_AC[conv])
        opt_res_P_conv_AC[conv]      = pyo.value(model.P_conv_s_AC[conv])
        
        opt_res_Q_conv_AC_dict[name] = pyo.value(model.Q_conv_s_AC[conv])
        opt_res_Q_conv_AC[conv]      = pyo.value(model.Q_conv_s_AC[conv])
        
        opt_res_P_loss_conv[conv] = pyo.value(model.P_conv_loss[conv])


    opt_res_P_extGrid_dict = {}
    opt_res_Q_extGrid_dict  = {}
    opt_res_P_extGrid_dict ['Total'] = 0
    opt_res_Q_extGrid_dict ['Total'] = 0
    opt_res_P_extGrid = np.zeros(grid.nn_AC)
    opt_res_Q_extGrid = np.zeros(grid.nn_AC)
    for extGrid in AC_nodes_extGrid:
        name = grid.node_names_AC[extGrid]
        opt_res_P_extGrid_dict [name] = pyo.value(model.P_Gen[extGrid])
        opt_res_Q_extGrid_dict [name] = pyo.value(model.Q_Gen[extGrid])
        
        opt_res_P_extGrid[extGrid] = pyo.value(model.P_Gen[extGrid])
        opt_res_Q_extGrid[extGrid] = pyo.value(model.Q_Gen[extGrid])
        
        opt_res_P_extGrid_dict ['Total'] += pyo.value(model.P_Gen[extGrid])
        opt_res_Q_extGrid_dict ['Total'] += pyo.value(model.Q_Gen[extGrid])
    s=1
    
    if Dict== True:
        return opt_res_P_conv_DC_dict, opt_res_P_conv_AC_dict ,opt_res_Q_conv_AC_dict, opt_res_P_extGrid_dict,opt_res_Q_extGrid_dict
    else:
        return opt_res_P_conv_DC,opt_res_P_conv_AC,opt_res_Q_conv_AC,opt_res_P_extGrid,opt_res_Q_extGrid


def ExportACDC_model_toPyflowACDC(grid,model):
    grid.V_AC =np.zeros(grid.nn_AC)
    grid.Theta_V_AC=np.zeros(grid.nn_AC)
    grid.V_DC=np.zeros(grid.nn_DC)
    
    for node in grid.nodes_AC:
        nAC= node.nodeNumber
        node.V    =np.float64(pyo.value(model.V_AC[nAC]))
        node.theta=np.float64(pyo.value(model.thetha_AC[nAC]))
        node.P_s  =np.float64(pyo.value(model.P_conv_AC[nAC]))
        node.Q_s  =np.float64(pyo.value(model.Q_conv_AC[nAC]))
        node.P_Gen=np.float64(pyo.value(model.P_Gen[nAC]))
        node.Q_Gen=np.float64(pyo.value(model.Q_Gen[nAC]))
        node.P_INJ=node.PGi-node.PLi + node.P_s + node.P_Gen
        node.Q_INJ=node.QGi-node.QLi + node.Q_s + node.Q_Gen
        grid.V_AC[nAC]=node.V
        grid.Theta_V_AC[nAC]=node.theta
    for node in grid.nodes_DC:
        nDC = node.nodeNumber
        node.V    =np.float64(pyo.value(model.V_DC[nDC]))
        node.P    =np.float64(pyo.value(model.P_conv_DC[nDC]))
        node.P_INJ=node.PGi-node.PLi + node.P 
        grid.V_DC[nDC]=node.V
    for conv in grid.Converters_ACDC:
        nconv = conv.ConvNumber
        conv.P_DC  =np.float64(pyo.value(model.P_conv_DC[conv.Node_DC.nodeNumber]))
        conv.P_AC  =np.float64(pyo.value(model.P_conv_s_AC[nconv]))
        conv.Q_AC  =np.float64(pyo.value(model.Q_conv_s_AC[nconv]))
        conv.Pc    =np.float64(pyo.value(model.P_conv_c_AC[nconv]))
        conv.Qc    =np.float64(pyo.value(model.Q_conv_c_AC[nconv]))
        conv.P_loss=np.float64(pyo.value(model.P_conv_loss[nconv]))
        conv.P_loss_tf = abs(conv.P_AC-conv.Pc)
        conv.U_c   =np.float64(pyo.value(model.Uc[nconv]))
        conv.U_f   =np.float64(pyo.value(model.Uf[nconv]))
        conv.U_s   =np.float64(pyo.value(model.V_AC[conv.Node_AC.nodeNumber]))
        conv.th_c  =np.float64(pyo.value(model.th_c[nconv]))
        conv.th_f  =np.float64(pyo.value(model.th_f[nconv]))
        conv.th_s  =np.float64(pyo.value(model.thetha_AC[conv.Node_AC.nodeNumber]))
    grid.Line_AC_calc()
    grid.Line_DC_calc()