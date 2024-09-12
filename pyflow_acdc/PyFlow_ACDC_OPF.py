# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:24:05 2024

@author: BernardoCastro
"""
import numpy as np
import pyomo.environ as pyo
import pandas as pd

weights_def = {
    'Ext_Gen'         : {'w': 0},
    'Energy_cost'     : {'w': 0},
    'Curtailment_Red' : {'w': 0},
    'AC_losses'       : {'w': 0},
    'DC_losses'       : {'w': 0},
    'Converter_Losses': {'w': 0},
    'Social_Cost'     : {'w': 0},
    
}



 
def assign_nodeToMarket(Grid, node_name, new_market_name):
        """ Assign node to a new market and remove it from its previous market """
        new_market = None
        old_market = None
        node_to_reassign = None
        
        # Find the new market
        for market in Grid.Markets:
            if market.name == new_market_name:
                new_market = market
                break

        if new_market is None:
            raise ValueError(f"Market {new_market_name} not found.")
        
        # Remove node from its old market
        for market in Grid.Markets:
            for node in market.nodes_AC:
                if node.name == node_name:
                    old_market = market
                    node_to_reassign = node
                    break
            if old_market:
                break
            
        if old_market is not None:
            old_market.nodes_AC = [node for node in old_market.nodes_AC if node.name != node_name]
        
        # If the node was not found in any market, check Grid.nodes_AC
        if node_to_reassign is None:
            for node in Grid.nodes_AC:
                if node.name == node_name:
                    node_to_reassign = node
                    break
                
        if node_to_reassign is None:
            raise ValueError(f"Node {node_name} not found.")
        
        # Add node to the new market
        if node_to_reassign not in new_market.nodes_AC:
            new_market.nodes_AC.append(node_to_reassign)
            
            
def add_generators_fromcsv(Grid,Gen_csv):
    Gen_data = pd.read_csv(Gen_csv)
    Gen_data = Gen_data.set_index('Node')
    
    
    for index, row in Gen_data.iterrows():
        var_name = index
        MWmax = Gen_data.at[index, 'MWmax']
        MWmin = Gen_data.at[index, 'MWmin']
        MVArmin = Gen_data.at[index, 'MVArmin']
        MVArmax = Gen_data.at[index, 'MVArmax']
        if 'Price' in Gen_data.columns:
            price = Gen_data.at[index, 'Price']
        else:
            price=None
        if 'Market' in Gen_data.columns:
            Market = Gen_data.at[index, 'Market']
        else:
            Market=None
        
        

        add_gen(Grid,var_name,market=Market,price=price,MVAmax=MWmax,MWmin=MWmin,MVArmin=MVArmin,MVArmax=MVArmax)   
        s=1
def add_gen(Grid, node_name, market=None,price=None,MVAmax=99999,MWmin=0,MVArmin=None,MVArmax=None):
    if MVArmin is None:
        MVArmin=-MVAmax
    if MVArmax is None:
        MVArmax=MVAmax
    if market  is None :
        if price  is None:
            price=1
    else:
        assign_nodeToMarket(Grid, node_name, market)    
    for node in Grid.nodes_AC:
        if node_name == node.name:
            
            node.extGrid = 2
            if price  is not None:
                node.price=price
            else:
                for M in Grid.Markets:
                    if market == M.name:
                            node.price = M.price
            node.Max_pow_gen=MVAmax/Grid.S_base
            node.Min_pow_gen=MWmin/Grid.S_base
            node.Max_pow_genR=MVArmax/Grid.S_base
            node.Min_pow_genR=MVArmin/Grid.S_base
            # if node.type!= 'PV':
            node.PGi = 0
            node.QGi = 0
    
    
def add_extGrid(Grid, node_name, market=None,price=None,MVA=99999,MVArmin=None,MVArmax=None,Allow_sell=False):
    
    
    
    if MVArmin is None:
        MVArmin=-MVA
    if MVArmax is None:
        MVArmax=MVA
    if market  is None:
        if price  is None:
            price=1
    else:
        assign_nodeToMarket(Grid, node_name, market)
    for node in Grid.nodes_AC:
        if node_name == node.name:
            node.extGrid = 2
            if price  is not None:
                node.price=price
            else:
                for M in Grid.Markets:
                    if market == M.name:
                            node.price = M.price
            node.Max_pow_gen=MVA/Grid.S_base
            node.Max_pow_genR=MVArmax/Grid.S_base
            node.Min_pow_genR=MVArmin/Grid.S_base
            if Allow_sell== True:
                node.Min_pow_gen=-MVA/Grid.S_base
            # if node.type!= 'PV':
            node.PGi = 0
            node.QGi = 0
            
def add_extReact(Grid, node_name, market=0,MVArmin=-99999,MVArmax=99999):
    for node in Grid.nodes_AC:
        if node_name == node.name:
            node.extGrid = 1
            node.Max_pow_genR=MVArmax/Grid.S_base
            node.Min_pow_genR=MVArmin/Grid.S_base
            # if node.type!= 'PV':
            node.QGi=0
            
def add_WPP(Grid, node_name,base, available=1,market=None,price=None,Offshore=False):
    
    if market  is None:
        if price  is None:
            price=1
    else:
        if Offshore==True:
            omarket= f'o{market}'
            assign_nodeToMarket(Grid, node_name, omarket)
            for M in Grid.Markets:
                if market == M.name:
                    M.ImportExpand+=base/Grid.S_base
        else:       
            assign_nodeToMarket(Grid, node_name, market)

    s=1
    for node in Grid.nodes_AC:
        if node_name == node.name:
            
            node.RenSource = True
            node.set_renewable_source(base,Grid.S_base)
            node.PGi_available=available
            if price  is not None:
                node.price=price
            else:
                for M in Grid.Markets:
                    if market == M.name:
                            node.price = M.price
 

def OPF_AC(grid,ObjRule=weights_def,PV_set=False,OnlyGen=True,sigma=1.05):
    
    model= OPF_createModel_AC(grid,PV_set)
      
    """
    
    """
    ObjRule['DC_losses']['w']=0 
    ObjRule['Converter_Losses']['w']=0
        
    kap=0
    [obj_rule,model]= OPF_obj(model,grid,ObjRule,kap,OnlyGen,sigma)
    """
    """

    if grid.Converters_ACDC is not None:
        if any(conv.OPF_fx for conv in grid.Converters_ACDC):
                fx_conv(model, grid)
                    
                
    """
    """
    [model,results]= OPF_solve(model)

    ExportAC_model_toPyflowACDC(grid,model)   
    grid.OPF_run=True  
    return model, results
                

def OPF_ACDC(grid,ObjRule=weights_def,kap=1e-3,PV_set=False,OnlyGen=True,sigma=1.05,Markets=False):
    if OnlyGen == False:
        grid.OnlyGen=False
    if  ObjRule['Social_Cost']['w']!=0 :
        Markets=True
    if  ObjRule['Curtailment_Red']['w']!=0 :
        grid.CurtCost=True
    model= OPF_createModel_ACDC(grid,PV_set,Markets)
      
    """
    """
    
    
    
    [obj_rule,model]= OPF_obj(model,grid,ObjRule,kap,OnlyGen,sigma)
    """
    """

    
    if any(conv.OPF_fx for conv in grid.Converters_ACDC):
            fx_conv(model, grid)
                
                
    """
    """
    [model,results]= OPF_solve(model)

    ExportACDC_model_toPyflowACDC(grid,model,Markets)   
    grid.OPF_run=True  
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


def OPF_solve(model):
    
    
    opt = pyo.SolverFactory("ipopt")
    results = opt.solve(model)
    
    return model , results

def OPF_updateParam(grid,model):
 
    for n in grid.nodes_AC:
        model.P_Gain_known_AC[n.nodeNumber] = n.PGi
        model.P_renSource[n.nodeNumber] = n.PGi_ren
        model.P_Load_known_AC[n.nodeNumber] = n.PLi
        model.Q_known_AC[n.nodeNumber] = n.QGi-n.QLi
        model.price[n.nodeNumber] = n.price
        
    for n in grid.nodes_DC:
        model.P_known_DC[n.nodeNumber] = n.P_DC
    

    return model

def OPF_obj(model,grid,ObjRule,kap,OnlyGen,sigma):
    
    if sigma != grid.sigma: 
        grid.sigma=sigma         
    
    
    obj_expr=0
    
    # for node in  model.nodes_AC:
    #     nAC=grid.nodes_AC[node]
    #     if nAC.Num_conv_connected >= 2:
    #         obj_expr += sum(model.Q_conv_s_AC[conv]**2 for conv in nAC.connected_conv)

    def formula_Min_Ext_Gen():
        if ObjRule['Ext_Gen']['w']==0:
            return 0
        return sum((model.P_Gen[node]**2 + (model.Q_Gen[node]**2)) for node in model.nodes_AC)

    def formula_Energy_cost():
        if ObjRule['Energy_cost']['w']==0:
            return 0
        elif OnlyGen == True:
            nodes_with_Gen = [node for node in model.nodes_AC if grid.nodes_AC[node].extGrid == 2]
            return sum((model.P_Gen[node]* model.price[node]) for node in nodes_with_Gen)*grid.S_base
        else :
            nodes_with_RenSource = [node for node in model.nodes_AC if grid.nodes_AC[node].RenSource == True]
            nodes_with_Gen = [node for node in model.nodes_AC if grid.nodes_AC[node].extGrid == 2]
            nodes_with_conv= [node for node in model.nodes_AC if grid.nodes_AC[node].Num_conv_connected != 0]
            return sum(model.P_Gen[node]* model.price[node] for node in nodes_with_Gen)*grid.S_base  \
                    + sum(model.P_renSource[node]*model.price[node]*model.curtail[node] for node in nodes_with_RenSource)*grid.S_base \
                    + sum(model.P_conv_AC[node]*model.price[node] for node in nodes_with_conv)*grid.S_base
    def formula_AC_losses():
        if ObjRule['AC_losses']['w']==0:
            return 0
        return sum(model.PAC_line_loss[line] for line in model.lines_AC)

    def formula_DC_losses():
        if ObjRule['DC_losses']['w']==0:
            return 0
        return sum(model.PDC_line_loss[line] for line in model.lines_DC)

    def formula_Converter_Losses():
        if ObjRule['Converter_Losses']['w']==0:
            return 0
        return sum(model.P_conv_loss[conv] for conv in model.conv)
               
    def formula_curtailment_red():
        if ObjRule['Curtailment_Red']['w']==0:
            return 0
        nodes_with_RenSource = [node for node in model.nodes_AC if grid.nodes_AC[node].RenSource == True]
        return sum((1-model.curtail[node])*model.P_renSource[node]*model.price[node] for node in nodes_with_RenSource)*grid.S_base*grid.sigma
    
    def formula_curtailment_SC():
       if ObjRule['Social_Cost']['w']==0:
           return 0
       return sum(model.SocialCost[market] for market in model.M)
   
    for key, entry in ObjRule.items():
        if key == 'Ext_Gen':
            entry['f'] = formula_Min_Ext_Gen()
        elif key == 'Energy_cost':
            entry['f'] = formula_Energy_cost()
        elif key == 'AC_losses':
            entry['f'] = formula_AC_losses()
        elif key == 'DC_losses':
            entry['f'] = formula_DC_losses()
        elif key == 'Converter_Losses':
            entry['f'] = formula_Converter_Losses()
        elif key == 'Curtailment_Red':   
            entry ['f'] = formula_curtailment_red()
        elif key == 'Social_Cost':
            entry['f']  =formula_curtailment_SC()
    s=1
    total_weight = sum(entry['w'] for entry in ObjRule.values())
    if total_weight== 0:
        weighted_sum=0
    else:
        weighted_sum = sum(entry['w'] / total_weight * entry['f'] for entry in ObjRule.values())+ kap*obj_expr
    
    model.obj = pyo.Objective(rule=weighted_sum, sense=pyo.minimize)
    
    return weighted_sum ,model

def OPF_createModel_ACDC(grid,PV_set,Markets):
    
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
    P_Gain_known_AC = {}
    P_renSource ={}
    P_Load_known_AC = {}
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
    NumConvP_i={}
    u_min_ac=list(range(0, grid.nn_AC))
    u_min_dc=list(range(0, grid.nn_DC))
    u_max_ac=list(range(0, grid.nn_AC))
    u_max_dc=list(range(0, grid.nn_DC))
    u_c_min=list(range(0, grid.nconv))
    u_c_max=list(range(0, grid.nconv))
    
    market2node={}
    
    market_prices={}
    market_as={}
    market_bs={}
    PGL_min={}
    PGL_max={}
    PL_market={}
    
    P_conv_loss = {}

    AC_nodes_connected_conv = []
    DC_nodes_connected_conv = []

    AC_nodes_extGrid = []
    AC_nodes_extGridR = []
    curtail_nodes =[]
    AC_slack = []
    AC_PV = []

    DC_slack = []
    
    nn_M=0
    node2market={}
    
    if Markets == True:
        for m in grid.Markets:
            market2node[m.market_num]=[]
            nn_M+=1
            market_prices[m.market_num]=m.price
            market_as[m.market_num]=m.a
            market_bs[m.market_num]=m.b
            PGLmin=m.PGL_min
            PGLmax=m.PGL_max
            import_M= m.import_pu_L
            export_M= m.export_pu_G*(sum (node.PGi_ren+node.Max_pow_gen for node in m.nodes_AC))
            PL_market[m.market_num]=0
            for n in m.nodes_AC:
                market2node[m.market_num].append(n.nodeNumber)
                node2market[n.nodeNumber]=m.market_num
                PL_market[m.market_num]+=n.PLi
            PGL_min[m.market_num]=max(PGLmin,-import_M*PL_market[m.market_num])
            PGL_max[m.market_num]=min(PGLmax,export_M)
        lista_M= list(range(0,nn_M))  
    
    
    for n in grid.nodes_AC:
        if n.type == 'Slack':
            AC_slack.append(n.nodeNumber)
        if n.type == 'PV':
            AC_PV.append(n.nodeNumber)
        if n.extGrid == 2:
            AC_nodes_extGrid.append(n.nodeNumber)
        elif n.extGrid == 1:
            AC_nodes_extGridR.append(n.nodeNumber)
        if n.RenSource == True :
            curtail_nodes.append(n.nodeNumber)
    for n in grid.nodes_DC:
        if n.type == 'Slack':
            DC_slack.append(n.nodeNumber)
    s = 1



    for n in grid.nodes_AC:
        V_ini_AC[n.nodeNumber] = n.V_ini
        Theta_ini[n.nodeNumber] = n.theta_ini
        P_Gain_known_AC[n.nodeNumber] = n.PGi
        P_renSource[n.nodeNumber] = n.PGi_ren
        P_Load_known_AC[n.nodeNumber] = n.PLi
        Q_know[n.nodeNumber] = n.QGi-n.QLi
        u_min_ac[n.nodeNumber] = n.Umin
        u_max_ac[n.nodeNumber] = n.Umax
        P_conv_AC[n.nodeNumber] = 0
        Q_conv_AC[n.nodeNumber] = 0
       
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
        
        S_limit_conv[conv.ConvNumber] = conv.MVA_max/grid.S_base

        P_conv_DC[conv.Node_DC.nodeNumber] = conv.P_DC
        P_conv_limit[conv.Node_DC.nodeNumber] = conv.MVA_max/grid.S_base
        
        NumConvP_i [conv.ConvNumber] =conv.NumConvP
        
        u_c_min[conv.ConvNumber] = conv.Ucmin
        u_c_max[conv.ConvNumber] = conv.Ucmax
        

        P_conv_loss[conv.ConvNumber] = conv.P_loss

    AC_nodes_no_extGrid = [node for node in lista_nodos_AC if node not in AC_nodes_extGrid]


    """
    MODEL INITIATION
    """



    model = pyo.ConcreteModel()
    
    model.name="AC/DC hybrid OPF"
    
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
    #Market Variables
    def Market_P_bounds(model, market):
        nM = grid.Markets[market]
        return (nM.PGL_min,nM.PGL_max)
    
    
    
    if  Markets == True:
        model.M = pyo.Set(initialize=lista_M)
        model.Pm = pyo.Var(model.M,bounds=Market_P_bounds)
        model.Pm_load = pyo.Var(model.M)
        model.price = pyo.Var(model.nodes_AC,initialize=price)
        model.market_price = pyo.Var(model.M)
        model.market_a = pyo.Param(model.M,initialize=market_as,mutable=True)
        model.market_b = pyo.Param(model.M,initialize=market_bs,mutable=True)
        model.SocialCost = pyo.Var(model.M)
        
    else:
        model.price      = pyo.Param(model.nodes_AC, initialize=price,mutable=True)
    ### AC Variables
    #AC nodes variables
    model.V_AC       = pyo.Var(model.nodes_AC, bounds=lambda model, node: (u_min_ac[node], u_max_ac[node]), initialize=V_ini_AC)
    model.thetha_AC  = pyo.Var(model.nodes_AC, bounds=(-1.6, 1.6), initialize=Theta_ini)

    model.P_Gain_known_AC = pyo.Param(model.nodes_AC, initialize=P_Gain_known_AC,mutable=True)
    model.P_Load_known_AC = pyo.Param(model.nodes_AC, initialize=P_Load_known_AC,mutable=True)
    
    model.P_renSource = pyo.Param(model.nodes_AC, initialize=P_renSource, mutable=True)
    
    
    model.Q_known_AC = pyo.Param(model.nodes_AC, initialize=Q_know,mutable=True)
    
    
    def curtail_bounds(model, node):
        nAC = grid.nodes_AC[node]
        if nAC.nodeNumber in curtail_nodes:
            return (0,1)  
        else: 
            return (1,1)
    model.curtail = pyo.Var(model.nodes_AC, bounds=curtail_bounds, initialize=1)
    def P_Gen_bounds(model, node):
        nAC = grid.nodes_AC[node]
        return (nAC.Min_pow_gen,nAC.Max_pow_gen)
        
    def Q_Gen_bounds(model, node):
        nAC = grid.nodes_AC[node]
        return (nAC.Min_pow_genR,nAC.Max_pow_genR)
    
    def P_gen_ini(model,node):
        nAC=  grid.nodes_AC[node]
        ini=0
        if nAC.Min_pow_gen>0:
            ini=nAC.Min_pow_gen
        return (ini)
    
   
    model.P_Gen = pyo.Var(model.nodes_AC, bounds=P_Gen_bounds, initialize=P_gen_ini)
    model.Q_Gen = pyo.Var(model.nodes_AC, bounds=Q_Gen_bounds, initialize=0)
    
    s=1
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
    model.P_known_DC = pyo.Param(model.nodes_DC, initialize=P_known_DC,mutable=True)
    
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
    
    model.NumConvP = pyo.Param(model.conv, initialize=NumConvP_i)
    
    # def P_conv_bounds(model, node):
    #     return (-P_conv_limit[node], P_conv_limit[node])
    
   
   
    model.P_conv_loss = pyo.Var(model.conv, initialize=P_conv_loss)
    
    model.P_conv_DC = pyo.Var(model.nodes_DC, initialize=0)
    
    model.P_conv_AC = pyo.Var(model.nodes_AC, initialize=0)
    model.Q_conv_AC = pyo.Var(model.nodes_AC, initialize=0)
    
    model.P_conv_s_AC  = pyo.Var(model.conv, initialize=0)   
    model.Q_conv_s_AC = pyo.Var(model.conv, initialize=0)

    model.P_conv_c_AC  = pyo.Var(model.conv, initialize=0.0001)   
    model.Q_conv_c_AC = pyo.Var(model.conv, initialize=0.0001)
    
    model.P_conv_c_AC_sq = pyo.Var(model.conv, bounds=(1e-100,None), initialize=0.1)   
    model.Q_conv_c_AC_sq = pyo.Var(model.conv, bounds=(1e-100,None), initialize=0.1)
    
    """EQUALITY CONSTRAINTS
    """
    #Market equality constraints
    
    def market_price_formula(model,market):
        return model.market_price[market]==2*model.market_a[market]*model.Pm[market]*grid.S_base+model.market_b[market]
        
    def node_price_set(model,node):
        try: 
            market=node2market[node]
            return model.market_price[market]== model.price[node] 
        except:
            return model.price[node]==0
        
    
    def P_market(model,market):
        Pm=sum(model.P_Gain_known_AC[node]+P_renSource[node]*model.curtail[node]-model.P_Load_known_AC[node]  + model.P_Gen[node] for node in market2node[market])
        return model.Pm[market] ==Pm
    def P_market_load(model,market):
        Pl=sum(model.P_Load_known_AC[node] for node in market2node[market])
        return model.Pm_load[market] ==Pl
    
    def Social_cost(model,market):
        return model.SocialCost[market]== model.market_a[market]*(model.Pm[market]*grid.S_base)**2+model.market_b[market]*(model.Pm[market]*grid.S_base)
    if Markets == True:
        model.market_price_constraint = pyo.Constraint(model.M,rule=market_price_formula)
        model.price_constraint = pyo.Constraint(model.nodes_AC,rule=node_price_set)
        
        model.Pm_constraint = pyo.Constraint(model.M,rule=P_market)
        model.PmL_constraint = pyo.Constraint(model.M,rule=P_market_load)
        model.SC_constraint = pyo.Constraint(model.M,rule=Social_cost)
    
    ### AC constraints
    # AC node constraints
    def P_AC_node_rule(model, node):
        P_sum = sum(
                model.V_AC[node] * model.V_AC[k] *
                (np.real(grid.Ybus_AC[node, k]) * pyo.cos(model.thetha_AC[node] - model.thetha_AC[k]) +
                 np.imag(grid.Ybus_AC[node, k]) * pyo.sin(model.thetha_AC[node] - model.thetha_AC[k]))
                for k in model.nodes_AC if grid.Ybus_AC[node, k] != 0   )   

        return P_sum == model.P_Gain_known_AC[node]+model.P_renSource[node]*model.curtail[node]-model.P_Load_known_AC[node] + model.P_conv_AC[node] + model.P_Gen[node]

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
       
        l = grid.lines_AC[line]
        f = l.fromNode.nodeNumber
        t = l.toNode.nodeNumber
        Vf=model.V_AC[f]
        Vt=model.V_AC[t]
        Gtt=np.real(l.Ybus_branch[1,1])
        Gtf=np.real(l.Ybus_branch[1,0])
        Btf=np.imag(l.Ybus_branch[1,0])
        thf=model.thetha_AC[f]
        tht=model.thetha_AC[t]
        
        Pto= Vt*Vt*Gtt + Vf*Vt*(Gtf*pyo.cos(tht - thf) + Btf*pyo.sin(tht - thf))
       
        
        return model.PAC_to[line] == Pto
    
    def P_from_AC_line(model,line):       
       l = grid.lines_AC[line]
       f = l.fromNode.nodeNumber
       t = l.toNode.nodeNumber
       Vf=model.V_AC[f]
       Vt=model.V_AC[t]
       Gff=np.real(l.Ybus_branch[0,0])
       Gft=np.real(l.Ybus_branch[0,1])
       Bft=np.imag(l.Ybus_branch[0,1])
       thf=model.thetha_AC[f]
       tht=model.thetha_AC[t]
       
       Pfrom= Vf*Vf*Gff + Vf*Vt*(Gft*pyo.cos(thf - tht) + Bft*pyo.sin(thf - tht))

       return model.PAC_from[line] == Pfrom
    
    def Q_to_AC_line(model,line):   
        l = grid.lines_AC[line]
        f = l.fromNode.nodeNumber
        t = l.toNode.nodeNumber
        Vf=model.V_AC[f]
        Vt=model.V_AC[t]
       
        thf=model.thetha_AC[f]
        tht=model.thetha_AC[t]
        
        Btt=np.imag(l.Ybus_branch[1,1])
        Gtf=np.real(l.Ybus_branch[1,0])
        Btf=np.imag(l.Ybus_branch[1,0])
        
        Qto   = -Vt*Vt*Btt + Vf*Vt*(Gtf*pyo.sin(tht - thf) - Btf*pyo.cos(tht - thf))
         
        
        return model.QAC_to[line] == Qto
    
    def Q_from_AC_line(model,line):       
       l = grid.lines_AC[line]
       f = l.fromNode.nodeNumber
       t = l.toNode.nodeNumber
       Vf=model.V_AC[f]
       Vt=model.V_AC[t]
      
       Bff=np.imag(l.Ybus_branch[0,0])
       Gft=np.real(l.Ybus_branch[0,1])
       Bft=np.imag(l.Ybus_branch[0,1])
       thf=model.thetha_AC[f]
       tht=model.thetha_AC[t]
       

       Qfrom = -Vf*Vf*Bff + Vf*Vt*(Gft*pyo.sin(thf - tht) - Bft*pyo.cos(thf - tht))
      

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
                    P_sum += pol*model.V_DC[i] * (model.V_DC[i]-model.V_DC[k])*Y

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
        
       Gc  = element.Gc   * model.NumConvP[conv]
       Bc  = element.Bc   * model.NumConvP[conv]
       Gtf = element.Gtf  * model.NumConvP[conv]
       Btf = element.Btf  * model.NumConvP[conv]
       Bf  = element.Bf   * model.NumConvP[conv]
       
       if element.Bf == 0:
           Ztf = element.Ztf
           Zc = element.Zc
           Zeq = Ztf+Zc
           Yeq = 1/Zeq

           Gc = np.real(Yeq)  * model.NumConvP[conv]
           Bc = np.imag(Yeq)  * model.NumConvP[conv]
           
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
       
       Gc = element.Gc    * model.NumConvP[conv]
       Bc = element.Bc    * model.NumConvP[conv]
       Gtf = element.Gtf  * model.NumConvP[conv]
       Btf = element.Btf  * model.NumConvP[conv]
       Bf = element.Bf    * model.NumConvP[conv]
       
       if element.Bf == 0:
           Ztf = element.Ztf
           Zc = element.Zc
           Zeq = Ztf+Zc
           Yeq = 1/Zeq

           Gc = np.real(Yeq)  * model.NumConvP[conv]
           Bc = np.imag(Yeq)  * model.NumConvP[conv]
           
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
       
       Gc = element.Gc    * model.NumConvP[conv]
       Bc = element.Bc    * model.NumConvP[conv]
       Gtf = element.Gtf  * model.NumConvP[conv]
       Btf = element.Btf  * model.NumConvP[conv]
       Bf = element.Bf    * model.NumConvP[conv]
       
       if element.Bf == 0:
           Ztf = element.Ztf
           Zc = element.Zc
           Zeq = Ztf+Zc
           Yeq = 1/Zeq

           Gc = np.real(Yeq)  * model.NumConvP[conv]
           Bc = np.imag(Yeq)  * model.NumConvP[conv]
           
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
        
       Gc = element.Gc    * model.NumConvP[conv]
       Bc = element.Bc    * model.NumConvP[conv]
       Gtf = element.Gtf  * model.NumConvP[conv]
       Btf = element.Btf  * model.NumConvP[conv]
       Bf = element.Bf    * model.NumConvP[conv]
       
       if element.Bf == 0:
           Ztf = element.Ztf
           Zc = element.Zc
           Zeq = Ztf+Zc
           Yeq = 1/Zeq

           Gc = np.real(Yeq)  * model.NumConvP[conv]
           Bc = np.imag(Yeq)  * model.NumConvP[conv]
           
           
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

            
       if element.Bf == 0 or element.Gtf == 0:
        return pyo.Constraint.Skip
           
       else:
           Gc = element.Gc    * model.NumConvP[conv]
           Bc = element.Bc    * model.NumConvP[conv]
           Gtf = element.Gtf  * model.NumConvP[conv]
           Btf = element.Btf  * model.NumConvP[conv]
           Bf = element.Bf    * model.NumConvP[conv]
                
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
           
           Gc = element.Gc    * model.NumConvP[conv]
           Bc = element.Bc    * model.NumConvP[conv]
           Gtf = element.Gtf  * model.NumConvP[conv]
           Btf = element.Btf  * model.NumConvP[conv]
           Bf = element.Bf    * model.NumConvP[conv]

         
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
       P_conv = sum(model.P_conv_s_AC[conv] for conv in nAC.connected_conv)
                  
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

    # def Psqr(model,conv):
    #     return model.P_conv_c_AC_sq[conv]== model.P_conv_c_AC[conv]**2

    # def Qsqr(model,conv):
    #     return model.Q_conv_c_AC_sq[conv]== model.Q_conv_c_AC[conv]**2
    
    # model.Conv_PCsq_constraint = pyo.Constraint(model.conv, rule=Psqr)
    # model.Conv_QCsq_constraint = pyo.Constraint(model.conv, rule=Qsqr)
    


    def Conv_loss_rule(model, conv):
        element=grid.Converters_ACDC[conv]
        nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
        nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
        a = grid.Converters_ACDC[conv].a_conv * model.NumConvP[conv]
        b = grid.Converters_ACDC[conv].b_conv
      

        # current = pyo.sqrt(model.P_conv_c_AC_sq[conv]+model.Q_conv_c_AC_sq[conv])/(model.Uc[conv])
        currentsqr = (model.P_conv_c_AC[conv]**2+model.Q_conv_c_AC[conv]**2)/(model.Uc[conv]**2)

        

        # c_inver = (element.c_inver_og /model.NumConvP[conv])*element.basekA**2/grid.S_base
        # c_rect = (element.c_rect_og   /model.NumConvP[conv])*element.basekA**2/grid.S_base
        
        
        c_inver=grid.Converters_ACDC[conv].c_inver /model.NumConvP[conv]
        c_rect=grid.Converters_ACDC[conv].c_rect   /model.NumConvP[conv]
    
       
        P_loss = a  +c_rect * currentsqr
    
        
        return model.P_conv_loss[conv] == P_loss

    
    model.Conv_DC_constraint = pyo.Constraint(model.conv, rule=Conv_DC_rule)
    model.Conv_PAC_constraint = pyo.Constraint(model.nodes_AC, rule=Conv_PAC_rule)
    model.Conv_QAC_constraint = pyo.Constraint(model.nodes_AC, rule=Conv_Q_rule)
    model.Conv_loss_constraint = pyo.Constraint(model.conv, rule=Conv_loss_rule)
    s=1
    
    """
    INEQUALITY CONSTRAINTS
    """
    # Market inequality constraints
    
    def import_rule(model,market):
        return model.Pm[market] >= PGL_min[market]
    def export_rule(model,market):
        return model.Pm[market] <= PGL_max[market]
    
    
    if Markets ==True:
        model.import_constraint = pyo.Constraint(model.M,rule=import_rule)
        model.export_constraint = pyo.Constraint(model.M,rule=export_rule)
    
    #AC lines inequality
    def S_gen_AC_limit_rule(model,node):
        nAC = grid.nodes_AC[node]
        return model.P_Gen[node]**2+model.Q_Gen[node]**2 <= nAC.Max_pow_gen**2 
    
    model.S_gen_AC_limit_constraint   = pyo.Constraint(model.nodes_AC, rule=S_gen_AC_limit_rule)
   
    
    def S_to_AC_limit_rule(model,line):
        
        return model.PAC_to[line]**2+model.QAC_to[line]**2 <= S_lineAC_limit[line]**2
    def S_from_AC_limit_rule(model,line):
        
        return model.PAC_from[line]**2+model.QAC_from[line]**2 <= S_lineAC_limit[line]**2
    
    
    model.S_to_AC_limit_constraint   = pyo.Constraint(model.lines_AC, rule=S_to_AC_limit_rule)
    model.S_from_AC_limit_constraint = pyo.Constraint(model.lines_AC, rule=S_from_AC_limit_rule)
    
    
    #DC lines inequality
    
    #they set in the variables themselves
    
    #Converters inequality 
    
    def Conv_ACc_Limit_rule(model, conv):
        nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
        element= grid.Converters_ACDC[conv]
        return (model.P_conv_c_AC[conv]**2+model.Q_conv_c_AC[conv]**2) <= (S_limit_conv[conv]*model.NumConvP[conv])**2*0.99 
    
    
    def Conv_ACs_Limit_rule(model, conv):
        nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
        element= grid.Converters_ACDC[conv]
        return (model.P_conv_s_AC[conv]**2+model.Q_conv_s_AC[conv]**2) <= (S_limit_conv[conv]*model.NumConvP[conv])**2*0.99 
    
    def Conv_DC_Limit_rule(model, conv):
        element= grid.Converters_ACDC[conv]
        nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
        return (model.P_conv_DC[nDC]**2) <= (P_conv_limit[nDC]*model.NumConvP[conv])**2*0.99
    
    model.Conv_ACc_Limit_constraint = pyo.Constraint(model.conv, rule=Conv_ACc_Limit_rule)
    model.Conv_ACs_Limit_constraint = pyo.Constraint(model.conv, rule=Conv_ACs_Limit_rule)
    model.Conv_DC_Limit_constraint = pyo.Constraint(model.conv, rule=Conv_DC_Limit_rule)
    
    return model


def OPF_createModel_AC(grid,PV_set):
    
    """Translation of element wise to internal numbering
    """

    lista_nodos_AC = list(range(0, grid.nn_AC))
    lista_lineas_AC = list(range(0,grid.nl_AC))
    
    price = {}
    V_ini_AC = {}
    Theta_ini = {}
    P_Gain_known_AC = {}
    P_Load_known_AC = {}
    Q_know = {}

    P_renSource={}    

    S_lineAC_limit ={}

    S_limit = {}

    
    P_conv_limit = {}
    u_min_ac=list(range(0, grid.nn_AC))
    u_max_ac=list(range(0, grid.nn_AC))
  
    

    P_conv_loss = {}

    AC_nodes_connected_conv = []
    DC_nodes_connected_conv = []

    AC_nodes_extGrid = []
    AC_nodes_extGridR = []
    curtail_nodes =[]
    AC_slack = []
    AC_PV = []


    for n in grid.nodes_AC:
        if n.type == 'Slack':
            AC_slack.append(n.nodeNumber)
        if n.type == 'PV':
            AC_PV.append(n.nodeNumber)
        if n.extGrid == 2:
            AC_nodes_extGrid.append(n.nodeNumber)
        elif n.extGrid == 1:
            AC_nodes_extGridR.append(n.nodeNumber)
        
        if n.RenSource ==True :
            curtail_nodes.append(n.nodeNumber)
   


    for n in grid.nodes_AC:
        V_ini_AC[n.nodeNumber] = n.V_ini
        Theta_ini[n.nodeNumber] = n.theta_ini
        P_Gain_known_AC[n.nodeNumber] = n.PGi
        P_renSource[n.nodeNumber] = n.PGi_ren
        P_Load_known_AC[n.nodeNumber] = n.PLi
        Q_know[n.nodeNumber] = n.QGi-n.QLi
        u_min_ac[n.nodeNumber] = n.Umin
        u_max_ac[n.nodeNumber] = n.Umax

        price[n.nodeNumber] = n.price
    
    max_price=max(price)
    for l in grid.lines_AC:
        S_lineAC_limit[l.lineNumber]= l.MVA_rating/grid.S_base
    
  

    AC_nodes_no_extGrid = [node for node in lista_nodos_AC if node not in AC_nodes_extGrid]


    """
    MODEL INITIATION
    """



    model = pyo.ConcreteModel()
    
    model.name="AC OPF"
    
    "Model Sets"
    model.nodes_AC   = pyo.Set(initialize=lista_nodos_AC)
    model.lines_AC   = pyo.Set(initialize=lista_lineas_AC)
    model.AC_no_extG = pyo.Set(initialize=AC_nodes_no_extGrid)
    model.AC_slacks  = pyo.Set(initialize=AC_slack)
    model.AC_PVs     = pyo.Set(initialize=AC_PV)

  
     
    """Variables and limits
    """
    ### AC Variables
    #AC nodes variables
    model.V_AC       = pyo.Var(model.nodes_AC, bounds=lambda model, node: (u_min_ac[node], u_max_ac[node]), initialize=V_ini_AC)
    model.thetha_AC  = pyo.Var(model.nodes_AC, bounds=(-3.1416,3.1416), initialize=Theta_ini)

    model.P_Gain_known_AC = pyo.Param(model.nodes_AC, initialize=P_Gain_known_AC,mutable=True)
    model.P_Load_known_AC = pyo.Param(model.nodes_AC, initialize=P_Load_known_AC,mutable=True)
    
    model.P_renSource = pyo.Param(model.nodes_AC, initialize=P_renSource, mutuable=True)
    
    
    model.Q_known_AC = pyo.Param(model.nodes_AC, initialize=Q_know,mutable=True)
    model.price      = pyo.Param(model.nodes_AC, initialize=price,mutable=True)
    
    model.max_price = pyo.Param(initialize=max_price)

    

    
    def curtail_bounds(model, node):
        nAC = grid.nodes_AC[node]
        if nAC.nodeNumber in curtail_nodes:
            return (0,1)  
        else: 
            return (1,1)
    model.curtail = pyo.Var(model.nodes_AC, bounds=curtail_bounds, initialize=1)
    def P_Gen_bounds(model, node):
        nAC = grid.nodes_AC[node]
        return (nAC.Min_pow_gen,nAC.Max_pow_gen)
        
    def Q_Gen_bounds(model, node):
        nAC = grid.nodes_AC[node]
        return (nAC.Min_pow_genR,nAC.Max_pow_genR)
    
    def P_gen_ini(model,node):
        nAC=  grid.nodes_AC[node]
        ini=0
        if nAC.Min_pow_gen>0:
            ini=nAC.Min_pow_gen
        return (ini)
    
   
    model.P_Gen = pyo.Var(model.nodes_AC, bounds=P_Gen_bounds, initialize=P_gen_ini)
    model.Q_Gen = pyo.Var(model.nodes_AC, bounds=Q_Gen_bounds, initialize=0)
    
    s=1
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

        return P_sum == model.P_Gain_known_AC[node]+P_renSource[node]*model.curtail[node]-model.P_Load_known_AC[node]  + model.P_Gen[node]

    def Q_AC_node_rule(model, node):

        Q_sum = sum(
            model.V_AC[node] * model.V_AC[k] *
            (np.real(grid.Ybus_AC[node, k]) * pyo.sin(model.thetha_AC[node] - model.thetha_AC[k]) -
             np.imag(grid.Ybus_AC[node, k]) * pyo.cos(model.thetha_AC[node] - model.thetha_AC[k]))
            for k in model.nodes_AC if grid.Ybus_AC[node, k] != 0)

        return Q_sum == model.Q_known_AC[node]  + model.Q_Gen[node]

    model.P_AC_node_constraint = pyo.Constraint(model.nodes_AC, rule=P_AC_node_rule)
    model.Q_AC_node_constraint = pyo.Constraint(model.nodes_AC, rule=Q_AC_node_rule)
    
    # AC line equality constraints
    
    def P_to_AC_line(model,line):   
       
        l = grid.lines_AC[line]
        f = l.fromNode.nodeNumber
        t = l.toNode.nodeNumber
        Vf=model.V_AC[f]
        Vt=model.V_AC[t]
        Gtt=np.real(l.Ybus_branch[1,1])
        Gtf=np.real(l.Ybus_branch[1,0])
        Btf=np.imag(l.Ybus_branch[1,0])
        thf=model.thetha_AC[f]
        tht=model.thetha_AC[t]
        
        Pto= Vt*Vt*Gtt + Vf*Vt*(Gtf*pyo.cos(tht - thf) + Btf*pyo.sin(tht - thf))
       
        
        return model.PAC_to[line] == Pto
    
    def P_from_AC_line(model,line):       
       l = grid.lines_AC[line]
       f = l.fromNode.nodeNumber
       t = l.toNode.nodeNumber
       Vf=model.V_AC[f]
       Vt=model.V_AC[t]
       Gff=np.real(l.Ybus_branch[0,0])
       Gft=np.real(l.Ybus_branch[0,1])
       Bft=np.imag(l.Ybus_branch[0,1])
       thf=model.thetha_AC[f]
       tht=model.thetha_AC[t]
       
       Pfrom= Vf*Vf*Gff + Vf*Vt*(Gft*pyo.cos(thf - tht) + Bft*pyo.sin(thf - tht))

       return model.PAC_from[line] == Pfrom
    
    def Q_to_AC_line(model,line):   
        l = grid.lines_AC[line]
        f = l.fromNode.nodeNumber
        t = l.toNode.nodeNumber
        Vf=model.V_AC[f]
        Vt=model.V_AC[t]
       
        thf=model.thetha_AC[f]
        tht=model.thetha_AC[t]
        
        Btt=np.imag(l.Ybus_branch[1,1])
        Gtf=np.real(l.Ybus_branch[1,0])
        Btf=np.imag(l.Ybus_branch[1,0])
        
        Qto   = -Vt*Vt*Btt + Vf*Vt*(Gtf*pyo.sin(tht - thf) - Btf*pyo.cos(tht - thf))
         
        
        return model.QAC_to[line] == Qto
    
    def Q_from_AC_line(model,line):       
       l = grid.lines_AC[line]
       f = l.fromNode.nodeNumber
       t = l.toNode.nodeNumber
       Vf=model.V_AC[f]
       Vt=model.V_AC[t]
      
       Bff=np.imag(l.Ybus_branch[0,0])
       Gft=np.real(l.Ybus_branch[0,1])
       Bft=np.imag(l.Ybus_branch[0,1])
       thf=model.thetha_AC[f]
       tht=model.thetha_AC[t]
       

       Qfrom = -Vf*Vf*Bff + Vf*Vt*(Gft*pyo.sin(thf - tht) - Bft*pyo.cos(thf - tht))
      

       return model.QAC_from[line] == Qfrom
    
    def P_loss_AC_rule(model,line):
        return model.PAC_line_loss[line]== model.PAC_to[line]+model.PAC_from[line]
    
    
    model.Pto_AC_line_constraint   = pyo.Constraint(model.lines_AC, rule=P_to_AC_line)
    model.Pfrom_AC_line_constraint = pyo.Constraint(model.lines_AC, rule=P_from_AC_line)
    model.Qto_AC_line_constraint   = pyo.Constraint(model.lines_AC, rule=Q_to_AC_line)
    model.Qfrom_AC_line_constraint = pyo.Constraint(model.lines_AC, rule=Q_from_AC_line)
    model.P_AC_loss_constraint     = pyo.Constraint(model.lines_AC, rule=P_loss_AC_rule)
    
    
    
   
    """
    INEQUALITY CONSTRAINTS
    """
    #AC lines inequality
    def S_gen_AC_limit_rule(model,node):
        nAC = grid.nodes_AC[node]
        return model.P_Gen[node]**2+model.Q_Gen[node]**2 <= nAC.Max_pow_gen**2 
    
    model.S_gen_AC_limit_constraint   = pyo.Constraint(model.nodes_AC, rule=S_gen_AC_limit_rule)
   
    
   
    
    def S_to_AC_limit_rule(model,line):
        
        return model.PAC_to[line]**2+model.QAC_to[line]**2 <= S_lineAC_limit[line]**2
    def S_from_AC_limit_rule(model,line):
        
        return model.PAC_from[line]**2+model.QAC_from[line]**2 <= S_lineAC_limit[line]**2
    
    
    model.S_to_AC_limit_constraint   = pyo.Constraint(model.lines_AC, rule=S_to_AC_limit_rule)
    model.S_from_AC_limit_constraint = pyo.Constraint(model.lines_AC, rule=S_from_AC_limit_rule)
    
    
   
    return model


def OPF_conv_results(grid,model,Dict=False):
    lista_conv = list(range(0, grid.nconv))
    
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
    opt_res_curtailment_dict ={}
    opt_res_React_extGrid_dict ={}
  
    opt_res_P_extGrid = np.zeros(grid.nn_AC)
    opt_res_Q_extGrid = np.zeros(grid.nn_AC)
    opt_res_curtailment = np.ones(grid.nn_AC)
    opt_res_React_extGrid = np.zeros(grid.nn_AC)
    
    for node in grid.nodes_AC:
        extGrid=node.nodeNumber
        name = grid.node_names_AC[extGrid]
        
        
        if node.extGrid == 2:
            opt_res_P_extGrid_dict [name] = pyo.value(model.P_Gen[extGrid])
            opt_res_Q_extGrid_dict [name] = pyo.value(model.Q_Gen[extGrid])
            
            opt_res_P_extGrid[extGrid] = pyo.value(model.P_Gen[extGrid])
            opt_res_Q_extGrid[extGrid] = pyo.value(model.Q_Gen[extGrid])
            
        elif node.extGrid ==1:
                opt_res_React_extGrid_dict [name] = pyo.value(model.Q_Gen[extGrid])
                opt_res_React_extGrid  [extGrid] = pyo.value(model.Q_Gen[extGrid])
        if node.RenSource ==True:
            opt_res_curtailment_dict [name] = pyo.value(model.curtail[extGrid])
            opt_res_curtailment[extGrid] = pyo.value(model.curtail[extGrid])

    
    s=1
    
    if Dict== True:
        return opt_res_P_conv_DC_dict, opt_res_P_conv_AC_dict ,opt_res_Q_conv_AC_dict, opt_res_P_extGrid_dict,opt_res_Q_extGrid_dict,opt_res_curtailment_dict,opt_res_React_extGrid_dict
    else:
        return opt_res_P_conv_DC,opt_res_P_conv_AC,opt_res_Q_conv_AC,opt_res_P_extGrid,opt_res_Q_extGrid,opt_res_curtailment,opt_res_React_extGrid


def ExportACDC_model_toPyflowACDC(grid,model,Markets):
    grid.V_AC =np.zeros(grid.nn_AC)
    grid.Theta_V_AC=np.zeros(grid.nn_AC)
    grid.V_DC=np.zeros(grid.nn_DC)
    
    if Markets== True:
        
        for m in grid.Markets:
            nM= m.market_num
            m.price=np.float64(pyo.value(model.market_price[nM]))
            
    for node in grid.nodes_AC:
        nAC= node.nodeNumber
        node.V    =np.float64(pyo.value(model.V_AC[nAC]))
        node.theta=np.float64(pyo.value(model.thetha_AC[nAC]))
        node.P_s  =np.float64(pyo.value(model.P_conv_AC[nAC]))
        node.Q_s  =np.float64(pyo.value(model.Q_conv_AC[nAC]))
        if node.extGrid == 1:
            node.QGi_opt=np.float64(pyo.value(model.Q_Gen[nAC]))
        elif node.extGrid == 2:
            node.PGi_opt=np.float64(pyo.value(model.P_Gen[nAC]))
            node.QGi_opt=np.float64(pyo.value(model.Q_Gen[nAC]))
        if node.RenSource ==True:
            node.curtailment= np.float64(pyo.value(model.curtail[nAC]))
        # node.P_INJ=node.PGi*node.curtailment+node.PGi_opt-node.PLi + node.P_s 
        # node.Q_INJ=node.QGi+node.QGi_opt-node.QLi + node.Q_s 
        # if Markets ==True:
        #     node.price = np.float64(pyo.value(model.price[nAC]))
        grid.V_AC[nAC]=node.V
        grid.Theta_V_AC[nAC]=node.theta
    
    Pf = np.zeros((grid.nn_AC, 1))
    Qf = np.zeros((grid.nn_AC, 1))
    
    for node in grid.nodes_AC:
        i = node.nodeNumber
        for k in range(grid.nn_AC):
            G = np.real(grid.Ybus_AC[i, k])
            B = np.imag(grid.Ybus_AC[i, k])
            Pf[i] += grid.V_AC[i]*grid.V_AC[k] * \
                (G*np.cos(grid.Theta_V_AC[i]-grid.Theta_V_AC[k]) +
                 B*np.sin(grid.Theta_V_AC[i]-grid.Theta_V_AC[k]))
            Qf[i] += grid.V_AC[i]*grid.V_AC[k] * \
                (G*np.sin(grid.Theta_V_AC[i]-grid.Theta_V_AC[k]) -
                 B*np.cos(grid.Theta_V_AC[i]-grid.Theta_V_AC[k]))
  
    for node in grid.nodes_AC:
        i = node.nodeNumber
        node.P_INJ = Pf[i].item()
        node.Q_INJ = Qf[i].item()
    
    
    
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
    
def ExportAC_model_toPyflowACDC(grid,model):
    grid.V_AC =np.zeros(grid.nn_AC)
    grid.Theta_V_AC=np.zeros(grid.nn_AC)
    grid.V_DC=np.zeros(grid.nn_DC)
    
            
    for node in grid.nodes_AC:
        nAC= node.nodeNumber
        node.V    =np.float64(pyo.value(model.V_AC[nAC]))
        node.theta=np.float64(pyo.value(model.thetha_AC[nAC]))
       
        if node.extGrid == 1:
            node.QGi_opt=np.float64(pyo.value(model.Q_Gen[nAC]))
        elif node.extGrid == 2:
            node.PGi_opt=np.float64(pyo.value(model.P_Gen[nAC]))
            node.QGi_opt=np.float64(pyo.value(model.Q_Gen[nAC]))
        if node.RenSource ==True:
            node.curtailment= np.float64(pyo.value(model.curtail[nAC]))
        # node.P_INJ=node.PGi*node.curtailment+node.PGi_opt-node.PLi + node.P_s 
        # node.Q_INJ=node.QGi+node.QGi_opt-node.QLi + node.Q_s 
        grid.V_AC[nAC]=node.V
        grid.Theta_V_AC[nAC]=node.theta
   
    Pf = np.zeros((grid.nn_AC, 1))
    Qf = np.zeros((grid.nn_AC, 1))
    
    for node in grid.nodes_AC:
        i = node.nodeNumber
        for k in range(grid.nn_AC):
            G = np.real(grid.Ybus_AC[i, k])
            B = np.imag(grid.Ybus_AC[i, k])
            Pf[i] += grid.V_AC[i]*grid.V_AC[k] * \
                (G*np.cos(grid.Theta_V_AC[i]-grid.Theta_V_AC[k]) +
                 B*np.sin(grid.Theta_V_AC[i]-grid.Theta_V_AC[k]))
            Qf[i] += grid.V_AC[i]*grid.V_AC[k] * \
                (G*np.sin(grid.Theta_V_AC[i]-grid.Theta_V_AC[k]) -
                 B*np.cos(grid.Theta_V_AC[i]-grid.Theta_V_AC[k]))
  
    for node in grid.nodes_AC:
        i = node.nodeNumber
        node.P_INJ = Pf[i].item()
        node.Q_INJ = Qf[i].item()
   
    
    grid.Line_AC_calc()
      