# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:06:32 2024

@author: BernardoCastro
"""

import csv
import sys
import networkx as nx
import pandas as pd
import numpy as np



def pol2cart(r, theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y


def pol2cartz(r, theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = x+1j*y
    return z


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return rho, theta


def cartz2pol(z):
    r = np.abs(z)
    theta = np.angle(z)
    return r, theta

class Grid:
    def __init__(self, S_base: float, nodes_AC: list = None, lines_AC: list = None, Converters: list = None, nodes_DC: list = None, lines_DC: list = None):
        
        self.Graph_toPlot= nx.MultiGraph()
        self.node_positions={}
        self.S_base = S_base

        self._nodes_AC = nodes_AC if nodes_AC else []
        self._nodes_DC = nodes_DC if nodes_DC else []
        
        self._nodes_dict_AC = None  # Cache for AC nodes dictionary
        self._pq_nodes = None  # Cache for PQ nodes
        self._pv_nodes = None  # Cache for PV nodes
        self._slack_nodes = None  # Cache for Slack nodes
        
        self._nodes_dict_DC = None  # Cache for DC nodes dictionary
        self._PAC_nodes = None  
        self._P_nodes = None  
        self._droop_nodes = None  
        self._slackDC_nodes = None  
        
        
        self.lines_AC = lines_AC if lines_AC else []
        self.lines_AC_exp = []
        self.lines_AC_tf  = []
        
        self.Converters_ACDC = Converters if Converters else []
        
        self.lines_DC = lines_DC if lines_DC else []
    
            
        self.slack_bus_number_AC = []
        self.slack_bus_number_DC = []
        

        self.iter_flow_AC = []
        self.iter_flow_DC = []

        self.OPF_run= False
        self.TEP_run=False
        self.TEP_res=None
        self.time_series_results = {
            'PF_results': pd.DataFrame(),  # Time_series_res
            'line_loading': pd.DataFrame(),  # Time_series_line_res
            'ac_line_loading': pd.DataFrame(),  # TS_AC_line_res
            'dc_line_loading': pd.DataFrame(),  # TS_DC_line_res
            'grid_loading': pd.DataFrame(),  # Time_series_grid_loading
            'converter_p_dc': pd.DataFrame(),  # Time_series_Opt_res_P_conv_DC
            'converter_q_ac': pd.DataFrame(),  # Time_series_Opt_res_Q_conv_AC
            'converter_p_ac': pd.DataFrame(),  # Time_series_Opt_res_P_conv_AC
            'real_power_opf': pd.DataFrame(),  # Time_series_Opt_res_P_extGrid
            'reactive_power_opf': pd.DataFrame(),  # Time_series_Opt_res_Q_extGrid
            'curtailment': pd.DataFrame(),  # Time_series_Opt_curtailment
            'converter_loading': pd.DataFrame(),  # Time_series_conv_res
            'real_power_by_zone': pd.DataFrame(),  # Time_series_Opt_Gen_perPriceZone
            'prices_by_zone': pd.DataFrame()  # Time_series_price
            }
        
        self.VarPrice = False
        self.OnlyGen = True
        self.CurtCost=False

        self.MixedBinCont = False
        self.TEP_n_years = 25
        self.TEP_discount_rate =0.02
        
        
      
 

        if self.nodes_AC:
            self.Update_Graph_AC()
            self.Update_PQ_AC()
           
        if self.nodes_DC:
            self.Update_Graph_DC()
            self.Update_P_DC()
        else:
            self.Num_Grids_DC=0
        # #Call Y bus formula to fill matrix
        self.create_Ybus_AC()
        self.create_Ybus_DC()
        
        
        self.Generators =[]
        
        
        self.RenSource_zones=[]
        self.RenSource_zones_dic={}
        self.RenSources =[]
        self.rs2node = {'DC': {},
                        'AC': {}}
        
        self.Time_series = []
        self.Time_series_dic ={}
        
        self.Price_Zones =[]
        self.Price_Zones_dic ={}
      
        self.Clusters ={}
        
    
        self.OPF_Price_Zones_constraints_used=False
        
   
        self.OWPP_node_to_ts={}
        # Node type differentiation
        
    
    @property
    def nodes_AC(self):
        return self._nodes_AC

    # Setter for AC nodes that updates the dictionary
    @nodes_AC.setter
    def nodes_AC(self, new_nodes_AC):
        self._nodes_AC = new_nodes_AC
        self._invalidate_node_caches()

    # Property for DC nodes
    @property
    def nodes_DC(self):
        return self._nodes_DC

    # Setter for DC nodes that updates the dictionary
    @nodes_DC.setter
    def nodes_DC(self, new_nodes_DC):
        self._nodes_DC = new_nodes_DC
        self._invalidate_DC_caches()  # Invalidate cache when nodes change
        
    # Property to return dictionary of AC nodes
    @property
    def nodes_dict_AC(self):
        if self._nodes_dict_AC is None:  # Rebuild if cache is invalid
            self._nodes_dict_AC = {node.name: idx for idx, node in enumerate(self._nodes_AC)}
        return self._nodes_dict_AC

    # Property to return dictionary of DC nodes
    @property
    def nodes_dict_DC(self):
        if self._nodes_dict_DC is None:  # Rebuild if cache is invalid
            self._nodes_dict_DC = {node.name: idx for idx, node in enumerate(self._nodes_DC)}
        return self._nodes_dict_DC

    # Method to extend AC nodes
    def extend_nodes_AC(self, new_nodes):
        self._nodes_AC.extend(new_nodes)
        self._invalidate_node_caches()  # Invalidate the cache

    # Method to remove an AC node
    def remove_nodes_AC(self, node):
        self._nodes_AC.remove(node)
        self._invalidate_node_caches()  # Invalidate the cache
        
    def _invalidate_node_caches(self):
        """Reset all cached node lists when nodes_AC changes."""
        self._pq_nodes = None
        self._pv_nodes = None
        self._slack_nodes = None
        self._nodes_dict_AC = None
        
    # Method to extend DC nodes
    def extend_nodes_DC(self, new_nodes):
        self._nodes_DC.extend(new_nodes)
        self._invalidate_DC_caches()  # Invalidate the cache

    # Method to remove a DC node
    def remove_nodes_DC(self, node):
        self._nodes_DC.remove(node)
        self._invalidate_DC_caches()  # Invalidate the cache    
        
    def _invalidate_DC_caches(self):
        """Reset all cached DC node lists when nodes_DC changes."""
        self._PAC_nodes = None
        self._P_nodes = None
        self._droop_nodes = None
        self._slackDC_nodes = None
        self._nodes_dict_DC = None        
    # AC grid properties
    @property
    def nn_AC(self):
        return len(self.nodes_AC) if self.nodes_AC is not None else 0  # Number of AC nodes
    
    @property
    def npq(self):
        return len(self.pq_nodes) if self.pq_nodes is not None else 0  # Number of PQ nodes
    
    @property
    def npv(self):
        return len(self.pv_nodes) if self.pv_nodes is not None else 0  # Number of PV nodes
    
    @property
    def pq_nodes(self):
        if self._pq_nodes is None:
            self._pq_nodes = [node for node in self._nodes_AC if node.type == 'PQ']
        return self._pq_nodes

    @property
    def pv_nodes(self):
        if self._pv_nodes is None:
            self._pv_nodes = [node for node in self._nodes_AC if node.type == 'PV']
        return self._pv_nodes

    @property
    def slack_nodes(self):
        if self._slack_nodes is None:
            self._slack_nodes = [node for node in self._nodes_AC if node.type == 'Slack']
        return self._slack_nodes
    
    @property
    def nl_AC(self):
        return len(self.lines_AC) if self.lines_AC is not None else 0   
    
    @property
    def nle_AC(self): 
        return len(self.lines_AC_exp) if self.lines_AC_exp is not None else 0   
    
    @property
    def nttf(self): 
        return len(self.lines_AC_tf) if self.lines_AC_tf is not None else 0     
    
    # DC grid properties
    @property
    def nn_DC(self):
        return len(self.nodes_DC) if self.nodes_DC is not None else 0  # Number of DC nodes
    
    @property
    def nPAC(self):
        return len(self.PAC_nodes) if self.PAC_nodes is not None else 0  # Number of PAC nodes
    
    @property
    def nP(self):
        return len(self.P_nodes) if self.P_nodes is not None else 0  # Number of P nodes
    
    @property
    def nDroop(self):
        return len(self.droop_nodes) if self.droop_nodes is not None else 0  # Number of droop nodes

    @property
    def PAC_nodes(self):
        if self._PAC_nodes is None:
            self._PAC_nodes = [node for node in self._nodes_DC if node.type == 'PAC']
        return self._PAC_nodes

    @property
    def P_nodes(self):
        if self._P_nodes is None:
            self._P_nodes = [node for node in self._nodes_DC if node.type == 'P']
        return self._P_nodes

    @property
    def droop_nodes(self):
        if self._droop_nodes is None:
            self._droop_nodes = [node for node in self._nodes_DC if node.type == 'Droop']
        return self._droop_nodes

    @property
    def slackDC_nodes(self):
        if self._slackDC_nodes is None:
            self._slackDC_nodes = [node for node in self._nodes_DC if node.type == 'Slack']
        return self._slackDC_nodes  

    @property
    def nl_DC(self):
        return len(self.lines_DC) if self.lines_DC is not None else 0   
       
    # ACDC Converter properties
    @property
    def nconv(self):
        return len(self.Converters_ACDC) if self.Converters_ACDC is not None else 0  # Number of converters
    
    @property
    def nconvP(self):
        return len(self.P_Conv) if self.P_Conv is not None else 0  # Number of P converters
    
    @property
    def nconvD(self):
        return len(self.Droop_Conv) if self.Droop_Conv is not None else 0  # Number of Droop converters
    
    @property
    def nconvS(self):
        return len(self.Slack_Conv) if self.Slack_Conv is not None else 0  # Number of Slack converters


    @property
    def P_Conv(self):
        P_Conv = [conv for conv in self.Converters_ACDC if conv.type == 'P']
        return P_Conv

    @property
    def Slack_Conv(self):
        Slack_Conv = [
            conv for conv in self.Converters_ACDC if conv.type == 'Slack']
        return Slack_Conv

    @property
    def Droop_Conv(self):
        Droop_Conv = [
            conv for conv in self.Converters_ACDC if conv.type == 'Droop']
        return Droop_Conv
    
    def Update_Graph_DC(self):
        self.Graph_DC = nx.Graph()

        "Checking for un used nodes "
        used_nodes = set()

        # Iterate through lines
        for line in self.lines_DC:
            used_nodes.add(line.toNode)
            used_nodes.add(line.fromNode)

        # Iterate through converters

        if self.Converters_ACDC != None:
            for converter in self.Converters_ACDC:
                used_nodes.add(converter.Node_DC)

        # Filter out unused nodes
        nodes = [node for node in self.nodes_DC if node in used_nodes]

        for node in nodes:
            self.node_positions[node]=(node.x_coord,node.y_coord)
            
            if node in used_nodes:
                node.used = True

        self.Graph_DC_unused_nodes = [node for node in self.nodes_DC if not node.used]

        for line in self.lines_DC:
            self.Graph_toPlot.add_edge(line.fromNode, line.toNode, line=line)
            self.Graph_DC.add_edge(line.fromNode, line.toNode,line=line)

        self.Grids_DC = list(nx.connected_components(self.Graph_DC))
        self.Num_Grids_DC = len(self.Grids_DC)
        self.Graph_node_to_Grid_index_DC = {}
        self.Graph_line_to_Grid_index_DC = {}
        self.Graph_grid_to_MTDC={}
        
        self.load_grid_DC=np.zeros(self.Num_Grids_DC)
        self.rating_grid_DC=np.zeros(self.Num_Grids_DC)
        self.Graph_number_lines_DC=np.zeros(self.Num_Grids_DC)

        self.Graph_kV_base = np.zeros(self.Num_Grids_DC)
        self.num_MTDC=0
        self.MTDC = {} 
        
        for i, Grid in enumerate(self.Grids_DC):
            for node in Grid:
                self.Graph_node_to_Grid_index_DC[node.nodeNumber] = i
                for line in self.lines_DC:
                    if line.fromNode == node or line.toNode == node:
                        self.Graph_line_to_Grid_index_DC[line] = i
                        self.Graph_kV_base[i] = line.kV_base
        for line in self.lines_DC:
            g=self.Graph_line_to_Grid_index_DC[line]
            self.Graph_number_lines_DC[g]+=1
            self.rating_grid_DC[g]+=line.MW_rating
        
        self.num_slackDC = np.zeros(self.Num_Grids_DC)
        for i in range(self.Num_Grids_DC):
            if self.Graph_number_lines_DC[i] >=2:
                self.MTDC[self.num_MTDC]=i
                self.Graph_grid_to_MTDC[i]=self.num_MTDC
                self.num_MTDC+=1
            for node in self.Grids_DC[i]:
                if node.type == 'Slack':
                    self.num_slackDC[i] += 1

            s = 1
            if self.num_slackDC[i] == 0:
                print(
                    f'For Grid DC {i+1} no slack bus found, results may not be accurate')

            if self.num_slackDC[i] > 1:
                print(f'For Grid DC {i+1} more than one slack bus found')
                sys.exit()
         
        s = 1

   
    def Update_Graph_AC(self):
        self.Graph_AC = nx.MultiGraph()
        

        "Checking for un used nodes "
        used_nodes = set()

        # Iterate through lines
        for line in self.lines_AC:
            used_nodes.add(line.toNode)
            used_nodes.add(line.fromNode)

        # Iterate through converters
        if self.Converters_ACDC != None:

            for converter in self.Converters_ACDC:
                used_nodes.add(converter.Node_AC)
                self.Graph_toPlot.add_node(converter.Node_AC) 

        # Filter out unused nodes
        nodes = [node for node in self.nodes_AC if node in used_nodes]

        for node in nodes:
            self.node_positions[node]=(node.x_coord,node.y_coord)
            
            if node in used_nodes:
                node.used = True

        self.Graph_AC_unused_nodes = [
            node for node in self.nodes_AC if not node.used]

        s = 1

    
        "Creating Graphs to differentiate Grids"
        for line in self.lines_AC + self.lines_AC_exp+ self.lines_AC_tf:
            self.Graph_AC.add_edge(line.fromNode, line.toNode,line=line)
            self.Graph_toPlot.add_edge(line.fromNode, line.toNode,line=line)
            line.toNode.stand_alone = False
            line.fromNode.stand_alone = False
        
        
        
        for node in self.nodes_AC:
            if node.stand_alone:
                node.type = 'Slack'
                self.Graph_AC.add_node(node)
                   
            
        self.Grids_AC = list(nx.connected_components(self.Graph_AC))
        self.Num_Grids_AC = len(self.Grids_AC)
        self.Graph_node_to_Grid_index_AC = {}
        self.Graph_line_to_Grid_index_AC = {}
        self.load_grid_AC=np.zeros(self.Num_Grids_AC)
        self.rating_grid_AC=np.zeros(self.Num_Grids_AC)
        self.Graph_number_lines_AC=np.zeros(self.Num_Grids_AC)

        for i, Grid in enumerate(self.Grids_AC):
            for node in Grid:
                self.Graph_node_to_Grid_index_AC[node.nodeNumber] = i
                for line in self.lines_AC:
                    if line.fromNode == node or line.toNode == node:
                        self.Graph_line_to_Grid_index_AC[line] = i

        for line in self.lines_AC:
            g=self.Graph_line_to_Grid_index_AC[line]
            self.rating_grid_AC[g]+=line.MVA_rating
            self.Graph_number_lines_AC[g]+=1
        "Slack identification"
        self.num_slackAC = np.zeros(self.Num_Grids_AC)

        for i in range(self.Num_Grids_AC):

            for node in self.Grids_AC[i]:
                if node.type == 'Slack':
                    self.num_slackAC[i] += 1
            if self.num_slackAC[i] == 0:
                print(f'For Grid AC {i+1} no slack bus found.')
                print(f'Please set one before any calculations')
                # sys.exit()
            if self.num_slackAC[i] > 1:
                print(
                    f'For Grid AC {i+1} more than one slack bus found, results may not be accurate')
        
        
        s = 1
        
    def get_linesAC_by_node(self, nodeNumber):
        lines = [line for line in self.lines_AC if
                 (line.toNode.nodeNumber == nodeNumber or line.fromNode.nodeNumber == nodeNumber)]
        return lines

    def get_linesDC_by_node(self, nodeNumber):
        lines = [line for line in self.lines_DC if
                 (line.toNode.nodeNumber == nodeNumber or line.fromNode.nodeNumber == nodeNumber)]
        return lines

    def get_lineDC_by_nodes(self, fromNode, toNode):
        lines = [line for line in self.lines_DC if
                 (line.toNode.nodeNumber == fromNode and line.fromNode.nodeNumber == toNode) or
                 (line.toNode.nodeNumber == toNode and line.fromNode.nodeNumber == fromNode)]
        return lines[0] if lines else None

    
    def Update_P_DC(self):

        self.P_DC = np.vstack([node.P_DC for node in self.nodes_DC])
        self.Pconv_DC = np.vstack([node.Pconv for node in self.nodes_DC])
        self.Pconv_DCDC = np.vstack([node.PconvDC for node in self.nodes_DC])
    def Update_PQ_AC(self):
        for node in self.nodes_AC:
            node.Q_s_fx=sum(self.Converters_ACDC[conv].Q_AC for conv  in node.connected_conv if self.Converters_ACDC[conv].AC_type=='PQ')
            node.Q_s   = sum(self.Converters_ACDC[conv].Q_AC for conv  in node.connected_conv if self.Converters_ACDC[conv].AC_type!='PQ')
        # # Negative means power leaving the system, positive means injected into the system at a node  
       
        self.P_AC = np.vstack([node.PGi+sum(rs.PGi_ren*rs.gamma for rs in node.connected_RenSource)
                               +sum(gen.PGen for gen in node.connected_gen)
                               -node.PLi for node in self.nodes_AC])
        self.Q_AC = np.vstack([node.QGi+sum(gen.QGen for gen in node.connected_gen)
                               -node.QLi +node.Q_s_fx for node in self.nodes_AC])
        self.Ps_AC = np.vstack([node.P_s for node in self.nodes_AC])
        self.Qs_AC = np.vstack([node.Q_s for node in self.nodes_AC])

        s = 1

    def create_Ybus_AC(self):
        self.Ybus_AC = np.zeros((self.nn_AC, self.nn_AC), dtype=complex)
        self.AdmitanceVec_AC = np.zeros((self.nn_AC), dtype=complex)
        Ybus_nn= np.zeros((self.nn_AC),dtype=complex)
        # off diagonal elements
        for k in range(self.nl_AC):
            line = self.lines_AC[k]
            fromNode = line.fromNode.nodeNumber
            toNode = line.toNode.nodeNumber

            
            branch_ff = line.Ybus_branch[0, 0]
            branch_ft = line.Ybus_branch[0, 1]
            branch_tf = line.Ybus_branch[1, 0]
            branch_tt = line.Ybus_branch[1, 1]
            
            
            self.Ybus_AC[toNode, fromNode]+=branch_tf
            self.Ybus_AC[fromNode, toNode]+=branch_ft
            
            self.AdmitanceVec_AC[fromNode] += line.Y/2
            self.AdmitanceVec_AC[toNode] += line.Y/2
            
            Ybus_nn[fromNode] += branch_ff
            Ybus_nn[toNode] += branch_tt
            

        for m in range(self.nn_AC):
            node = self.nodes_AC[m]

            self.AdmitanceVec_AC[m] += node.Reactor
            Ybus_nn[m] += node.Reactor
            self.Ybus_AC[m, m] = Ybus_nn[m]
            
    def create_Ybus_DC(self):
        self.Ybus_DC = np.zeros((self.nn_DC, self.nn_DC), dtype=float)

        # off diagonal elements
        for k in range(self.nl_DC):
            line = self.lines_DC[k]
            fromNode = line.fromNode.nodeNumber
            toNode = line.toNode.nodeNumber

            self.Ybus_DC[fromNode, toNode] -= 1/line.Z
            self.Ybus_DC[toNode, fromNode] = self.Ybus_DC[fromNode, toNode]

        # Diagonal elements
        for m in range(self.nn_DC):
            self.Ybus_DC[m, m] = -self.Ybus_DC[:,
                                               m].sum() if self.Ybus_DC[:, m].sum() != 0 else 1.0

    def Check_SlacknDroop(self, change_slack2Droop):
        for conv in self.Converters_ACDC:
            if conv.type == 'Slack':

                DC_node = conv.Node_DC

                node_count = 0

                P_syst = 0
                for conv_other in self.Converters_ACDC:
                    DC_node_other = conv_other.Node_DC
                    connected = nx.has_path(
                        self.Graph_DC, DC_node, DC_node_other)
                    if connected == True:
                        P_syst += -conv_other.P_DC
                    else:
                        # print(f"Nodes {DC_node.name} and {DC_node_other.name} are not connected.")
                        node_count += 1

                if change_slack2Droop == True:
                    if self.nn_DC-node_count != 2:

                        conv.type = 'Droop'
                        DC_node.type = 'Droop'
                conv.P_DC = P_syst
                DC_node.Pconv = P_syst

                self.Update_P_DC()

            elif conv.type == 'Droop':

                DC_node = conv.Node_DC

                node_count = 0

                for conv_other in self.Converters_ACDC:
                    DC_node_other = conv_other.Node_DC
                    connected = nx.has_path(self.Graph_DC, DC_node, DC_node_other)
                    if connected == False:
                        node_count += 1

                if self.nn_DC-node_count == 2:
                    g=self.Graph_node_to_Grid_index_DC[DC_node.nodeNumber]
                    
                    if any(node.type == 'Slack' for node in self.Grids_DC[g]):
                        s=1
                    else:
                        conv.type = 'Slack'
                        DC_node.type = 'Slack'
                        print(f"Changing converter {conv.name} to Slack")
                self.Update_P_DC()

    

    def Line_AC_calc(self):
        try: 
            V_cart = pol2cartz(self.V_AC, self.Theta_V_AC)
        except: 
            self.V_AC =np.zeros(self.nn_AC)
            self.Theta_V_AC=np.zeros(self.nn_AC)
            for node in self.nodes_AC: 
                nAC=node.nodeNumber
                self.V_AC[nAC]=node.V
                self.Theta_V_AC[nAC]=node.theta
            V_cart = pol2cartz(self.V_AC, self.Theta_V_AC)
            
        
        self.I_AC_cart = np.matmul(self.Ybus_AC, V_cart)
        self.I_AC_m = abs(self.I_AC_cart)
        self.I_AC_th = np.angle(self.I_AC_cart)

  
        for line in self.lines_AC:
            i = line.fromNode.nodeNumber
            j = line.toNode.nodeNumber
        
            i_from = line.Ybus_branch[0,0]*V_cart[i]+line.Ybus_branch[0,1]*V_cart[j]
            i_to = line.Ybus_branch[1,0]*V_cart[i]+line.Ybus_branch[1,1]*V_cart[j]
            
            Sfrom = V_cart[i]*np.conj(i_from)
            Sto = V_cart[j]*np.conj(i_to)
        
            line.loss = Sfrom+Sto
            
            line.fromS=Sfrom
            line.toS=Sto
            line.i_from,_ = cartz2pol(i_from)
            line.i_to,_ = cartz2pol(i_to)


    def Line_DC_calc(self):
        V = self.V_DC
        Ybus = self.Ybus_DC
        self.I_DC = np.matmul(Ybus, V)

        Iij = np.zeros((self.nn_DC, self.nn_DC), dtype=float)
        Pij_DC = np.zeros((self.nn_DC, self.nn_DC), dtype=float)

        s = 1
        for line in self.lines_DC:
            i = line.fromNode.nodeNumber
            j = line.toNode.nodeNumber
            pol = line.pol

            Iij[i, j] = (V[i]-V[j])*-Ybus[i, j]
            Iij[j, i] = (V[j]-V[i])*-Ybus[i, j]

            Pij_DC[i, j] = V[i]*(Iij[i, j])*pol
            Pij_DC[j, i] = V[j]*(Iij[j, i])*pol
            
            line.toP=Pij_DC[j,i]*line.np_line
            line.fromP=Pij_DC[i,j]*line.np_line

        L_loss = np.zeros(self.nl_DC, dtype=float)

        for line in self.lines_DC:
            l = line.lineNumber
            i = line.fromNode.nodeNumber
            j = line.toNode.nodeNumber

            L_loss[l] = (Pij_DC[i, j]+Pij_DC[j, i])*line.np_line
            line.loss = (Pij_DC[i, j]+Pij_DC[j, i])*line.np_line

        self.L_loss_DC = L_loss

        self.Pij_DC = Pij_DC

        self.Iij_DC = Iij
        s = 1

        
        
class Gen_AC:
    genNumber =0
    names = set()
    
    @classmethod
    def reset_class(cls):
        cls.genNumber = 0
        cls.names = set()
             
    @property
    def name(self):
        return self._name

    
    def __init__(self,name, node,Max_pow_gen: float,Min_pow_gen: float,Max_pow_genR: float,Min_pow_genR: float,quadratic_cost_factor: float=0,linear_cost_factor: float=0,Pset:float=0,Qset:float=0,S_rated:float=None,gen_type='Other'):
        self.genNumber = Gen_AC.genNumber
        Gen_AC.genNumber += 1
        self.Node_AC=node.name
        self.geometry= node.geometry
        self.kV_base = node.kV_base
        self.PZ = node.PZ
        self.hover_text = None
        self.gen_type=gen_type
        self.Max_pow_gen=Max_pow_gen
        self.Min_pow_gen=Min_pow_gen
        self.Max_pow_genR=Max_pow_genR
        self.Min_pow_genR=Min_pow_genR
        
        self.Max_S= S_rated
        
        self.lf=linear_cost_factor
        self.qf=quadratic_cost_factor
        
        self.price_zone_link = False
        
        node.connected_gen.append(self)
        
        self.PGen=Pset
        self.QGen=Qset
        
        self.Pset=Pset
        self.Qset=Qset
        
        if name in Gen_AC.names:
            count = 1
            new_name = f"{name}_{count}"
            
            while new_name in Gen_AC.names:
                count += 1
                new_name = f"{name}_{count}"
            name = new_name
        if name is None:
            self._name = str(node.name)
        else:
            self._name = name

        Gen_AC.names.add(self.name)
        
class Ren_Source:
    rsNumber =0
    names = set()
    
    @classmethod
    def reset_class(cls):
        cls.rsNumber = 0
        cls.names = set()
             
    @property
    def name(self):
        return self._name

    
    def __init__(self,name,node,PGi_ren_base: float,rs_type='Wind'):
        self.rsNumber = Ren_Source.rsNumber
        Ren_Source.rsNumber += 1
        
        self.connected= 'AC'
        self.rs_type = rs_type
        
        self.curtailable= True
       
        
        self.Node=node.name
        
        self.geometry= node.geometry
        self.kV_base = node.kV_base
        self.PZ = node.PZ
        
        self.PGi_ren_base=PGi_ren_base
        self.PGi_ren = 0 
        self._PRGi_available=1
        
        
        self.TS_dict = {
            'PRGi_available': None
        }
        
        self.PGRi_linked=False
        self.Ren_source_zone=None
        
        self.gamma = 1
        self.min_gamma = 0.0
        self.sigma=1.05
        
        self.QGi_ren = 0
        self.Qmax=0
        self.Qmin=0
        
        self.Max_S= None
            
        node.connected_RenSource.append(self)
        node.RenSource=True
        
        self.update_PGi_ren()
        
        if name in Ren_Source.names:
            count = 1
            new_name = f"{name}_{count}"
            
            while new_name in Ren_Source.names:
                count += 1
                new_name = f"{name}_{count}"
            name = new_name
        if name is None:
            self._name = str(self.Node.name)
        else:
            self._name = name

        Ren_Source.names.add(self.name)
        
        self.hover_text = None
    @property
    def PRGi_available(self):
        return self._PRGi_available

    @PRGi_available.setter
    def PRGi_available(self, value):
        self._PRGi_available = value
        self.update_PGi_ren()
     
    def update_PGi_ren(self):
        self.PGi_ren = self.PGi_ren_base * self._PRGi_available
   
    
class Node_AC:  
    nodeNumber = 0
    names = set()
    
    @classmethod
    def reset_class(cls):
        cls.nodeNumber = 0
        cls.names = set()
        
    @property
    def name(self):
        return self._name

    def __init__(self, node_type: str, Voltage_0: float, theta_0: float,kV_base:float, Power_Gained: float=0, Reactive_Gained: float=0, Power_load: float=0, Reactive_load: float=0, name=None, Umin=0.9, Umax=1.1,Gs:float= 0,Bs:float=0,x_coord=None,y_coord=None):
        # type: (1=PQ, 2=PV, 3=Slack)
        self.nodeNumber = Node_AC.nodeNumber
        Node_AC.nodeNumber += 1
        self.type = node_type

        self.kV_base = kV_base
        self.V_ini = Voltage_0
        self.theta_ini = theta_0
        self.V = np.copy(self.V_ini)
        self.theta = np.copy(self.theta_ini)
        self.PGi = Power_Gained
        self.PGi_opt =0
        
        # self.PGi_ren_base=0
        self.PGi_ren= 0 
        # self._PRGi_available=1
        self.RenSource=False
        # self.PGRi_linked=False
        # self.Ren_source_zone=None
        
        self.PLi_linked= True
        self.PLi= Power_load
        self.PLi_base = Power_load
        self._PLi_factor =1
        
        self.TS_dict = {
            'Load' : None,
            'price': None,
            }
        
        self.QGi = Reactive_Gained
        self.QGi_opt =0
        self.QLi = Reactive_load
        
       

        self.Qmin = 0
        self.Qmax = 0
        self.Reactor = Gs+ Bs*1j
        # self.Q_max = Q_max
        # self.Q_min = Q_min
        # self.P_AC = self.PGi-self.PLi
        # self.Q_AC = self.QGi-self.QLi
        self.P_s = 0
        self.Q_s = 0
        self.Q_s_fx = 0  # reactive power by converters in PQ mode
        self.P_s_new = np.copy(self.P_s)
        self.used = False
        self.stand_alone = True
        
       

        self.price = 0.0
        self.Num_conv_connected=0
        self.connected_conv=set()
   
        
        self.curtailment=1

        # self.Max_pow_gen=0
        # self.Min_pow_gen=0
        # self.Max_pow_genR=0
        # self.Min_pow_genR=0
        self.connected_gen=[]
        self.connected_RenSource=[]
        self.connected_toExpLine=[]
        self.connected_fromExpLine=[]
        self.connected_toTFLine=[]
        self.connected_fromTFLine=[]
        
        
        self.Umax= Umax
        self.Umin=Umin
        
        self.x_coord=x_coord
        self.y_coord=y_coord
        
        self.PZ = None
        self.hover_text = None
        self.geometry=None
        if name in Node_AC.names:
            Node_AC.nodeNumber -= 1
            raise NameError("Already used name '%s'." % name)
        if name is None:
            self._name = str(self.nodeNumber)
        else:
            self._name = name

        Node_AC.names.add(self.name)
  
            
    @property
    def PLi_factor(self):
        return self._PLi_factor

    @PLi_factor.setter
    def PLi_factor(self, value):
        self._PLi_factor = value
        self.update_PLi()        
       
    def update_PLi(self):
        self.PLi = self.PLi_base * self._PLi_factor
        
class Node_DC:
    nodeNumber = 0
    names = set()
    
    @classmethod
    def reset_class(cls):
        cls.nodeNumber = 0
        cls.names = set()
    
    @property
    def name(self):
        return self._name

    def __init__(self, node_type: str, Voltage_0: float, Power_Gained: float, Power_load: float,kV_base:float, name=None, Umin=0.95, Umax=1.05,x_coord=None,y_coord=None):
        # type: (1=P, 2=Droop, 3=Slack)
        self.nodeNumber = Node_DC.nodeNumber
        Node_DC.nodeNumber += 1

        self.V_ini = Voltage_0
        self.type = node_type
        self.kV_base = kV_base
        
        self.PGi = Power_Gained
        self.PLi_linked= True
        self.PLi= Power_load
        self.PLi_base = Power_load
        self._PLi_factor =1
        
        self.TS_dict = {
            'Load' : None,
            'price': None,
            }
        
        self.P_DC = self.PGi-self.PLi
        self.V = np.copy(self.V_ini)
        self.P_INJ = 0
        self.Pconv = 0
        self.used = False
        self.PconvDC = 0
        self.P = 0
        
        self.price = 0.0
        
        self.Nconv= None
        self.Nconv_i=None
        self.ConvInv = False
        self.conv_loading=0
        self.conv_MW= 0
        
        self.Umax=Umax
        self.Umin=Umin
        
        self.x_coord=x_coord
        self.y_coord=y_coord
        
        self.PZ = None
        self.hover_text = None
        self.geometry=None
        self.connected_RenSource=[]
        
        
        if name in Node_DC.names:
            Node_DC.nodeNumber -= 1
            raise NameError("Already used name '%s'." % name)
        if name is None:
            self._name = str(self.nodeNumber)
        else:
            self._name = name

        Node_DC.names.add(self.name)

    @property
    def PLi_factor(self):
         return self._PLi_factor

    @PLi_factor.setter
    def PLi_factor(self, value):
         self._PLi_factor = value
         self.update_PLi()        
        
    def update_PLi(self):
         self.PLi = self.PLi_base * self._PLi_factor
         
class Line_AC:
    lineNumber = 0
    names = set()

    @classmethod
    def reset_class(cls):
        cls.lineNumber = 0
        cls.names = set()
        
    @property
    def name(self):
        return self._name
    
    def get_relevant_attributes(self):
        """Method to return only the relevant attributes for the subclass."""
        return {
            'fromNode': self.fromNode,
            'toNode': self.toNode,
            'Resistance': self.R,
            'Reactance': self.X,
            'Conductance': self.G,
            'Susceptance': self.B,
            'MVA_rating': self.MVA_rating,
            'kV_base': self.kV_base,
            'Length_km':self.Length_km,
            'm': self.m,
            'shift': self.shift,
            'name': self._name,
            'geometry':self.geometry,
            'isTf': self.isTf
        }
    
    def remove(self):
        """Method to handle line removal from the class-level attributes."""
        Line_AC.lineNumber -= 1  # Decrement the line number counter
        Line_AC.names.remove(self._name)  # Remove the line's name from the set
        
    def __init__(self, fromNode: Node_AC, toNode: Node_AC, Resistance: float, Reactance: float, Conductance: float, Susceptance: float, MVA_rating: float, kV_base: float,Length_km:float=1.0,m:float=1, shift:float=0, name=None,geometry=None,isTf=False):
        self.lineNumber = Line_AC.lineNumber
        Line_AC.lineNumber += 1

        self.fromNode = fromNode
        self.toNode = toNode
        self.R = Resistance
        self.X = Reactance
        self.G = Conductance
        self.B = Susceptance
        self.Z = self.R + self.X * 1j
        self.Y = self.G + self.B * 1j
        self.kV_base = kV_base
        self.MVA_rating = MVA_rating
        
        self.m =m
        self.shift = shift
        
        self.fromS=0
        self.toS=0
        
        self.loss =0
        
        self.geometry=geometry
        self.direction = 'from'
        
        self.Length_km=Length_km
        
        tap= self.m * np.exp(1j*self.shift)            
        #Yft
        branch_ft = -(1/self.Z)/np.conj(tap)
        
        #Ytf
        branch_tf = -(1/self.Z)/tap
        
        branch_ff=(1/self.Z+self.Y/2)/(self.m**2)
        branch_tt=(1/self.Z+self.Y/2)
        
        self.Ybus_branch=np.array([[branch_ff, branch_ft],[branch_tf, branch_tt]])
        
        

        if name in Line_AC.names:
            Line_AC.lineNumber -= 1
            raise NameError("Already used name '%s'." % name)

        if name is None:
            self._name = str(self.lineNumber)
        else:
            self._name = name
            
        self.hover_text = None  
        
        self.isTf = isTf
        
        if self.toNode.kV_base != self.fromNode.kV_base or self.m !=1 or self.shift !=0:
            self.isTf=True
        Line_AC.names.add(self.name)

class Exp_Line_AC(Line_AC):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        
        self.direction = 'from'
        self.base_cost = None
        self.life_time = 1
        self.exp_inv=1
        self.cost_perMVAkm = None
        self.phi=0
        
        self.np_line=0
        self.np_line_i= 0
        self.np_line_max = 1
        self.np_line_opf=True
        self.hover_text = None
        
        self.toNode.connected_toExpLine.append(self)
        self.fromNode.connected_fromExpLine.append(self)

class TF_Line_AC:
    trafNumber = 0
    names = set()

    @classmethod
    def reset_class(cls):
        cls.trafNumber = 0
        cls.names = set()
        
    @property
    def name(self):
        return self._name

    def __init__(self, fromNode: Node_AC, toNode: Node_AC, Resistance: float, Reactance: float, Conductance: float, Susceptance: float, MVA_rating: float, kV_base: float,m:float=1, shift:float=0, name=None):
        self.trafNumber = TF_Line_AC.trafNumber
        TF_Line_AC.trafNumber += 1

        self.fromNode = fromNode
        self.toNode = toNode
        self.direction = 'from'
        self.R = Resistance
        self.X = Reactance
        self.G = Conductance
        self.B = Susceptance
        self.Z = self.R + self.X * 1j
        self.Y = self.G + self.B * 1j
        self.kV_base = kV_base
        self.MVA_rating = MVA_rating
        
        self.m =m
        self.shift = shift
        
        tap= self.m * np.exp(1j*self.shift)            
        #Yft
        branch_ft = -(1/self.Z)/np.conj(tap)
        
        #Ytf
        branch_tf = -(1/self.Z)/tap
        
        branch_ff=(1/self.Z+self.Y/2)/(self.m**2)
        branch_tt=(1/self.Z+self.Y/2)
        
        self.Ybus_branch=np.array([[branch_ff, branch_ft],[branch_tf, branch_tt]])
        
        self.fromS=0
        self.toS=0
        
        self.toNode.connected_toTFLine.append(self)
        self.fromNode.connected_fromTFLine.append(self)

        self.hover_text = None
        self.isTf = True
        if name in TF_Line_AC.names:
            TF_Line_AC.trafNumber -= 1
            raise NameError("Already used name '%s'." % name)

        if name is None:
            self._name = str(self.lineNumber)
        else:
            self._name = name
            
        TF_Line_AC.names.add(self.name)
        

class Line_DC:
    lineNumber = 0
    names = set()

    @classmethod
    def reset_class(cls):
        cls.lineNumber = 0
        cls.names = set()    

    @property
    def name(self):
        return self._name

    def __init__(self, fromNode: Node_DC, toNode: Node_DC, Resistance: float, MW_rating: float, kV_base: float,km:float=1, polarity='m', name=None):
        self.lineNumber = Line_DC.lineNumber
        Line_DC.lineNumber += 1

        self.m_sm_b = polarity

        if polarity == 'm':
            self.pol = 1
        elif polarity == 'b' or polarity == 'sm':
            self.pol = 2
        else:
            print('No viable polarity inserted pol =1')
            self.pol = 1

        self.fromNode = fromNode
        self.toNode = toNode
        self.R = Resistance
        self.MW_rating = MW_rating
        
        self.Z = self.R

        self.fromP=0
        self.toP=0        
        self.direction = 'from'
        
        self.Length_km=km
        
        self.loss =0
        
        self.base_cost = None
        self.life_time = None
        self.exp_inv=1
        self.cost_perMWkm = None
        self.phi=1
        
        self.np_line=1
        self.np_line_i= 1
        self.np_line_max = 1
        self.np_line_opf=False

        self.kV_base = kV_base
         
        self.hover_text = None
        self.geometry=None
        if name in Line_DC.names:
            Line_DC.lineNumber -= 1
            raise NameError("Already used name '%s'." % name)

        if name is None:
            self._name = str(self.lineNumber)
        else:
            self._name = name

        Line_DC.names.add(self.name)

class AC_DC_converter:
    ConvNumber = 0
    names = set()
    
    @classmethod
    def reset_class(cls):
        cls.ConvNumber = 0
        cls.names = set()
    
    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value
        self.Node_DC.type = value  # Update DC_node type when converter type changes

    @property
    def NumConvP(self):
        return self._NumConvP

    @NumConvP.setter
    def NumConvP(self, value):
        self._NumConvP = value
        self.Node_DC.Nconv= value
        P_DC = self.P_DC
        P_s = self.P_AC 
        Q_s = self.Q_AC 
        S = np.sqrt(P_s**2 + Q_s**2)
        self.Node_DC.conv_loading = max(S, abs(P_DC)) 
        
            
            
    def __init__(self, AC_type: str, DC_type: str, AC_node: Node_AC, DC_node: Node_DC, P_AC: float, Q_AC: float, P_DC: float, Transformer_resistance: float, Transformer_reactance: float, Phase_Reactor_R: float, Phase_Reactor_X: float, Filter: float, Droop: float, kV_base: float, MVA_max: float = 1.05,nConvP: float =1,polarity: int =1 ,lossa:float=1.103,lossb:float= 0.887,losscrect:float=2.885,losscinv:float=4.371,Ucmin: float = 0.85, Ucmax: float = 1.2, name=None):
        self.ConvNumber = AC_DC_converter.ConvNumber
        AC_DC_converter.ConvNumber += 1
        # type: (1=P, 2=droop, 3=Slack)
        
        self._NumConvP= nConvP
        self.NumConvP_i= nConvP
        self.NumConvP_max = nConvP
        
        self.NUmConvP_opf=False
        self.base_cost = None
        self.life_time = None
        self.exp_inv=1
        self.cost_perMVA = None
        self.phi=1
 
        self.cn_pol=   polarity 
        
        self.Droop_rate = Droop
        
        self.AC_type = AC_type

        self.AC_kV_base = kV_base

        self.Node_AC = AC_node
        
        AC_node.Num_conv_connected+=1
        
        self.Node_DC = DC_node
        self.Node_DC.Nconv= nConvP
        self.Node_DC.Nconv_i= nConvP
        self.Node_DC.ConvInv =True
        self.Node_DC.conv_MW=MVA_max* self.cn_pol
        # if self.AC_type=='Slack':
        #     # print(name)mm
        #     self.type='PAC'
        
        self.type = DC_type

        self.R_t = Transformer_resistance/self.cn_pol
        self.X_t = Transformer_reactance /self.cn_pol
        self.PR_R = Phase_Reactor_R /self.cn_pol
        self.PR_X = Phase_Reactor_X /self.cn_pol
        self.Bf = Filter * self.cn_pol
        self.P_DC = P_DC
        self.P_AC = P_AC
        self.Q_AC = Q_AC
        
        # self.Node_DC.type = DC_type
        self.Node_DC.Droop_rate = self.Droop_rate
        self.Node_DC.Pconv = self.P_DC
        
         
        self.a_conv_og  = lossa  * self.cn_pol # MVA
        self.b_conv_og  = lossb                  # kV
        self.c_rect_og  = losscrect  /self.cn_pol  # Ohm
        self.c_inver_og = losscinv /self.cn_pol  # Ohm

        # 1.103 0.887  2.885    4.371

        self.P_loss = 0
        self.P_loss_tf = 0

        self.U_s = 1
        if P_DC > 0:
            self.U_c = 0.98
            self.U_f = 0.99
        else:
            self.U_c = 1.1
            self.U_f = 1.05
        self.th_s = 0.09
        self.th_f = 0.1
        self.th_c = 0.11

        self.MVA_max = MVA_max * self.cn_pol
        self.Ucmin = Ucmin
        self.Ucmax = Ucmax
        self.OPF_fx=False
        self.OPF_fx_type='PDC'
        
        if self.AC_type=='Slack':
            self.OPF_fx_type='None'
            
        if self.AC_type == 'PV':
            if self.type == 'PAC':
                self.OPF_fx_type='PV'
            elif self.type == 'Slack':
                self.OPF_fx_type='None'
            if self.Node_AC.type == 'PQ':
                self.Node_AC.type = 'PV'
        if self.AC_type == 'PQ':
            if  self.type == 'PAC':
                self.OPF_fx_type='PQ'
            else:
                self.OPF_fx_type='Q'
            self.Node_AC.Q_s_fx += self.Q_AC
           

        self.Qc = 0
        self.Pc = 0

        self.Ztf = self.R_t+1j*self.X_t
        self.Zc = self.PR_R+1j*self.PR_X
        if self.Bf != 0:
            self.Zf = 1/(1j*self.Bf)
        else:
            self.Zf = 0

        if self.R_t != 0:
            self.Y_tf = 1/self.Ztf
            self.Gtf = np.real(self.Y_tf)
            self.Btf = np.imag(self.Y_tf)
        else:
            self.Gtf = 0
            self.Btf = 0

        if self.PR_R != 0:
            self.Y_c = 1/self.Zc
            self.Gc = np.real(self.Y_c)
            self.Bc = np.imag(self.Y_c)
        else:
            self.Gc = 0
            self.Bc = 0
            
        self.Z1 = 0
        self.Z2 = 0
        self.Z3 = 0
        if self.Zf != 0:
            self.Z2 = (self.Ztf*self.Zc+self.Zc*self.Zf+self.Zf*self.Ztf)/self.Zf
        if self.Zc != 0:
            self.Z1 = (self.Ztf*self.Zc+self.Zc*self.Zf+self.Zf*self.Ztf)/self.Zc
        if self.Ztf != 0:
            self.Z3 = (self.Ztf*self.Zc+self.Zc*self.Zf+self.Zf*self.Ztf)/self.Ztf


        self.hover_text = None
        self.geometry = None

        if name in AC_DC_converter.names:
            AC_DC_converter.ConvNumber -= 1
            raise NameError("Already used name '%s'." % name)

        if name is None:
            self._name = str(self.ConvNumber)
        else:
            self._name = name

        AC_DC_converter.names.add(self.name)
        self.Node_AC.connected_conv.add(self.ConvNumber)

        

class DC_DC_converter:
    ConvNumber = 0
    names = set()

    @classmethod
    def reset_class(cls):
        cls.ConvNumber = 0
        cls.names = set()

    
    @property
    def name(self):
        return self._name

    def __init__(self, element_type: str, fromNode: Node_DC, toNode: Node_DC, PowerTo: float, R: float, name=None):
        self.ConvNumber = DC_DC_converter.ConvNumber
        DC_DC_converter.ConvNumber += 1
        # type: (1=P, 2=droop, 3=Slack)
        self.type = element_type
        self.ConvNumber = DC_DC_converter.ConvNumber
        self.fromNode = fromNode
        self.toNode = toNode
        self.PowerTo = PowerTo
        self.R = R

        toNode.PconvDC += self.PowerTo
        self.Powerfrom = self.PowerTo+self.PowerTo**2*R

        fromNode.PconvDC -= self.Powerfrom

        
        if name is None:
            self._name = str(self.ConvNumber)
        else:
            self._name = name

        DC_DC_converter.names.add(self.name)

class Ren_source_zone:
    ren_source_num = 0
    names  = set()
    @classmethod
    def reset_class(cls):
        cls.ren_source_num = 0
        cls.names = set()
    
    @property
    def name(self):
        return self._name
    
    @property
    def PRGi_available(self):
        return self._PRGi_available

    @PRGi_available.setter
    def PRGi_available(self, value):
        self._PRGi_available = value
        for ren_source in self.RenSources:
                ren_source.PRGi_available=value
                ren_source.Ren_source_zone = self.name
       
    def __init__(self,name=None):
           self.ren_source_num = Ren_source_zone.ren_source_num
           Ren_source_zone.ren_source_num += 1
           
           self.RenSources=[]
           self._PRGi_available=1
           
           self.TS_dict = {
               'PRGi_available': None
           }
           
           if name is None:
               self._name = str(self.ren_source_num)
           else:
               self._name = name

class Price_Zone:
    price_zone_num = 0
    names = set()
    
    @classmethod
    def reset_class(cls):
        cls.price_zone_num = 0
        cls.names = set()
    
    @property
    def name(self):
        return self._name
    
    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, value):
        self._price = value
        for node in self.nodes_AC:
            node.price=value
            for gen in node.connected_gen:
                if gen.price_zone_link:
                    gen.lf=value
                    gen.qf=0
        # Notify all linked MTDC price_zones about the price change
        for mtdc_price_zone in self.mtdc_price_zones:
            mtdc_price_zone.update_price()  # Automatically update MTDC price_zone's price
            
        # If this price_zone has a linked price_zone, update the linked price_zone's price
        if self.linked_price_zone is not None:
            self.linked_price_zone.price = value  # This will trigger the price setter of the offshore price_zone

    
    @property
    def PLi_factor(self):
        return self._PLi_factor

    @PLi_factor.setter
    def PLi_factor(self, value):
        self._PLi_factor = value
        for node in self.nodes_AC:
            if node.PLi_linked:
                node.PLi_factor=value
        for node in self.nodes_DC:
            if node.PLi_linked:
                node.PLi_factor=value        

    def __init__(self,price=1,import_pu_L=1,export_pu_G=1,a=0,b=1,c=0,import_expand=0,name=None):
        self.price_zone_num = Price_Zone.price_zone_num
        Price_Zone.price_zone_num += 1
        
        self.import_pu_L=import_pu_L
        self.export_pu_G=export_pu_G
        self.nodes_AC=[]
        self.nodes_DC=[]
        self.ConvACDC=[]
        
        self._price=price
        self.a=a
        self.b=b
        self.c=c
        self.PGL_min=-np.inf
        self.PGL_max=np.inf
        
        
        self.TS_dict = {
            'Load' : None,
            'price': None,
            'a_CG': None,
            'b_CG': None,
            'c_CG': None,
            'PGL_min': None,
            'PGL_max': None
        }
        
        self.df= pd.DataFrame(columns=['time','a', 'b', 'c','price','PGL_min','PGL_max'])        
        self.df.set_index('time', inplace=True)
        self.mtdc_price_zones=[]
        
        self._PLi_factor=1
        
        self.ImportExpand_og=import_expand
        self.ImportExpand=import_expand
        if name is None:
            self._name = str(self.price_zone_num)
        else:
            self._name = name

        Price_Zone.names.add(self.name)
        
        # To hold the linked price_zone
        self.linked_price_zone = None
    
    
    def link_mtdc_price_zone(self, mtdc_price_zone):
        """Register an MTDC price_zone to be notified when this price_zone's price changes."""
        if mtdc_price_zone not in self.mtdc_price_zones:
            self.mtdc_price_zones.append(mtdc_price_zone)
            
    def link_price_zone(self, other_price_zone):
        """Link another price_zone to this price_zone"""
        self.linked_price_zone = other_price_zone
        
        other_price_zone.price = self.price  # Initially synchronize the price

class OffshorePrice_Zone(Price_Zone):
    def __init__(self, main_price_zone, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_price_zone = main_price_zone  # Keep reference to the main price_zone
        
        # Automatically set specific attributes for OffshorePrice_Zone
        self.a = 0
        self.b = 0
        self.PGL_min = 0
        self.PGL_max = np.inf

    @property
    def price(self):
        return self._price
    
    @price.setter
    def price(self, value):
        if value != self.main_price_zone.price:
            return  # Do not change offshore price_zone's price if it doesn't match main price_zone's price
        
        # Set the offshore price_zone price and update the nodes' prices
        self._price = value
        for node in self.nodes_AC:
            node.price = value  # Update prices of nodes in the offshore price_zone

class MTDCPrice_Zone(Price_Zone):
    def __init__(self, linked_price_zones=None, pricing_strategy='avg', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linked_price_zones = linked_price_zones or []  # List to store linked price_zones
        self.pricing_strategy = pricing_strategy  # 'min', 'max', or 'avg'
        self.a = 0  # Default specific MTDC properties
        self.b = 0
        self.PGL_min = 0
        self.PGL_max = np.inf
        # Register this MTDC price_zone with the linked price_zones
        for price_zone in self.linked_price_zones:
            price_zone.link_mtdc_price_zone(self)
        
        self.update_price()  # Set initial price based on linked price_zones

    def add_linked_price_zone(self, price_zone):
        """Add a price_zone to the linked price_zones list."""
        if price_zone not in self.linked_price_zones:
            self.linked_price_zones.append(price_zone)
            price_zone.link_mtdc_price_zone(self)
            self.update_price()  # Update price whenever a new price_zone is added

    def update_price(self):
        """Update the price of the MTDC price_zone based on the linked price_zones and strategy."""
        if not self.linked_price_zones:
            return  # No linked price_zones, no price change

        prices = [price_zone.price for price_zone in self.linked_price_zones]
        
        if self.pricing_strategy == 'min':
            self._price = min(prices)
        elif self.pricing_strategy == 'max':
            self._price = max(prices)
        elif self.pricing_strategy == 'avg':
            self._price = sum(prices) / len(prices)

        # Update node prices based on the new MTDC price
        for node in self.nodes_AC:
            node.price = self._price

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, value):
        # Price is managed by linked price_zones, so don't allow manual setting
        s=1
    # Allow for manual override if desired
    def override_price(self, value):
        self._price = value
        for node in self.nodes_AC:
            node.price = value
            
            
            
class TimeSeries:
    TS_num = 0
    names = set()
    
    @classmethod
    def reset_class(cls):
        cls.TS_num = 0
        cls.names = set()
    
    @property
    def name(self):
        return self._name

    def __init__(self, element_type: str, element_name:str, data: float, name=None):
        self.TS_num = TimeSeries.TS_num
        TimeSeries.TS_num += 1
        
        
        self.type = element_type
        self.element_name=element_name
        self.data = data
        
        s = 1
        if name is None:
            self._name = str(self.TS_AC_num)
        else:
            self._name = name

        TimeSeries.names.add(self.name)
