# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:59:08 2024

@author: BernardoCastro
"""
import numpy as np
from prettytable import PrettyTable as pt
import matplotlib.pyplot as plt
import pandas as pd

from .Classes import Price_Zone


class Results:
    def __init__(self, Grid, decimals=2, export=None):
        self.Grid = Grid
        self.dec = decimals
        self.export = export

    def options(self):
        # Get all attributes (including methods) of the class
        all_attributes = dir(self)

        # Filter out only the methods (defs)
        methods = [attribute for attribute in all_attributes if callable(
            getattr(self, attribute)) and not attribute.startswith('__')]

        # Print the method names
        for method_name in methods:
            print(method_name)
    # def export(self):

    def All(self):
        if self.Grid.nodes_AC != []:
            self.AC_Powerflow()
            self.AC_voltage()
            self.AC_lines_current()
            self.AC_lines_power()
        
        if self.Grid.nodes_DC != []:
            if self.Grid.nconv != 0:
                self.Converter()
            self.DC_bus()
            self.DC_lines_current()
            self.DC_lines_power()
            

            if self.Grid.Converters_DCDC != []:
                self.DC_converter()
        
        if self.Grid.nodes_AC != [] and self.Grid.nodes_DC != []:
            self.Slack_All()
            
        elif self.Grid.nodes_AC != []:
            self.Slack_AC()

        self.Power_loss()
        if self.Grid.OPF_run :
            if self.Grid.Generators != [] or self.Grid.Generators_DC != []:
                self.Ext_gen()
            if self.Grid.RenSources:
                self.Ext_REN()
            if not self.Grid.TEP_run and not self.Grid.MP_TEP_run:
                self.OBJ_res()
            if self.Grid.Price_Zones != []: 
                self.Price_Zone()    
        if self.Grid.lines_AC_exp+self.Grid.lines_AC_rec+self.Grid.lines_AC_ct != []:
            self.AC_exp_lines_power()
        if self.Grid.TEP_run:    
            self.TEP_N()
            if self.Grid.TEP_res is not None:
                self.TEP_TS_norm()
                self.TEP_ts_res()
            else:
                self.TEP_norm()
        if self.Grid.MP_TEP_run:
            self.MP_TEP_results()
        print('------')

    def All_AC(self):
        self.AC_Powerflow()
        self.AC_voltage()
        self.AC_lines_current()
        self.AC_lines_power()
        self.Slack_AC()
        self.Power_loss_AC()

    def All_DC(self):

        if self.Grid.nconv != 0:
            self.Converter()

        self.DC_bus()

        self.DC_lines_current()
        self.DC_lines_power()
        self.Slack_DC()
        self.Power_loss_DC()

    def Slack_All(self):
        table = pt()
        # Define the table headers
        table.field_names = ["Grid", "Slack node"]

        for i in range(self.Grid.Num_Grids_AC):
            for node in self.Grid.Grids_AC[i]:
                if node.type == 'Slack':
                    table.add_row([f'AC Grid {i+1}', node.name])
        for i in range(self.Grid.Num_Grids_DC):
            for node in self.Grid.Grids_DC[i]:
                if node.type == 'Slack':
                    table.add_row([f'DC Grid {i+1}', node.name])

        print('--------------')
        print('Slack nodes')
        print(table)

    def Slack_AC(self):
        table = pt()
        # Define the table headers
        table.field_names = ["Grid", "Slack node"]

        for i in range(self.Grid.Num_Grids_AC):
            for node in self.Grid.Grids_AC[i]:
                if node.type == 'Slack':
                    table.add_row([f'AC Grid {i+1}', node.name])

        print('--------------')
        print('Slack nodes')
        print(table)

    def Slack_DC(self):
        table = pt()
        # Define the table headers
        table.field_names = ["Grid", "Slack node"]
        slack = 0
        for i in range(self.Grid.Num_Grids_DC):
            for node in self.Grid.Grids_DC[i]:
                if node.type == 'Slack':
                    table.add_row([f'DC Grid {i+1}', node.name])
                    slack += 1

        print('--------------')
        print('Slack nodes')
        if slack == 0:
            print("No DC nodes are set as Slack")
        else:
            print(table)


    def Power_loss(self):
        table = pt()
        # Define the table headers
        table.field_names = ["Grid", "Power Loss (MW)","Load %"]
        generation=0 
        grid_loads = 0
        tot=0
        
        if self.Grid.nodes_AC != []:
            if self.Grid.OPF_run:
                P_AC = np.vstack([node.PGi+sum(rs.PGi_ren*rs.gamma for rs in node.connected_RenSource)
                                        +sum(gen.PGen for gen in node.connected_gen if gen.PGen >0) for node in self.Grid.nodes_AC])
                Q_AC = np.vstack([node.QGi+sum(gen.QGen for gen in node.connected_gen) for node in self.Grid.nodes_AC])
            else:
                P_AC = np.vstack([node.PGi+sum(rs.PGi_ren*rs.gamma for rs in node.connected_RenSource)
                                        +sum(gen.Pset for gen in node.connected_gen) for node in self.Grid.nodes_AC])
                Q_AC = np.vstack([node.QGi+sum(gen.Qset for gen in node.connected_gen) for node in self.Grid.nodes_AC])
            
            for node in self.Grid.nodes_AC:
                if not self.Grid.OPF_run and node.type == 'Slack':
                      PGi = node.P_INJ-(node.P_s.item())+node.PLi
                else:
                      PGi = P_AC[node.nodeNumber].item()
                generation += PGi*self.Grid.S_base      
                grid_loads += (node.PLi-sum(gen.PGen for gen in node.connected_gen if gen.PGen <0))*self.Grid.S_base

            self.lossP_AC = np.zeros(self.Grid.Num_Grids_AC)
            
            for line in self.Grid.lines_AC:
                node = line.fromNode
                G = self.Grid.Graph_node_to_Grid_index_AC[node.nodeNumber]
                Ploss = np.real(line.loss)*self.Grid.S_base
            
                Sfrom = abs(line.fromS)*self.Grid.S_base
                Sto   = abs(line.toS)*self.Grid.S_base

                load = max(Sfrom, Sto)
                
                self.Grid.load_grid_AC[G] += load
                
                self.lossP_AC[G] += Ploss

            
            for line in self.Grid.lines_AC_exp:
                if line.np_line>0.01:
                    node = line.fromNode
                    G = self.Grid.Graph_node_to_Grid_index_AC[node.nodeNumber]
                    Ploss = np.real(line.loss)*self.Grid.S_base
                    
                    Sfrom = abs(line.fromS)*self.Grid.S_base
                    Sto   = abs(line.toS)*self.Grid.S_base
    
                    load = max(Sfrom, Sto)
                    
                    self.Grid.load_grid_AC[G] += load
                    
                    self.lossP_AC[G] += Ploss

            for line in (self.Grid.lines_AC_rec + self.Grid.lines_AC_ct):
                node = line.fromNode
                G = self.Grid.Graph_node_to_Grid_index_AC[node.nodeNumber]
                Ploss = np.real(line.loss)*self.Grid.S_base
                
                Sfrom = abs(line.fromS)*self.Grid.S_base
                Sto   = abs(line.toS)*self.Grid.S_base

                load = max(Sfrom, Sto)
                
                self.Grid.load_grid_AC[G] += load
                
                self.lossP_AC[G] += Ploss

           
            
            for g in range(self.Grid.Num_Grids_AC):
                if self.Grid.rating_grid_AC[g]!=0:
                    gload=self.Grid.load_grid_AC[g]/self.Grid.rating_grid_AC[g]*100
                else:
                    gload=0
                table.add_row([f'AC Grid {g+1}', np.round(self.lossP_AC[g], decimals=self.dec),np.round(gload, decimals=self.dec)])
                tot += self.lossP_AC[g]

        if self.Grid.nodes_DC != []:
            for node in self.Grid.nodes_DC:
                generation+= (node.PGi
                              +sum(rs.PGi_ren*rs.gamma for rs in node.connected_RenSource)
                              +sum(gen.PGen for gen in node.connected_gen if gen.PGen >0))*self.Grid.S_base
                grid_loads += node.PLi*self.Grid.S_base


            self.lossP_DC = np.zeros(self.Grid.Num_Grids_DC)

            for line in self.Grid.lines_DC:
                node = line.fromNode
                G = self.Grid.Graph_node_to_Grid_index_DC[node.nodeNumber]

                Ploss = np.real(line.loss)*self.Grid.S_base
                
                self.lossP_DC[G] += Ploss
                
                       
                i = line.fromNode.nodeNumber
                j = line.toNode.nodeNumber
                p_to = self.Grid.Pij_DC[j, i]*self.Grid.S_base
                p_from = self.Grid.Pij_DC[i, j]*self.Grid.S_base

                load = max(p_to, p_from)
                
                self.Grid.load_grid_DC[G] += load
                
                
            for g in range(self.Grid.Num_Grids_DC):
                gload=self.Grid.load_grid_DC[g]/self.Grid.rating_grid_DC[g]*100
                table.add_row([f'DC Grid {g+1}', np.round(self.lossP_DC[g], decimals=self.dec),np.round(gload, decimals=self.dec)])
                tot += self.lossP_DC[g]

        if self.Grid.Converters_ACDC != []:
            P_loss_ACDC = 0
            for conv in self.Grid.Converters_ACDC:
                P_loss_ACDC += (conv.P_loss_tf+conv.P_loss)*self.Grid.S_base
                tot += (conv.P_loss_tf+conv.P_loss)*self.Grid.S_base
         
            table.add_row(['AC DC Converters', np.round(P_loss_ACDC, decimals=self.dec),""])


        eff = grid_loads/generation*100
        
        table.add_row(["Total loss", np.round(tot, decimals=self.dec),""])
        table.add_row(["     ", "",""])
        table.add_row(["Generation", np.round(generation, decimals=self.dec),""])
        table.add_row(["Load", np.round(grid_loads, decimals=self.dec),""])
        table.add_row(["Efficiency", f'{np.round(eff, decimals=0)}%',""])
        print('--------------')
        print('Power loss')
        print(table)

    def Power_loss_AC(self):
        table = pt()
        # Define the table headers
        table.field_names = ["Grid", "Power Loss (MW)"]
        self.lossP_AC = np.zeros(self.Grid.Num_Grids_AC)
        for line in self.Grid.lines_AC:
            node = line.fromNode
            G = self.Grid.Graph_node_to_Grid_index_AC[node.nodeNumber]
            Ploss = np.real(line.loss)*self.Grid.S_base
            
            self.lossP_AC[G] += Ploss

        tot = 0
        for g in range(self.Grid.Num_Grids_AC):
            table.add_row(
                [f'AC Grid {g+1}', np.round(self.lossP_AC[g], decimals=self.dec)])
            tot += self.lossP_AC[g]

        table.add_row(["Total loss", np.round(tot, decimals=self.dec)])
        print('--------------')
        print('Power loss AC')
        print(table)

    def Power_loss_DC(self):
        table = pt()
        # Define the table headers
        table.field_names = ["Grid", "Power Loss (MW)"]

        self.lossP_DC = np.zeros(self.Grid.Num_Grids_DC)

        for line in self.Grid.lines_DC:
            node = line.fromNode
            G = self.Grid.Graph_node_to_Grid_index_DC[node.nodeNumber]

            Ploss = np.real(line.loss)*self.Grid.S_base

            self.lossP_DC[G] += Ploss
        tot = 0

        for g in range(self.Grid.Num_Grids_DC):
            table.add_row(
                [f'DC Grid {g+1}', np.round(self.lossP_DC[g], decimals=self.dec)])
            tot += self.lossP_DC[g]

        table.add_row(["Total loss", np.round(tot, decimals=self.dec)])
        print('--------------')
        print('Power loss DC')
        print(table)

    def DC_bus(self):

        if self.Grid.OPF_run:
            P_DC = np.vstack([node.PGi+sum(rs.PGi_ren*rs.gamma for rs in node.connected_RenSource)
                                    +sum(gen.PGen for gen in node.connected_gen) for node in self.Grid.nodes_DC])
        else:
            P_DC = np.vstack([node.PGi+sum(rs.PGi_ren*rs.gamma for rs in node.connected_RenSource)
                                    +sum(gen.Pset for gen in node.connected_gen) for node in self.Grid.nodes_DC])
        print('--------------')
        print('Results DC')
        print('')
        table_all = pt()
        table_all.field_names = [
            "Node", "Power Gen (MW)", "Power Load (MW)", "Power Converter ACDC (MW)", "Power Converter DCDC (MW)",
            "Power injected (MW)", "Voltage (pu)", "Grid"]  # 7 fields

        for g in range(self.Grid.Num_Grids_DC):
            print(f'Grid DC {g+1}')

            table = pt()

            # Define the table headers
            table.field_names = [
                "Node", "Power Gen (MW)", "Power Load (MW)", "Power Converter ACDC (MW)", 
                "Power Converter DCDC (MW)", "Power injected (MW)", "Voltage (pu)"]  # 7 fields

            for node in self.Grid.nodes_DC:
                if self.Grid.Graph_node_to_Grid_index_DC[node.nodeNumber] == g:
                    if not self.Grid.OPF_run:
                        if node.type == 'Slack':
                            if self.Grid.nconv == 0:
                                if node.P_INJ > 0:
                                    node.PGi = node.P_INJ
                                else:
                                    node.PLi = abs(node.P_INJ)
                    conv  = np.round(node.Pconv*self.Grid.S_base, decimals=self.dec)
                    table.add_row([
                        node.name, 
                        np.round(P_DC[node.nodeNumber].item()*self.Grid.S_base, decimals=self.dec), 
                        np.round(node.PLi*self.Grid.S_base, decimals=self.dec), 
                        conv,
                        np.round(node.PconvDC*self.Grid.S_base, decimals=self.dec),
                        np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec), 
                        np.round(node.V, decimals=self.dec)
                    ])
                    table_all.add_row([
                        node.name, 
                        np.round(P_DC[node.nodeNumber].item()*self.Grid.S_base, decimals=self.dec), 
                        np.round(node.PLi*self.Grid.S_base, decimals=self.dec), 
                        conv,
                        np.round(node.PconvDC*self.Grid.S_base, decimals=self.dec),
                        np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec), 
                        np.round(node.V, decimals=self.dec),
                        g+1
                    ])

            print(table)

        if self.export is not None:
            csv_filename = f'{self.export}/DC_bus.csv'
            csv_data = table_all.get_csv_string()

            with open(csv_filename, 'w', newline='') as csvfile:
                csvfile.write(csv_data)

    def AC_Powerflow(self, Grid=None):
        print('--------------')
        print('Results AC power')
        print('')
        table_all = pt()

        if self.Grid.OPF_run:
            P_AC = np.vstack([node.PGi+sum(rs.PGi_ren*rs.gamma for rs in node.connected_RenSource)
                                    +sum(gen.PGen for gen in node.connected_gen) for node in self.Grid.nodes_AC])
            Q_AC = np.vstack([node.QGi+sum(gen.QGen for gen in node.connected_gen) for node in self.Grid.nodes_AC])
        else:
            P_AC = np.vstack([node.PGi+sum(rs.PGi_ren*rs.gamma for rs in node.connected_RenSource)
                                    +sum(gen.Pset for gen in node.connected_gen) for node in self.Grid.nodes_AC])
            Q_AC = np.vstack([node.QGi+sum(gen.Qset for gen in node.connected_gen) for node in self.Grid.nodes_AC])

        if self.Grid.nodes_DC == None:
            table_all.field_names = ["Node", "Power Gen (MW)", "Reactive Gen (MVAR)", "Power Load (MW)",
                                     "Reactive Load (MVAR)", "Power injected  (MW)", "Reactive injected  (MVAR)", "Grid"]
        else:
            table_all.field_names = ["Node", "Power Gen (MW)", "Reactive Gen (MVAR)", "Power Load (MW)", "Reactive Load (MVAR)",
                                     "Power converters DC(MW)", "Reactive converters DC (MVAR)", "Power injected  (MW)", "Reactive injected  (MVAR)", "Grid"]

        for g in range(self.Grid.Num_Grids_AC):
            if Grid == (g+1):
                print(f'Grid AC {g+1}')
                table = pt()
                if self.Grid.nodes_DC == None:
                    # Define the table headers
                    table.field_names = ["Node", "Power Gen (MW)", "Reactive Gen (MVAR)", "Power Load (MW)",
                                         "Reactive Load (MVAR)", "Power injected  (MW)", "Reactive injected  (MVAR)"]

                    for node in self.Grid.nodes_AC:
                        if self.Grid.Graph_node_to_Grid_index_AC[node.nodeNumber] == g:
                            PGi = P_AC[node.nodeNumber].item()
                            QGi = Q_AC[node.nodeNumber].item()
                            
                            if not self.Grid.OPF_run:
                                if node.type == 'Slack':
                                    PGi = node.P_INJ-(node.P_s)+node.PLi
                                    QGi = node.Q_INJ-(node.Q_s+node.Q_s_fx)+node.QLi

                                if node.type == 'PV':
                                    node.QGi = node.Q_INJ-(node.Q_s+node.Q_s_fx)+node.QLi

                            table.add_row([node.name, 
                                           np.round(PGi*self.Grid.S_base, 
                                            decimals=self.dec), 
                                            np.round(QGi*self.Grid.S_base, decimals=self.dec), 
                                            np.round(node.PLi*self.Grid.S_base, decimals=self.dec), 
                                            np.round(node.QLi*self.Grid.S_base, decimals=self.dec), 
                                            np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec), 
                                            np.round(node.Q_INJ*self.Grid.S_base, decimals=self.dec)])

                else:
                    # Define the table headers
                    table.field_names = ["Node", "Power Gen (MW)", "Reactive Gen (MVAR)", "Power Load (MW)", "Reactive Load (MVAR)",
                                         "Power converters DC(MW)", "Reactive converters DC (MVAR)", "Power injected  (MW)",
                                         "Reactive injected  (MVAR)"]

                    for node in self.Grid.nodes_AC:
                        if self.Grid.Graph_node_to_Grid_index_AC[node.nodeNumber] == g:
                            
                            PGi = P_AC[node.nodeNumber].item()
                            QGi = Q_AC[node.nodeNumber].item()
                            if not self.Grid.OPF_run:
                                if node.type == 'Slack':
                                    PGi = (node.P_INJ-node.P_s + node.PLi).item()
                                    QGi = node.Q_INJ-node.Q_s-node.Q_s_fx+node.QLi

                                if node.type == 'PV':
                                    QGi = node.Q_INJ -(node.Q_s+node.Q_s_fx)+node.QLi

                            table.add_row([
                                node.name,
                                np.round(PGi*self.Grid.S_base, decimals=self.dec),
                                np.round(QGi*self.Grid.S_base, decimals=self.dec),
                                np.round(node.PLi*self.Grid.S_base, decimals=self.dec),
                                np.round(node.QLi*self.Grid.S_base, decimals=self.dec),
                                np.round(node.P_s*self.Grid.S_base, decimals=self.dec).item(),
                                np.round((node.Q_s+node.Q_s_fx)*self.Grid.S_base, decimals=self.dec).item(),
                                np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec),
                                np.round(node.Q_INJ*self.Grid.S_base, decimals=self.dec)
                            ])
                            
                            table_all.add_row([
                                node.name,
                                np.round(PGi*self.Grid.S_base, decimals=self.dec),
                                np.round(QGi*self.Grid.S_base, decimals=self.dec),
                                np.round(node.PLi*self.Grid.S_base, decimals=self.dec),
                                np.round(node.QLi*self.Grid.S_base, decimals=self.dec),
                                np.round(node.P_s*self.Grid.S_base, decimals=self.dec).item(),
                                np.round((node.Q_s+node.Q_s_fx)*self.Grid.S_base, decimals=self.dec).item(), 
                                np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec), 
                                np.round(node.Q_INJ*self.Grid.S_base, decimals=self.dec),
                                g+1
                            ])
                          
                print(table)

            elif Grid == None:
                print(f'Grid AC {g+1}')
                table = pt()
                if self.Grid.nodes_DC == None:
                    # Define the table headers
                    table.field_names = ["Node", "Power Gen (MW)", "Reactive Gen (MVAR)", "Power Load (MW)",
                                         "Reactive Load (MVAR)", "Power injected  (MW)", "Reactive injected  (MVAR)"]

                    for node in self.Grid.nodes_AC:
                        if self.Grid.Graph_node_to_Grid_index_AC[node.nodeNumber] == g:
                            PGi = P_AC[node.nodeNumber].item()
                            QGi = Q_AC[node.nodeNumber].item()
                            if not self.Grid.OPF_run:
                                if node.type == 'Slack':
                                    PGi = node.P_INJ-(node.P_s)+node.PLi
                                    QGi = node.Q_INJ-(node.Q_s+node.Q_s_fx)+node.QLi

                                if node.type == 'PV':
                                    node.QGi = node.Q_INJ-(node.Q_s+node.Q_s_fx)+node.QLi
                                    
                            table.add_row([node.name, np.round(PGi*self.Grid.S_base, decimals=self.dec), np.round(QGi*self.Grid.S_base, decimals=self.dec), np.round(node.PLi*self.Grid.S_base, decimals=self.dec), np.round(
                                node.QLi*self.Grid.S_base, decimals=self.dec), np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec), np.round(node.Q_INJ*self.Grid.S_base, decimals=self.dec)])
                            table_all.add_row([node.name, np.round(PGi*self.Grid.S_base, decimals=self.dec), np.round(QGi*self.Grid.S_base, decimals=self.dec), np.round(node.PLi*self.Grid.S_base, decimals=self.dec), np.round(
                                node.QLi*self.Grid.S_base, decimals=self.dec), np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec), np.round(node.Q_INJ*self.Grid.S_base, decimals=self.dec), g+1])

                else:
                    # Define the table headers
                    table.field_names = ["Node", "Power Gen (MW)", "Reactive Gen (MVAR)", "Power Load (MW)", "Reactive Load (MVAR)",
                                         "Power converters DC(MW)", "Reactive converters DC (MVAR)", "Power injected  (MW)",
                                         "Reactive injected  (MVAR)"]

                    for node in self.Grid.nodes_AC:
                        if self.Grid.Graph_node_to_Grid_index_AC[node.nodeNumber] == g:
                            
                            PGi = P_AC[node.nodeNumber].item()
                            QGi = Q_AC[node.nodeNumber].item()
                            if not self.Grid.OPF_run:
                                if node.type == 'Slack':
                                    PGi = (node.P_INJ-node.P_s + node.PLi).item()
                                    QGi = node.Q_INJ-node.Q_s-node.Q_s_fx+node.QLi

                                if node.type == 'PV':
                                    QGi = node.Q_INJ -(node.Q_s+node.Q_s_fx)+node.QLi

                            table.add_row([node.name, 
                                           np.round(PGi*self.Grid.S_base, decimals=self.dec), 
                                           np.round(QGi*self.Grid.S_base, decimals=self.dec), 
                                           np.round(node.PLi*self.Grid.S_base, decimals=self.dec), 
                                           np.round(node.QLi*self.Grid.S_base, decimals=self.dec), 
                                           np.round(node.P_s*self.Grid.S_base, decimals=self.dec).item(),
                                           np.round((node.Q_s+node.Q_s_fx)*self.Grid.S_base, decimals=self.dec).item(), 
                                           np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec),
                                           np.round(node.Q_INJ*self.Grid.S_base, decimals=self.dec)])
                            table_all.add_row([node.name,
                                               np.round(PGi*self.Grid.S_base, decimals=self.dec), 
                                               np.round(QGi*self.Grid.S_base, decimals=self.dec), 
                                               np.round(node.PLi*self.Grid.S_base, decimals=self.dec), 
                                               np.round(node.QLi*self.Grid.S_base, decimals=self.dec), 
                                               np.round(node.P_s*self.Grid.S_base, decimals=self.dec).item(), 
                                               np.round((node.Q_s+node.Q_s_fx)*self.Grid.S_base, decimals=self.dec).item(), 
                                               np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec), 
                                               np.round(node.Q_INJ*self.Grid.S_base, decimals=self.dec), 
                                               g+1])

                print(table)
        if self.export is not None:
            csv_filename = f'{self.export}/AC_Powerflow.csv'
            csv_data = table_all.get_csv_string()

            with open(csv_filename, 'w', newline='') as csvfile:
                csvfile.write(csv_data)

    def AC_voltage(self):
        print('--------------')
        print('Results AC bus voltage')
        print('')
        table_all = pt()
        table_all.field_names = [
            "Bus", "Voltage (pu)", "Voltage angle (deg)", "Grid"]

        for g in range(self.Grid.Num_Grids_AC):
            print(f'Grid AC {g+1}')
            table = pt()

            table.field_names = ["Bus", "Voltage (pu)", "Voltage angle (deg)"]

            for node in self.Grid.nodes_AC:
                if self.Grid.Graph_node_to_Grid_index_AC[node.nodeNumber] == g:
                    table.add_row([node.name, np.round(node.V, decimals=self.dec), np.round(
                        np.degrees(node.theta), decimals=self.dec)])
                    table_all.add_row([node.name, np.round(node.V, decimals=self.dec), np.round(
                        np.degrees(node.theta), decimals=self.dec), g+1])

            if len(table.rows) > 0:  # Check if the table is not None and has at least one row
                print(table)

        if self.export is not None:
            csv_filename = f'{self.export}/AC_voltage.csv'
            csv_data = table_all.get_csv_string()

            with open(csv_filename, 'w', newline='') as csvfile:
                csvfile.write(csv_data)

    def AC_lines_current(self):
        
        print('--------------')
        print('Results AC Lines Currents')
        table_all = pt()
        table_all.field_names = ["Line", "From bus", "To bus",
                                 "i from (kA)", "i to (kA)", "Loading %","Capacity [MVA]", "Grid"]
        for g in range(self.Grid.Num_Grids_AC):
            print(f'Grid AC {g+1}')
            tablei = pt()
            tablei.field_names = ["Line", "From bus", "To bus",
                                  "i from (kA)", "i to (kA)", "Loading %","Capacity [MVA]"]

            for line in self.Grid.lines_AC:
                if self.Grid.Graph_line_to_Grid_index_AC[line] == g:
                    i = line.fromNode.nodeNumber
                    j = line.toNode.nodeNumber
                    I_base = self.Grid.S_base/line.kV_base

                    i_from = line.i_from*I_base/np.sqrt(3)

                    i_to = line.i_to*I_base/np.sqrt(3)
                    
                    Sfrom = abs(line.fromS)*self.Grid.S_base
                    Sto   = abs(line.toS)*self.Grid.S_base

                    load = max(Sfrom, Sto)/line.MVA_rating*100
                    if line.name == 'ICL4':
                        s=1
                    tablei.add_row([
                        line.name,
                        line.fromNode.name,
                        line.toNode.name,
                        np.round(i_from, decimals=self.dec),
                        np.round(i_to, decimals=self.dec),
                        np.round(load, decimals=self.dec),
                        int(line.MVA_rating)
                    ])
                    
                    table_all.add_row([
                        line.name,
                        line.fromNode.name,
                        line.toNode.name,
                        np.round(i_from, decimals=self.dec),
                        np.round(i_to, decimals=self.dec),
                        np.round(load, decimals=self.dec),
                        int(line.MVA_rating),
                        g+1
                    ])
            if len(tablei.rows) > 0:  # Check if the table is not None and has at least one row
                print(tablei)

        if self.export is not None:
            csv_filename = f'{self.export}/AC_line_current.csv'
            csv_data = table_all.get_csv_string()

            with open(csv_filename, 'w', newline='') as csvfile:
                csvfile.write(csv_data)

    def AC_exp_lines_power(self):

        print('--------------')
        print('Results AC Expansion Lines power')
        

        tablep = pt()
        tablep.field_names = ["Line", "From bus", "To bus",
                                "P from (MW)", "Q from (MVAR)", "P to (MW)", "Q to (MW)", "Power loss (MW)", "Q loss (MVAR)","Loading %"]
        for g in range(self.Grid.Num_Grids_AC):
            print(f'Grid AC {g+1}')
            for line in self.Grid.lines_AC_exp:
               if line.np_line>0.01:  
                    if self.Grid.Graph_line_to_Grid_index_AC[line] == g:
                        i = line.fromNode.nodeNumber
                        j = line.toNode.nodeNumber
                        
                        
                        p_from = np.real(line.fromS)*self.Grid.S_base
                        Q_from = np.imag(line.fromS)*self.Grid.S_base
    
                        p_to = np.real(line.toS)*self.Grid.S_base
                        Q_to = np.imag(line.toS)*self.Grid.S_base
    
                        Ploss = np.real(line.loss)*self.Grid.S_base
                        Qloss = np.imag(line.loss)*self.Grid.S_base

                        Sfrom = abs(line.fromS)*self.Grid.S_base
                        Sto   = abs(line.toS)*self.Grid.S_base

                        load = max(Sfrom, Sto)/(line.MVA_rating*line.np_line)*100

                        tablep.add_row([
                            line.name, 
                            line.fromNode.name, 
                            line.toNode.name, 
                            np.round(p_from, decimals=self.dec), 
                            np.round(Q_from, decimals=self.dec), 
                            np.round(p_to, decimals=self.dec), 
                            np.round(Q_to, decimals=self.dec), 
                            np.round(Ploss, decimals=self.dec), 
                            np.round(Qloss, decimals=self.dec),
                            np.round(load, decimals=self.dec)
                        ])

            for line in self.Grid.lines_AC_rec:
                 if self.Grid.Graph_line_to_Grid_index_AC[line] == g:
                        i = line.fromNode.nodeNumber
                        j = line.toNode.nodeNumber
                        
                        
                        p_from = np.real(line.fromS)*self.Grid.S_base
                        Q_from = np.imag(line.fromS)*self.Grid.S_base
    
                        p_to = np.real(line.toS)*self.Grid.S_base
                        Q_to = np.imag(line.toS)*self.Grid.S_base
    
                        Ploss = np.real(line.loss)*self.Grid.S_base
                        Qloss = np.imag(line.loss)*self.Grid.S_base

                        Sfrom = abs(line.fromS)*self.Grid.S_base
                        Sto   = abs(line.toS)*self.Grid.S_base

                        if line.rec_branch:
                            load = max(Sfrom, Sto)/(line.MVA_rating_new)*100
                        else:
                            load = max(Sfrom, Sto)/(line.MVA_rating)*100

                        tablep.add_row([
                            line.name, 
                            line.fromNode.name, 
                            line.toNode.name, 
                            np.round(p_from, decimals=self.dec), 
                            np.round(Q_from, decimals=self.dec), 
                            np.round(p_to, decimals=self.dec), 
                            np.round(Q_to, decimals=self.dec), 
                            np.round(Ploss, decimals=self.dec), 
                            np.round(Qloss, decimals=self.dec),
                            np.round(load, decimals=self.dec)
                        ])

            for line in self.Grid.lines_AC_ct:
                 if self.Grid.Graph_line_to_Grid_index_AC[line] == g:
                        i = line.fromNode.nodeNumber
                        j = line.toNode.nodeNumber
                        
                        
                        p_from = np.real(line.fromS)*self.Grid.S_base
                        Q_from = np.imag(line.fromS)*self.Grid.S_base
    
                        p_to = np.real(line.toS)*self.Grid.S_base
                        Q_to = np.imag(line.toS)*self.Grid.S_base
    
                        Ploss = np.real(line.loss)*self.Grid.S_base
                        Qloss = np.imag(line.loss)*self.Grid.S_base

                        Sfrom = abs(line.fromS)*self.Grid.S_base
                        Sto   = abs(line.toS)*self.Grid.S_base

                        chosen_line = line.active_config
                        load = max(Sfrom, Sto)/(line.MVA_rating_list[chosen_line])*100


                        tablep.add_row([
                            line.name, 
                            line.fromNode.name, 
                            line.toNode.name, 
                            np.round(p_from, decimals=self.dec), 
                            np.round(Q_from, decimals=self.dec), 
                            np.round(p_to, decimals=self.dec), 
                            np.round(Q_to, decimals=self.dec), 
                            np.round(Ploss, decimals=self.dec), 
                            np.round(Qloss, decimals=self.dec),
                            np.round(load, decimals=self.dec)
                        ])

            if len(tablep.rows) > 0:  # Check if the table is not None and has at least one row
                print(tablep)


    def AC_lines_power(self, Grid=None):
        
        print('--------------')
        print('Results AC Lines power')
        table_all = pt()
        table_all.field_names = ["Line", "From bus", "To bus",
                                 "P from (MW)", "Q from (MVAR)", "P to (MW)", "Q to (MW)", "Power loss (MW)", "Q loss (MVAR)", "Grid"]

        for g in range(self.Grid.Num_Grids_AC):
            if Grid == (g+1):
                print(f'Grid AC {g+1}')

                tablep = pt()
                tablep.field_names = ["Line", "From bus", "To bus",
                                      "P from (MW)", "Q from (MVAR)", "P to (MW)", "Q to (MW)", "Power loss (MW)", "Q loss (MVAR)"]

                for line in self.Grid.lines_AC:
                    if self.Grid.Graph_line_to_Grid_index_AC[line] == g:
                        i = line.fromNode.nodeNumber
                        j = line.toNode.nodeNumber
                        p_from = np.real(line.fromS)*self.Grid.S_base
                        Q_from = np.imag(line.fromS)*self.Grid.S_base

                        p_to = np.real(line.toS)*self.Grid.S_base
                        Q_to = np.imag(line.toS)*self.Grid.S_base

                        Ploss = np.real(line.loss)*self.Grid.S_base
                        Qloss = np.imag(line.loss)*self.Grid.S_base

                        tablep.add_row([
                            line.name, 
                            line.fromNode.name, 
                            line.toNode.name, 
                            np.round(p_from, decimals=self.dec), 
                            np.round(Q_from, decimals=self.dec), 
                            np.round(p_to, decimals=self.dec), 
                            np.round(Q_to, decimals=self.dec), 
                            np.round(Ploss, decimals=self.dec), 
                            np.round(Qloss, decimals=self.dec)
                        ])

            elif Grid == None:

                print(f'Grid AC {g+1}')

                tablep = pt()
                tablep.field_names = ["Line", "From bus", "To bus",
                                      "P from (MW)", "Q from (MVAR)", "P to (MW)", "Q to (MW)", "Power loss (MW)", "Q loss (MVAR)"]

                for line in self.Grid.lines_AC:
                    if self.Grid.Graph_line_to_Grid_index_AC[line] == g:
                        i = line.fromNode.nodeNumber
                        j = line.toNode.nodeNumber
                        
                        p_from = np.real(line.fromS)*self.Grid.S_base
                        Q_from = np.imag(line.fromS)*self.Grid.S_base

                        p_to = np.real(line.toS)*self.Grid.S_base
                        Q_to = np.imag(line.toS)*self.Grid.S_base

                        Ploss = np.real(line.loss)*self.Grid.S_base
                        Qloss = np.imag(line.loss)*self.Grid.S_base

                        tablep.add_row([
                            line.name, 
                            line.fromNode.name, 
                            line.toNode.name, 
                            np.round(p_from, decimals=self.dec), 
                            np.round(Q_from, decimals=self.dec), 
                            np.round(p_to, decimals=self.dec), 
                            np.round(Q_to, decimals=self.dec), 
                            np.round(Ploss, decimals=self.dec), 
                            np.round(Qloss, decimals=self.dec)
                        ])
                        table_all.add_row([line.name, line.fromNode.name, line.toNode.name, np.round(p_from, decimals=self.dec), np.round(Q_from, decimals=self.dec), np.round(
                            p_to, decimals=self.dec), np.round(Q_to, decimals=self.dec), np.round(Ploss, decimals=self.dec), np.round(Qloss, decimals=self.dec), g+1])

                if len(tablep.rows) > 0:  # Check if the table is not None and has at least one row
                    print(tablep)

        if self.export is not None:
            csv_filename = f'{self.export}/AC_line_power.csv'
            csv_data = table_all.get_csv_string()

            with open(csv_filename, 'w', newline='') as csvfile:
                csvfile.write(csv_data)
    def Ext_gen(self):
        print('--------------')
        print('External Generation optimization')
        table = pt()
        Ptot=0
        Qtot=0
        Pabs=0
        Qabs=0
        Stot=0
        Ltot=0
        costtot=0
        table.field_names = ["Generator","Node" ,"Power (MW)", "Reactive power (MVAR)","Quadratic Price €/MWh^2","Linear Price €/MWh","Fixed Cost €","Loading %","Cost k€"]
        for gen in self.Grid.Generators:
          if gen.np_gen>0.001:  
            Pgi=gen.PGen #+node.PGi_ren*node.curtailment
            Qgi=gen.QGen
            S= np.sqrt(Pgi**2+Qgi**2)
            Pgi*=self.Grid.S_base
            Qgi*=self.Grid.S_base
            if gen.Max_S is None:
                base=np.sqrt(gen.Max_pow_gen**2+max(abs(gen.Min_pow_genR),gen.Max_pow_genR)**2)
            else:
                base=gen.Max_S
            base *= gen.np_gen
            load=S/base*100
            fc=gen.fc*gen.np_gen
            cost=(Pgi**2*gen.qf+Pgi*gen.lf+fc)/1000
           
                
            table.add_row([gen.name,gen.Node_AC, np.round(Pgi, decimals=self.dec), np.round(Qgi, decimals=self.dec),
                           np.round(gen.qf, decimals=self.dec),  np.round(gen.lf, decimals=self.dec),np.round(fc, decimals=self.dec),
                           np.round(load, decimals=self.dec), np.round(cost, decimals=0)])
            Pabs+=abs(Pgi)
            Qabs+=abs(Qgi)
            Ptot+=Pgi
            Qtot+=Qgi
            Stot+=S
            costtot+=cost
            Ltot+=base

        for gen in self.Grid.Generators_DC:
          if gen.np_gen>0.001:  
            Pgi=gen.PGen*self.Grid.S_base
            
            base=gen.Max_pow_gen*self.Grid.S_base
            base *= gen.np_gen
            load=Pgi/base*100
            fc=gen.fc*gen.np_gen
            cost=(Pgi**2*gen.qf+Pgi*gen.lf+fc)/1000
           
                
            table.add_row([gen.name,gen.Node_DC, np.round(Pgi, decimals=self.dec), "----",
                           np.round(gen.qf, decimals=self.dec),  np.round(gen.lf, decimals=self.dec),np.round(fc, decimals=self.dec),
                           np.round(load, decimals=self.dec), np.round(cost, decimals=0)])
            Pabs+=abs(Pgi)
            
            Ptot+=Pgi
            
            Stot+=Pgi
            costtot+=cost
            Ltot+=base

        if Ltot !=0:
            load=Stot/Ltot*100
        else:
            load=0
        table.add_row(['Total',"", np.round(Ptot, decimals=self.dec), np.round(Qtot, decimals=self.dec),"",""," ","", np.round(costtot, decimals=0)])
        table.add_row(['Total abs',"", np.round(Pabs, decimals=self.dec), np.round(Qabs, decimals=self.dec), "","","",np.round(load, decimals=self.dec),""])
        print(table)
    
    def Ext_REN(self):
        print('--------------')
        print('Renewable energy sources')
        table = pt()
        table.field_names = ["Bus", "Base Power (MW)", "Curtailment %","Power Injected (MW)","Reactive Power Injected (MVAR)","Price €/MWh","Cost k€","Curtailment Cost [k€]"]
        bp=0
        tcur=0
        totcost=0
        totcurcost=0
        price=0
        for rs in self.Grid.RenSources:
                Pgi=rs.PGi_ren*self.Grid.S_base
                bp+=Pgi
                cur= (1-rs.gamma)*100
                tcur+=Pgi*(1-rs.gamma)
                PGicur=Pgi*(rs.gamma)
                QGi=rs.QGi_ren*self.Grid.S_base
                
                if not self.Grid.OnlyGen or self.Grid.OPF_Price_Zones_constraints_used:
                   
                    if rs.connected == 'AC':
                        node_num = self.Grid.rs2node['AC'][rs.rsNumber]
                        node = self.Grid.nodes_AC[node_num]
                        price=node.price
                    else:
                        node_num = self.Grid.rs2node['DC'][rs.rsNumber]
                        node = self.Grid.nodes_DC[node_num]
                        price=node.price
                    cost=PGicur*price/1000
                else:
                    cost=0 
                if self.Grid.CurtCost==False:
                    curcost=0
                else:    
                    curcost= (Pgi-PGicur)*node.price*(self.Grid.sigma)/1000
                table.add_row([rs.name, np.round(Pgi, decimals=self.dec), np.round(cur, decimals=self.dec),  np.round(PGicur, decimals=self.dec),np.round(QGi, decimals=self.dec),np.round(price, decimals=self.dec),np.round(cost, decimals=0),np.round(curcost, decimals=0)])
                totcost+=cost
                totcurcost+=curcost
        
     
        
        PGicur=bp-tcur
        cur=(tcur)/bp*100
        
        table.add_row(['Total', np.round(bp, decimals=self.dec), np.round(cur, decimals=self.dec),  np.round(PGicur, decimals=self.dec), "","",np.round(totcost, decimals=0),np.round(totcurcost, decimals=0)])

        print(table)    
    
    def TEP_ts_res(self):
       
        curt_used,curt_n,PN,GEN, SC , curt, curt_per, lines,conv,price,pgen,qgen= self.Grid.TEP_res
        
        
        table = pt()
        # Add columns to the PrettyTable
        data = PN.fillna('')
        field_names = [''] + [f'Net price zone power [MW] @ Case:{t}' for t in data.columns]
        table.field_names = field_names
        
        # Add rows to the PrettyTable
        for index, row in data.iterrows():
            table.add_row([index] + row.tolist())
            
        print(table)
        
        table = pt()
        # Add columns to the PrettyTable
        data_SC = SC.fillna('')
        field_names = [''] + [f'Social Cost [k€] @ Case:{t}' for t in data_SC.columns]
        table.field_names = field_names
        
        # Add rows to the PrettyTable
        for index, row in data_SC.iterrows():
            table.add_row([index] + row.tolist())
            
        print(table)
        
        
        table = pt()
        # Add columns to the PrettyTable
        data_price = price.fillna('')
        field_names = [''] + [f'Price Zone Price [€/Mwh] @ Case:{t}' for t in data_price.columns]
        table.field_names = field_names
        
        # Add rows to the PrettyTable
        for index, row in data_price.iterrows():
            table.add_row([index] + row.tolist())
            
        print(table)
        
        
        table = pt()
        # Add columns to the PrettyTable
        data_curt = curt.fillna('')
        field_names = [''] + [f'Curtialment [MW] @ Case:{t}' for t in data_curt.columns]
        table.field_names = field_names
        
        # Add rows to the PrettyTable
        for index, row in data_curt.iterrows():
            table.add_row([index] + row.tolist())
            
        print(table)
        
        
        
        table = pt()
        # Add columns to the PrettyTable
        data_lines = lines.fillna('')
        field_names = [''] + [f'Line loading [%] @ Case:{t}' for t in data_lines.columns]
        table.field_names = field_names
        
        # Add rows to the PrettyTable
        for index, row in data_lines.iterrows():
            table.add_row([index] + row.tolist())
            
        print(table)
        
        
        table = pt()
        # Add columns to the PrettyTable
        data_conv = conv.fillna('')
        field_names = [''] + [f'Converter loading [%] @ Case:{t}' for t in data_conv.columns]
        table.field_names = field_names
        
        # Add rows to the PrettyTable
        for index, row in data_conv.iterrows():
            table.add_row([index] + row.tolist())
            
        print(table)
        
        
        
        
    
    def TEP_N(self):
        
        table = pt()
        table.field_names = ["Element","Type" ,"Initial", "Optimized N","Maximum","Optimized Power Rating [MW]","Expansion Cost [€]"]
        tot=0
        
        for l in self.Grid.lines_AC_exp:
            if l.np_line_opf:
                if (l.np_line-l.np_line_b)>0.01:
                    element= l.name
                    ini= l.np_line_b
                    opt=l.np_line
                    pr= opt*l.MVA_rating
                    cost=(opt-ini)*l.base_cost
                    tot+=cost
                    maxn=l.np_line_max
                    table.add_row([element, "AC Line" ,ini, np.round(opt, decimals=2),maxn,np.round(pr, decimals=0).astype(int), f"{cost:,.2f}".replace(',', ' ')])
        
        for l in self.Grid.lines_AC_rec:
            if l.rec_line_opf:
                if l.rec_branch:
                    element= l.name
                    ini= 0
                    opt= l.rec_branch
                    pr= l.MVA_rating_new
                    cost= l.base_cost
                    tot+=cost
                    table.add_row([element, "AC Upgrade" ,"", "","",np.round(pr, decimals=0).astype(int), f"{cost:,.2f}".replace(',', ' ')])
        for l in self.Grid.lines_AC_ct:
            if l.array_opf:
                element= l.name
                ini= l.cable_types[l.ini_active_config]
                max=l.cable_types[l.max_active_config]
                ct=l.active_config
                type=l.cable_types[ct]
                pr= l.MVA_rating
                cost= l.base_cost[ct]
                tot+=cost
                table.add_row([element, "AC CT" ,ini, type, max,np.round(pr, decimals=0).astype(int), f"{cost:,.2f}".replace(',', ' ')])

        for l in self.Grid.lines_DC:
            if l.np_line_opf:
                if (l.np_line-l.np_line_b)>0.01:
                    element= l.name
                    ini= l.np_line_b
                    opt=l.np_line
                    pr= opt*l.MW_rating
                    cost=(opt-ini)*l.base_cost
                    tot+=cost
                    maxn=l.np_line_max
                    table.add_row([element, "DC Line" ,ini, np.round(opt, decimals=2),maxn,np.round(pr, decimals=0).astype(int), f"{cost:,.2f}".replace(',', ' ')])
                
        
        for cn in self.Grid.Converters_ACDC:
            if cn.NUmConvP_opf:
                if (cn.NumConvP-cn.NumConvP_b)>0.01:
                    element= cn.name
                    ini=cn.NumConvP_b
                    opt=cn.NumConvP
                    pr=opt*cn.MVA_max
                    cost=(opt-ini)*cn.base_cost
                    tot+=cost
                    maxn=cn.NumConvP_max
                    table.add_row([element, "ACDC Conv" ,ini,np.round(opt, decimals=2),maxn,np.round(pr, decimals=0).astype(int), f"{cost:,.2f}".replace(',', ' ')])
        

        for gen in self.Grid.Generators:
            if gen.np_gen_opf:
                if (gen.np_gen-gen.np_gen_b)>0.01:
                    element= gen.name
                    ini= gen.np_gen_b
                    opt= gen.np_gen
                    if gen.Max_S is not None:
                        pr= gen.Max_S*gen.np_gen*self.Grid.S_base
                    elif gen.Max_pow_gen !=0:
                        pr= gen.Max_pow_gen*gen.np_gen*self.Grid.S_base
                    else:
                        pr=gen.Max_pow_genR*gen.np_gen*self.Grid.S_base
                    cost= (opt-ini)*gen.base_cost
                    tot+=cost
                    maxn=gen.np_gen_max
                    table.add_row([element, "Generator" ,ini,np.round(opt, decimals=2),maxn,np.round(pr, decimals=0).astype(int), f"{cost:,.2f}".replace(',', ' ')])

        table.add_row(["Total", "" ,"","", "", "",f"{tot:,.2f}".replace(',', ' ')])
        
        print('--------------')
        print('Transmission Expansion Problem')
        print(table)

    def TEP_norm(self):
        table=pt()
        table.field_names = ["Objective","Weight" ,"Value","Weighted Value","NPV"]
        
        weights = self.Grid.OPF_obj

        for key, value in weights.items():
            # if value['w'] !=0:
                table.add_row([
                    key, 
                    f"{value['w']:.2f}", 
                    f"{value['v']:,.2f}".replace(',', ' '),
                    f"{value['w']*value['v']:,.2f}".replace(',', ' '),
                    f"{value['NPV']:,.2f}".replace(',', ' ')
                ])
        
        print(table)


    def OBJ_res(self):
       
        table=pt()
        table.field_names = ["Objective","Weight" ,"Value","Weighted Value"]
        
        weights = self.Grid.OPF_obj

        for key, value in weights.items():
            # if value['w'] !=0:
                table.add_row([
                    key, 
                    f"{value['w']:.2f}", 
                    f"{value['v']:,.2f}".replace(',', ' '),
                    f"{value['w']*value['v']:,.2f}".replace(',', ' ')
                ])
        
        print(table)
        
    def TEP_TS_norm(self):

        tot = 0
        tot_n = 0

        for l in self.Grid.lines_AC_exp:
            if l.np_line_opf:
                
                opt=l.np_line
                cost=((opt)*l.MVA_rating*l.Length_km*l.phi)*l.life_time*8760/(10**6)
                tot+=cost
                tot_n+=((opt)*l.MVA_rating*l.Length_km*l.phi)/1000

        for l in self.Grid.lines_DC:
            if l.np_line_opf:
                opt=l.np_line
                cost=((opt)*l.MW_rating*l.Length_km*l.phi)*l.life_time*8760/(10**6)
                tot+=cost
                tot_n+=((opt)*l.MW_rating*l.Length_km*l.phi)/1000
                
        
        for cn in self.Grid.Converters_ACDC:
            if cn.NUmConvP_opf:
                opt=cn.NumConvP
                cost=((opt)*cn.MVA_max*cn.phi)*cn.life_time*8760/(10**6)
                tot+=cost
                tot_n+=((opt)*cn.MVA_max*cn.phi)/1000
        

        curt_used,curt_n,PN,GEN, SC , curt, curt_per, lines,conv,price,pgen,qgen= self.Grid.TEP_res
        weight = SC.loc['Weight']
        table=pt()
        table.field_names = ["Price_Zone", "Normalized Cost Generation[k€/h]", "Average price [€/MWh]","Present Value Cost Gen [M€]"]
        
        n_years = self.Grid.TEP_n_years
        discount_rate = self.Grid.TEP_discount_rate
        for m in self.Grid.Price_Zones:
            if type(m) is Price_Zone:
                price_zone_weighted = SC.loc[m.name]
                weighted_total = price_zone_weighted * weight
                weighted_total = weighted_total.sum()
                weighted_price = price.loc[m.name]* weight
                weighted_price = weighted_price.sum()
                present_value=0
                for year in range(1, n_years + 1):
                    # Discount each yearly cash flow and add to the present value
                    s=1
                    present_value += (weighted_total * 8760) / ((1 + discount_rate) ** year)/1000
                
                
                
                table.add_row([m.name, np.round(weighted_total, decimals=2),np.round(weighted_price, decimals=2),np.round(present_value, decimals=2)])
                
        print(table)
        
        
        table=pt()
        table.field_names = ["Normalized Cost Generation[k€/h]","Normalized investment [k€/h]","Normalized Total cost [k€/h]"]
        
        weighted_sum = SC.loc['Weighted SC'].sum()
        
        table.add_row([np.round(weighted_sum, decimals=2), np.round(tot_n, decimals=2),np.round(weighted_sum+tot_n, decimals=2)])
        print(table)
        
        
        table=pt()
        table.field_names = ["Present Value Cost Generation[M€]","Investment [M€]","NPV [M€]"]
        tot_pv=0
        for year in range(1, n_years + 1):
            # Discount each yearly cash flow and add to the present value
            s=1
            tot_pv += (weighted_sum * 8760) / ((1 + discount_rate) ** year)/1000
        table.add_row([
            f"{np.round(tot_pv, decimals=2):,}".replace(',', ' '),
            f"{np.round(tot, decimals=2):,}".replace(',', ' '),
            f"{-np.round(tot_pv + tot, decimals=2):,}".replace(',', ' ')
        ])  
        print(table)

    def MP_TEP_results(self):
        # Check if the attribute exists and is a DataFrame
        if hasattr(self.Grid, "MP_TEP_results") and isinstance(self.Grid.MP_TEP_results, pd.DataFrame):
            df = self.Grid.MP_TEP_results
            table = pt()
            table.field_names = list(df.columns)
            for row in df.itertuples(index=False):
                table.add_row(list(row))
            print(table)
        else:
            print(self.Grid.MP_TEP_results)
        
    def Price_Zone(self):
        print('--------------')
        print('Price_Zone')
        table = pt()
        table.field_names = ["Price_Zone","Renewable Generation(MW)" ,"Generation (MW)", "Load (MW)","Import (MW)","Export (MW)","Price (€/MWh)"]
        table2 = pt()
        table2.field_names = ["Price_Zone","Social Cost [k€]","Renewable Gen Cost [k€]","Curtailment Cost [k€]","Generation Cost [k€]","Total Cost [k€]"]
        
        tot_sc=0
        tot_Rgen_cost=0
        tot_gen_cost=0
        tot_curt_cost=0
        tot_m_tot=0
        
        
        for m in self.Grid.Price_Zones:
            
            Rgen = sum(rs.PGi_ren * rs.gamma for node in m.nodes_AC for rs in node.connected_RenSource) * self.Grid.S_base
            
            
            gen = sum(node.PGi+node.PGi_opt for node in m.nodes_AC)*self.Grid.S_base
            load = sum(node.PLi for node in m.nodes_AC)*self.Grid.S_base
            ie = Rgen+gen-load
            price=m.price
            
            
            
            sc = (m.a*ie**2+ie*m.b)/1000
            s=1
            if not self.Grid.OnlyGen or self.Grid.OPF_Price_Zones_constraints_used:
                Rgen_cost=Rgen*m.price/1000
            else:
                Rgen_cost= 0          
            gen_cost = gen*m.price/1000
            if self.Grid.CurtCost==False:
                curt_cost=0
            else:  
                curt_cost= sum((rs.PGi_ren-rs.PGi_ren * rs.gamma)*rs.sigma*node.price for node in m.nodes_AC for rs in node.connected_RenSource)*self.Grid.S_base
            m_tot= Rgen_cost+gen_cost+curt_cost+sc
            
            tot_sc+=sc
            tot_Rgen_cost+=Rgen_cost
            tot_gen_cost+=gen_cost
            tot_curt_cost+=curt_cost
            tot_m_tot+=m_tot
            
            if ie >=0:
                export = ie
                imp = 0
            else: 
                export = 0
                imp = abs(ie)
            table.add_row([m.name,  np.round(Rgen, decimals=self.dec),np.round(gen, decimals=self.dec), np.round(load, decimals=self.dec),  np.round(imp, decimals=self.dec),np.round(export, decimals=self.dec),np.round(price, decimals=2)])
            table2.add_row([m.name,  np.round(sc, decimals=self.dec),np.round(Rgen_cost, decimals=self.dec), np.round(curt_cost, decimals=self.dec),  np.round(gen_cost, decimals=self.dec),np.round(m_tot, decimals=self.dec)])
        
        
        if len(table.rows) > 0:  # Check if the table is not None and has at least one row
            table2.add_row(['Total',  np.round(tot_sc, decimals=self.dec),np.round(tot_Rgen_cost, decimals=self.dec), np.round(tot_curt_cost, decimals=self.dec),  np.round(tot_gen_cost, decimals=self.dec),np.round(tot_m_tot, decimals=self.dec)])
            print(table)
            print(table2)
            
            
    
    def DC_lines_current(self):
        
        print('--------------')
        print('Results DC Lines current')
        table_all = pt()
        table_all.field_names = [
            "Line", "From bus", "To bus", "I (kA)", "Loading %","Capacity [kA]" ,"Polarity", "Grid"]
        for g in range(self.Grid.Num_Grids_DC):
            print(f'Grid DC {g+1}')
            tablei = pt()

            tablei.field_names = ["Line", "From bus",
                                  "To bus", "I (kA)", "Loading %","Capacity [kA]", "Polarity"]
            tablei.align["Polarity"] = 'l'

            for line in self.Grid.lines_DC:
                if line.np_line<0.01:
                    continue
                if self.Grid.Graph_line_to_Grid_index_DC[line] == g:

                    i = line.fromNode.nodeNumber
                    j = line.toNode.nodeNumber
                    I_base = self.Grid.S_base/line.kV_base
                    i_to = self.Grid.Iij_DC[j, i]*I_base
                    i_from = self.Grid.Iij_DC[i, j]*I_base
                    line_current = max(abs(i_to),abs(i_from))
    
                    p_to = line.toP*self.Grid.S_base/line.np_line
                    p_from = line.fromP*self.Grid.S_base/line.np_line

                    load = max(p_to, p_from)/line.MW_rating*100

                    if line.m_sm_b == 'm':
                        pol = "Monopolar (asymmetrically grounded)"
                    elif line.m_sm_b == 'sm':
                        pol = "Monopolar (symmetrically grounded)"
                    elif line.m_sm_b == 'b':
                        pol = "Bipolar"

                    tablei.add_row([line.name, line.fromNode.name, line.toNode.name, np.round(
                        line_current, decimals=self.dec), np.round(load, decimals=self.dec),np.round(line.MW_rating*line.np_line/(line.kV_base*line.pol),decimals=self.dec) ,pol])
                    table_all.add_row([line.name, line.fromNode.name, line.toNode.name, np.round(
                        line_current, decimals=self.dec), np.round(load, decimals=self.dec),np.round(line.MW_rating*line.np_line/(line.kV_base*line.pol),decimals=self.dec), pol, g+1])

            if len(tablei.rows) > 0:  # Check if the table is not None and has at least one row
                print(tablei)

        if self.export is not None:
            csv_filename = f'{self.export}/DC_line_current.csv'
            csv_data = table_all.get_csv_string()

            with open(csv_filename, 'w', newline='') as csvfile:
                csvfile.write(csv_data)
                
    def DC_lines_power(self):
        
        print('--------------')
        print('Results DC Lines power')
        table_all = pt()
        table_all.field_names = ["Line", "From bus", "To bus",
                                 "P from (MW)", "P to (MW)", "Power loss (MW)", "Capacity [MW]","Grid"]
        for g in range(self.Grid.Num_Grids_DC):
            print(f'Grid DC {g+1}')
            tablep = pt()
            tablep.field_names = ["Line", "From bus", "To bus",
                                  "P from (MW)", "P to (MW)", "Power loss (MW)", "Capacity [MW]"]

            for line in self.Grid.lines_DC:
                if line.np_line <= 0.01:
                    continue
                
                if self.Grid.Graph_line_to_Grid_index_DC[line] == g:
                    
                   
                    p_to = line.toP*self.Grid.S_base
                    p_from = line.fromP*self.Grid.S_base

                    Ploss = np.real(line.loss)*self.Grid.S_base

                    tablep.add_row([line.name, line.fromNode.name, line.toNode.name, np.round(
                        p_from, decimals=self.dec), np.round(p_to, decimals=self.dec), np.round(Ploss, decimals=self.dec),int(line.MW_rating*line.np_line)])
                    table_all.add_row([line.name, line.fromNode.name, line.toNode.name, np.round(
                        p_from, decimals=self.dec), np.round(p_to, decimals=self.dec), np.round(Ploss, decimals=self.dec),int(line.MW_rating*line.np_line),g+1])

            if len(tablep.rows) > 0:  # Check if the table is not None and has at least one row
                print(tablep)

        if self.export is not None:
            csv_filename = f'{self.export}/DC_line_power.csv'
            csv_data = table_all.get_csv_string()

            with open(csv_filename, 'w', newline='') as csvfile:
                csvfile.write(csv_data)

    def DC_converter(self):
        table = pt()

        table.field_names = ["Converter", "From node", "To node",
                             "Power from (MW)", "Power To (MW))", "Power Loss (MW)"]
        for conv in self.Grid.Converters_DCDC:
            convid = conv.name
            fromnode = conv.fromNode.name
            tonode = conv.toNode.name
            fromMW = conv.Powerfrom*self.Grid.S_base
            toMW = conv.Powerto*self.Grid.S_base
            loss = np.abs(fromMW+toMW)

            table.add_row([convid, fromnode, tonode, np.round(fromMW, decimals=self.dec), np.round(
                toMW, decimals=self.dec), np.round(loss, decimals=self.dec)])
        print('-----------')
        print('DC DC Coverters')
        print(table)

    def Converter(self):
        table = pt()
        table2 = pt()
        table.field_names = ["Converter", "AC node", "DC node","Power s AC (MW)","Reactive s AC (MVAR)", "Power c AC (MW)", "Power DC(MW)", "Reactive power (MVAR)", "Power loss IGBTs (MW)", "Power loss AC elements (MW)"]
        table2.field_names = ["Converter","AC control mode", "DC control mode","Loading %","Capacity [MVA]"]
        for conv in self.Grid.Converters_ACDC:
            if conv.NumConvP<=0.01:
                continue
            P_DC = np.round(conv.P_DC*self.Grid.S_base, decimals=self.dec)
            P_s = np.round(conv.P_AC*self.Grid.S_base, decimals=self.dec)
            Q_s = np.round(conv.Q_AC*self.Grid.S_base, decimals=self.dec)
            P_c = np.round(conv.Pc*self.Grid.S_base, decimals=self.dec)
            Q_c = np.round(conv.Qc*self.Grid.S_base, decimals=self.dec)
            P_loss = np.round(conv.P_loss*self.Grid.S_base, decimals=self.dec)
            Ploss_tf = np.round(conv.P_loss_tf*self.Grid.S_base, decimals=self.dec)
            S = np.sqrt(P_s**2+Q_s**2)
            
            loading= np.round(max(S,abs(P_DC))/(conv.MVA_max*conv.NumConvP)*100, decimals=self.dec)
            table.add_row([conv.name, conv.Node_AC.name,
                          conv.Node_DC.name, P_s,Q_s ,P_c, P_DC, Q_c, P_loss, Ploss_tf])
            table2.add_row([conv.name, conv.AC_type, conv.type,loading,int(conv.MVA_max*conv.NumConvP)])

        print('------------')
        print('AC DC Converters')
        if len(table.rows) > 0:  # Check if the table is not None and has at least one row
            print(table)
            print(table2)

        if self.export is not None:
            csv_filename = f'{self.export}/Converter_results.csv'
            csv_data = table.get_csv_string()

            with open(csv_filename, 'w', newline='') as csvfile:
                csvfile.write(csv_data)

    
    
    
