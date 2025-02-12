# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:59:08 2024

@author: BernardoCastro
"""
import numpy as np
from prettytable import PrettyTable as pt
import matplotlib.pyplot as plt
import pandas as pd

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

        self.AC_Powerflow()
        self.AC_voltage()
        self.AC_lines_current()
        self.AC_lines_power()
        
        if self.Grid.nodes_DC != None:
            if self.Grid.nconv != 0:
                self.Converter()
            self.DC_bus()
            self.DC_lines_current()
            self.DC_lines_power()
            self.Slack_All()
        else:
            self.Slack_AC()

        self.Power_loss()
        if self.Grid.OPF_run :
            self.Ext_gen()
            if any(node.RenSource for node in self.Grid.nodes_AC):
            
                self.Ext_WPP()
            self.Price_Zone()    
        if self.Grid.TEP_run:
            self.TEP_N()
            self.TEP_ts_res()
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

        if self.Grid.Converters_DCDC != None:
            self.DC_converter()

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

    def DC_converter(self):
        table = pt()

        table.field_names = ["Converter", "From node", "To node",
                             "Power from (MW)", "Power To (MW))", "Power Loss (MW)"]
        for conv in self.Grid.Converters_DCDC:
            convid = conv.name
            fromnode = conv.fromNode.name
            tonode = conv.toNode.name
            fromMW = conv.Powerfrom*self.Grid.S_base
            toMW = conv.PowerTo*self.Grid.S_base
            loss = fromMW-toMW

            table.add_row([convid, fromnode, tonode, np.round(fromMW, decimals=self.dec), np.round(
                toMW, decimals=self.dec), np.round(loss, decimals=self.dec)])
        print('-----------')
        print('DC DC Coverters')
        print(table)

    def Power_loss(self):
        table = pt()
        # Define the table headers
        table.field_names = ["Grid", "Power Loss (MW)","Load %"]
        if self.Grid.nodes_AC != None:
            self.lossP_AC = np.zeros(self.Grid.Num_Grids_AC)
            for line in self.Grid.lines_AC:
                node = line.fromNode
                G = self.Grid.Graph_node_to_Grid_index_AC[node.nodeNumber]
                Ploss = np.real(line.loss)*self.Grid.S_base
               
                     
                i = line.fromNode.nodeNumber
                j = line.toNode.nodeNumber
                Sfrom = abs(line.fromS)*self.Grid.S_base
                Sto   = abs(line.toS)*self.Grid.S_base

                load = max(Sfrom, Sto)
                
                self.Grid.load_grid_AC[G] += load
                
                self.lossP_AC[G] += Ploss
                
            tot = 0
            for g in range(self.Grid.Num_Grids_AC):
                if self.Grid.rating_grid_AC[g]!=0:
                    gload=self.Grid.load_grid_AC[g]/self.Grid.rating_grid_AC[g]*100
                else:
                    gload=0
                table.add_row([f'AC Grid {g+1}', np.round(self.lossP_AC[g], decimals=self.dec),np.round(gload, decimals=self.dec)])
                tot += self.lossP_AC[g]

        if self.Grid.nodes_DC != None:

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

        if self.Grid.Converters_ACDC != None:
            P_loss_ACDC = 0
            for conv in self.Grid.Converters_ACDC:
                P_loss_ACDC += (conv.P_loss_tf+conv.P_loss)*self.Grid.S_base
                tot += (conv.P_loss_tf+conv.P_loss)*self.Grid.S_base
         
            table.add_row(['AC DC Converters', np.round(P_loss_ACDC, decimals=self.dec),""])

        if self.Grid.Converters_DCDC != None:
            P_loss_DCDC = 0
            for conv in self.Grid.Converters_DCDC:
                P_loss_DCDC += (abs(conv.PowerTo-conv.Powerfrom))*self.Grid.S_base
                tot += (abs(conv.PowerTo-conv.Powerfrom))*self.Grid.S_base

            table.add_row(['DC DC Converters', np.round(P_loss_DCDC, decimals=self.dec),""])

        table.add_row(["Total loss", np.round(tot, decimals=self.dec),""])
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
            G = self.Grid.Graph_node_to_Grid_index_DC[node]

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
        print('--------------')
        print('Results DC')
        print('')
        table_all = pt()
        table_all.field_names = [
            "Node", "Power Gen (MW)", "Power Load (MW)", "Power Converter ACDC (MW)", "Power Converter DCDC (MW)", "Power injected (MW)", "Voltage (pu)", "Grid"]
        for g in range(self.Grid.Num_Grids_DC):
            print(f'Grid DC {g+1}')

            table = pt()

            # Define the table headers
            table.field_names = [
                "Node", "Power Gen (MW)", "Power Load (MW)", "Power Converter ACDC (MW)", "Power Converter DCDC (MW)", "Power injected (MW)", "Voltage (pu)"]

            for node in self.Grid.nodes_DC:
                if self.Grid.Graph_node_to_Grid_index_DC[node.nodeNumber] == g:
                    if node.type == 'Slack':
                        if self.Grid.nconv == 0:
                            if node.P_INJ > 0:
                                node.PGi = node.P_INJ
                            else:
                                node.PLi = abs(node.P_INJ)
                    conv  = np.round(node.P*self.Grid.S_base, decimals=self.dec)
                    convDC =np.round(node.PconvDC *self.Grid.S_base, decimals=self.dec)
                    table.add_row([node.name, np.round(node.PGi*self.Grid.S_base, decimals=self.dec), np.round(node.PLi*self.Grid.S_base,
                                  decimals=self.dec), conv,convDC, np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec), np.round(node.V, decimals=self.dec)])
                    table_all.add_row([node.name, np.round(node.PGi*self.Grid.S_base, decimals=self.dec), np.round(node.PLi*self.Grid.S_base,
                                      decimals=self.dec), conv,convDC, np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec), np.round(node.V, decimals=self.dec), g+1])

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
                            PGi = node.PGi+node.PGi_ren*node.curtailment +node.PGi_opt
                            QGi = node.QGi +node.QGi_opt
                            if node.type == 'Slack':
                                PGi = node.P_INJ+node.PLi
                                QGi = node.Q_INJ+node.QLi

                            if node.type == 'PV':
                                node.QGi = node.Q_INJ - \
                                    (node.Q_s+node.Q_s_fx)+node.QLi

                            table.add_row([node.name, np.round(PGi*self.Grid.S_base, decimals=self.dec), np.round(QGi*self.Grid.S_base, decimals=self.dec), np.round(node.PLi*self.Grid.S_base, decimals=self.dec), np.round(
                                node.QLi*self.Grid.S_base, decimals=self.dec), np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec), np.round(node.Q_INJ*self.Grid.S_base, decimals=self.dec)])

                else:
                    # Define the table headers
                    table.field_names = ["Node", "Power Gen (MW)", "Reactive Gen (MVAR)", "Power Load (MW)", "Reactive Load (MVAR)",
                                         "Power converters DC(MW)", "Reactive converters DC (MVAR)", "Power injected  (MW)", "Reactive injected  (MVAR)"]

                    for node in self.Grid.nodes_AC:
                        if self.Grid.Graph_node_to_Grid_index_AC[node.nodeNumber] == g:
                            PGi = node.PGi +node.PGi_res*node.curtailment+node.PGi_opt
                            QGi = node.QGi +node.QGi_opt
                            if node.type == 'Slack':
                                PGi = (node.P_INJ-node.P_s + node.PLi).item()
                                QGi = node.Q_INJ-node.Q_s-node.Q_s_fx+node.QLi

                            if node.type == 'PV':
                                QGi = node.Q_INJ -(node.Q_s+node.Q_s_fx)+node.QLi

                            table.add_row([node.name, np.round(PGi*self.Grid.S_base, decimals=self.dec), np.round(QGi*self.Grid.S_base, decimals=self.dec), np.round(node.PLi*self.Grid.S_base, decimals=self.dec), np.round(node.QLi*self.Grid.S_base, decimals=self.dec), np.round(
                                node.P_s*self.Grid.S_base, decimals=self.dec).item(), np.round((node.Q_s+node.Q_s_fx)*self.Grid.S_base, decimals=self.dec).item(), np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec), np.round(node.Q_INJ*self.Grid.S_base, decimals=self.dec)])
                          
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
                            PGi = node.PGi+node.PGi_ren*node.curtailment +node.PGi_opt
                            QGi = node.QGi +node.QGi_opt
                            if node.type == 'Slack':
                                PGi = node.P_INJ+node.PLi
                                QGi = node.Q_INJ+node.QLi

                            if node.type == 'PV':
                                node.QGi = node.Q_INJ - \
                                    (node.Q_s+node.Q_s_fx)+node.QLi

                            table.add_row([node.name, np.round(PGi*self.Grid.S_base, decimals=self.dec), np.round(QGi*self.Grid.S_base, decimals=self.dec), np.round(node.PLi*self.Grid.S_base, decimals=self.dec), np.round(
                                node.QLi*self.Grid.S_base, decimals=self.dec), np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec), np.round(node.Q_INJ*self.Grid.S_base, decimals=self.dec)])
                            table_all.add_row([node.name, np.round(PGi*self.Grid.S_base, decimals=self.dec), np.round(QGi*self.Grid.S_base, decimals=self.dec), np.round(node.PLi*self.Grid.S_base, decimals=self.dec), np.round(
                                node.QLi*self.Grid.S_base, decimals=self.dec), np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec), np.round(node.Q_INJ*self.Grid.S_base, decimals=self.dec), g+1])

                else:
                    # Define the table headers
                    table.field_names = ["Node", "Power Gen (MW)", "Reactive Gen (MVAR)", "Power Load (MW)", "Reactive Load (MVAR)",
                                         "Power converters DC(MW)", "Reactive converters DC (MVAR)", "Power injected  (MW)", "Reactive injected  (MVAR)"]

                    for node in self.Grid.nodes_AC:
                        if self.Grid.Graph_node_to_Grid_index_AC[node.nodeNumber] == g:
                            PGi = node.PGi+node.PGi_ren*node.curtailment +node.PGi_opt
                            QGi = node.QGi +node.QGi_opt
                            if node.type == 'Slack':
                                PGi = (node.P_INJ-node.P_s + node.PLi).item()
                                QGi = node.Q_INJ-node.Q_s-node.Q_s_fx+node.QLi

                            if node.type == 'PV':
                                QGi = node.Q_INJ -(node.Q_s+node.Q_s_fx)+node.QLi

                            table.add_row([node.name, np.round(PGi*self.Grid.S_base, decimals=self.dec), np.round(QGi*self.Grid.S_base, decimals=self.dec), np.round(node.PLi*self.Grid.S_base, decimals=self.dec), np.round(node.QLi*self.Grid.S_base, decimals=self.dec), np.round(
                                node.P_s*self.Grid.S_base, decimals=self.dec).item(), np.round((node.Q_s+node.Q_s_fx)*self.Grid.S_base, decimals=self.dec).item(), np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec), np.round(node.Q_INJ*self.Grid.S_base, decimals=self.dec)])
                            table_all.add_row([node.name, np.round(PGi*self.Grid.S_base, decimals=self.dec), np.round(QGi*self.Grid.S_base, decimals=self.dec), np.round(node.PLi*self.Grid.S_base, decimals=self.dec), np.round(node.QLi*self.Grid.S_base, decimals=self.dec), np.round(
                                node.P_s*self.Grid.S_base, decimals=self.dec).item(), np.round((node.Q_s+node.Q_s_fx)*self.Grid.S_base, decimals=self.dec).item(), np.round(node.P_INJ*self.Grid.S_base, decimals=self.dec), np.round(node.Q_INJ*self.Grid.S_base, decimals=self.dec), g+1])

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
                    tablei.add_row([line.name, line.fromNode.name, line.toNode.name, np.round(
                        i_from, decimals=self.dec), np.round(i_to, decimals=self.dec), np.round(load, decimals=self.dec),int(line.MVA_rating)])
                    table_all.add_row([line.name, line.fromNode.name, line.toNode.name, np.round(
                        i_from, decimals=self.dec), np.round(i_to, decimals=self.dec), np.round(load, decimals=self.dec),int(line.MVA_rating), g+1])
            if len(tablei.rows) > 0:  # Check if the table is not None and has at least one row
                print(tablei)

        if self.export is not None:
            csv_filename = f'{self.export}/AC_line_current.csv'
            csv_data = table_all.get_csv_string()

            with open(csv_filename, 'w', newline='') as csvfile:
                csvfile.write(csv_data)

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

                        tablep.add_row([line.name, line.fromNode.name, line.toNode.name, np.round(p_from, decimals=self.dec), np.round(Q_from, decimals=self.dec), np.round(
                            p_to, decimals=self.dec), np.round(Q_to, decimals=self.dec), np.round(Ploss, decimals=self.dec), np.round(Qloss, decimals=self.dec)])

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

                        tablep.add_row([line.name, line.fromNode.name, line.toNode.name, np.round(p_from, decimals=self.dec), np.round(Q_from, decimals=self.dec), np.round(
                            p_to, decimals=self.dec), np.round(Q_to, decimals=self.dec), np.round(Ploss, decimals=self.dec), np.round(Qloss, decimals=self.dec)])
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
        table.field_names = ["Generator","Node" ,"Power (MW)", "Reactive power (MVAR)","Quadratic Price €/MWh^2","Linear Price €/MWh","Loading %","Cost k€"]
        for gen in self.Grid.Generators:
            Pgi=gen.PGen #+node.PGi_ren*node.curtailment
            Qgi=gen.QGen
            S= np.sqrt(Pgi**2+Qgi**2)
            Pgi*=self.Grid.S_base
            Qgi*=self.Grid.S_base
            if gen.Max_S is None:
                base=np.sqrt(gen.Max_pow_gen**2+max(abs(gen.Min_pow_genR),gen.Max_pow_genR)**2)
            else:
                base=gen.Max_S
            load=S/base*100
            cost=(Pgi**2*gen.qf+Pgi*gen.lf)/1000
           
                
            table.add_row([gen.name,gen.Node_AC.name, np.round(Pgi, decimals=self.dec), np.round(Qgi, decimals=self.dec),  np.round(gen.qf, decimals=self.dec),  np.round(gen.lf, decimals=self.dec),np.round(load, decimals=self.dec), np.round(cost, decimals=0)])
            Pabs+=abs(Pgi)
            Qabs+=abs(Qgi)
            Ptot+=Pgi
            Qtot+=Qgi
            Stot+=S
            costtot+=cost
            Ltot+=gen.Max_pow_gen
            
        if Ltot !=0:
            load=Stot/Ltot*100
        else:
            load=0
        table.add_row(['Total',"", np.round(Ptot, decimals=self.dec), np.round(Qtot, decimals=self.dec),"",""," ", np.round(costtot, decimals=0)])
        table.add_row(['Total abs',"", np.round(Pabs, decimals=self.dec), np.round(Qabs, decimals=self.dec), "","",np.round(load, decimals=self.dec),""])
        print(table)
    
    def Ext_WPP(self):
        print('--------------')
        print('Renewable energy sources')
        table = pt()
        table.field_names = ["Bus", "Base Power (MW)", "Curtailment %","Power Injected (MW)","Price €/MWh","Cost k€","Curtailment Cost [k€]"]
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
                table.add_row([rs.name, np.round(Pgi, decimals=self.dec), np.round(cur, decimals=self.dec),  np.round(PGicur, decimals=self.dec),np.round(price, decimals=self.dec),np.round(cost, decimals=0),np.round(curcost, decimals=0)])
                totcost+=cost
                totcurcost+=curcost
        
     
        
        PGicur=bp-tcur
        cur=(tcur)/bp*100
        
        table.add_row(['Total', np.round(bp, decimals=self.dec), np.round(cur, decimals=self.dec),  np.round(PGicur, decimals=self.dec) ,"",np.round(totcost, decimals=0),np.round(totcurcost, decimals=0)])

        print(table)    
    
    def TEP_ts_res(self):
       
        PN,GEN, SC , curt, curt_per, lines,conv,price= self.Grid.TEP_res
        
        
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
        
        
        
        
    
    def TEP_N(self,p=True):
        
        table = pt()
        table.field_names = ["Element","Type" ,"Initial", "Optimized N","Maximum","Optimized Power Rating [MW]","Expansion Cost [M€]"]
        tot=0
        tot_n=0
        
        for l in self.Grid.lines_AC_exp:
            if l.np_line_opf:
                element= l.name
                ini= l.np_line_i
                opt=l.np_line
                pr= opt*l.MVA_rating
                cost=((opt)*l.MVA_rating*l.Length_km*l.phi)*l.life_time*8760/(10**6)
                tot+=cost
                maxn=l.np_line_max
                tot_n+=((opt)*l.MVA_rating*l.Length_km*l.phi)/1000
                table.add_row([element, "AC Line" ,ini, np.round(opt, decimals=2),maxn,np.round(pr, decimals=0).astype(int), np.round(cost, decimals=2)])
        
        
        
        for l in self.Grid.lines_DC:
            if l.np_line_opf:
                element= l.name
                ini= l.np_line_i
                opt=l.np_line
                pr= opt*l.MW_rating
                cost=((opt)*l.MW_rating*l.Length_km*l.phi)*l.life_time*8760/(10**6)
                tot+=cost
                maxn=l.np_line_max
                tot_n+=((opt)*l.MW_rating*l.Length_km*l.phi)/1000
                table.add_row([element, "DC Line" ,ini, np.round(opt, decimals=2),maxn,np.round(pr, decimals=0).astype(int), np.round(cost, decimals=2)])
                
        
        for cn in self.Grid.Converters_ACDC:
            if cn.NUmConvP_opf:
                element= cn.name
                ini=cn.NumConvP_i
                opt=cn.NumConvP
                pr=opt*cn.MVA_max
                cost=((opt)*cn.MVA_max*cn.phi)*cn.life_time*8760/(10**6)
                tot+=cost
                tot_n+=((opt)*cn.MVA_max*cn.phi)/1000
                maxn=cn.NumConvP_max
                table.add_row([element, "ACDC Conv" ,ini,np.round(opt, decimals=2),maxn,np.round(pr, decimals=0).astype(int), np.round(cost, decimals=2)])
        
        table.add_row(["Total", "" ,"","", "", "",np.round(tot, decimals=2)])
        
        print('--------------')
        print('Transmission Expansion Problem')
        print(table)
        
        PN,GEN, SC , curt, curt_per, lines,conv,price= self.Grid.TEP_res
        weight = SC.loc['Weight']
        table=pt()
        table.field_names = ["Price_Zone", "Normalized social cost[k€/h]", "Average price [€/MWh]"]
        
        
        for m in self.Grid.Price_Zones:
            from .Classes import Price_Zone
            if type(m) is Price_Zone:
                price_zone_weighted = SC.loc[m.name]
                weighted_total = price_zone_weighted * weight
                weighted_total = weighted_total.sum()
                weighted_price = price.loc[m.name]* weight
                weighted_price = weighted_price.sum()
                table.add_row([m.name, np.round(weighted_total, decimals=2),np.round(weighted_price, decimals=2)])
                
        print(table)
        
        
        table=pt()
        table.field_names = ["Normalized social cost[k€/h]","Normalized investment [k€/h]","Normalized Total cost [k€/h]"]
        
        weighted_sum = SC.loc['Weighted SC'].sum()
        table.add_row([np.round(weighted_sum, decimals=2), np.round(tot_n, decimals=2),np.round(weighted_sum+tot_n, decimals=2)])
        print(table)
        
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
            "Line", "From bus", "To bus", "I (kA)", "Loading %","Capacity [MW]" ,"Polarity", "Grid"]
        for g in range(self.Grid.Num_Grids_DC):
            print(f'Grid DC {g+1}')
            tablei = pt()

            tablei.field_names = ["Line", "From bus",
                                  "To bus", "I (kA)", "Loading %","Capacity [MW]", "Polarity"]
            tablei.align["Polarity"] = 'l'

            for line in self.Grid.lines_DC:
                if line.np_line==0:
                    continue
                if self.Grid.Graph_line_to_Grid_index_DC[line] == g:

                    i = line.fromNode.nodeNumber
                    j = line.toNode.nodeNumber
                    I_base = self.Grid.S_base/line.kV_base
                    i_to = self.Grid.Iij_DC[j, i]*I_base

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
                        i_to, decimals=self.dec), np.round(load, decimals=self.dec),int(line.MW_rating) ,pol])
                    table_all.add_row([line.name, line.fromNode.name, line.toNode.name, np.round(
                        i_to, decimals=self.dec), np.round(load, decimals=self.dec),int(line.MW_rating), pol, g+1])

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
                                 "P from (MW)", "P to (MW)", "Power loss (MW)", "Grid"]
        for g in range(self.Grid.Num_Grids_DC):
            print(f'Grid DC {g+1}')
            tablep = pt()
            tablep.field_names = ["Line", "From bus", "To bus",
                                  "P from (MW)", "P to (MW)", "Power loss (MW)"]

            for line in self.Grid.lines_DC:
                if self.Grid.Graph_line_to_Grid_index_DC[line] == g:
                    
                   
                    p_to = line.toP*self.Grid.S_base
                    p_from = line.fromP*self.Grid.S_base

                    Ploss = np.real(line.loss)*self.Grid.S_base

                    tablep.add_row([line.name, line.fromNode.name, line.toNode.name, np.round(
                        p_from, decimals=self.dec), np.round(p_to, decimals=self.dec), np.round(Ploss, decimals=self.dec)])
                    table_all.add_row([line.name, line.fromNode.name, line.toNode.name, np.round(
                        p_from, decimals=self.dec), np.round(p_to, decimals=self.dec), np.round(Ploss, decimals=self.dec), g+1])

            if len(tablep.rows) > 0:  # Check if the table is not None and has at least one row
                print(tablep)

        if self.export is not None:
            csv_filename = f'{self.export}/DC_line_power.csv'
            csv_data = table_all.get_csv_string()

            with open(csv_filename, 'w', newline='') as csvfile:
                csvfile.write(csv_data)

    def Converter(self):
        table = pt()
        table2 = pt()
        table.field_names = ["Converter", "AC node", "DC node","Power s AC (MW)","Reactive s AC (MVAR)", "Power c AC (MW)", "Power DC(MW)", "Reactive power (MVAR)", "Power loss IGBTs (MW)", "Power loss AC elements (MW)"]
        table2.field_names = ["Converter","AC control mode", "DC control mode","Loading %","Capacity [MVA]"]
        for conv in self.Grid.Converters_ACDC:
            if conv.NumConvP==0:
                continue
            if conv.type == 'Slack':
                P_DC = np.round(conv.Node_DC.P*self.Grid.S_base,
                                decimals=self.dec)
            else:
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
            table2.add_row([conv.name, conv.AC_type, conv.type,loading,int(conv.MVA_max)])

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

    def Time_series_prob(self, element_name):
        a = self.Grid.Time_series
        
        df_res = self.Grid.Time_series_res
        df_line_res = self.Grid.Time_series_line_res
        df_conv_res = self.Grid.Time_series_conv_res
  
        merged_df = pd.concat([df_res, df_line_res,df_conv_res], axis=1)

        for ts in a:
             if ts.name == element_name:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

                    # Plot PDF
                    ax1.hist(ts.data, bins=100, density=True,
                             alpha=0.5, color='b')
                    ax1.set_title('Probability Density Function (PDF)')
                    ax1.set_xlabel(ts.name)
                    ax1.set_ylabel('Probability')

                    # Plot CDF
                    sorted_data = np.sort(ts.data)
                    cumulative_prob = np.linspace(0, 1, len(sorted_data))

                    # Plot the CDF as a line
                    ax2.plot(sorted_data, cumulative_prob,
                             marker='.', linestyle='-')
                    ax2.set_title('Cumulative Distribution Function (CDF)')
                    ax2.set_xlabel(ts.name)
                    ax2.set_ylabel('Probability')

                    plt.show()

        for col in merged_df.columns:
            if col.startswith(element_name) or col.endswith(element_name):
                # Assuming a normal distribution

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

                # Plot PDF
                ax1.hist(merged_df[col], bins=100,
                         density=True, alpha=0.5, color='b')
                ax1.set_title('Probability Density Function (PDF)')
                ax1.set_xlabel(col)
                ax1.set_ylabel('Probability')

                # Plot CDF
                sorted_data = np.sort(merged_df[col])
                cumulative_prob = np.linspace(0, 1, len(sorted_data))

                # Plot the CDF as a line
                ax2.plot(sorted_data, cumulative_prob,
                         marker='.', linestyle='-')
                ax2.set_title('Cumulative Distribution Function (CDF)')
                ax2.set_xlabel(col)
                ax2.set_ylabel('Probability')

                plt.show()

    def Time_series_plots(self, start=1, end=9999,All=False):

        if self.Grid.Time_series_res.index[-1] < end:
            end = self.Grid.Time_series_res.index[-1]

        if self.Grid.Time_series_res.index[0] > start:
            start = self.Grid.Time_series_res.index[0]
        
        np_start = start - 1
        np_end = end - 1

        for ts in self.Grid.Time_series:
           if ts.type  in ['WPP', 'OWPP','SF','REN']:
               # Assuming ts.data is a pandas Series or a list of data points indexed by time
               time_range = slice(np_start, np_end + 1)
               # Plot ts.data (NumPy array indexed by np.index)
               plt.plot(range(start, end + 1), ts.data[time_range], label=ts.name)
        plt.title('Renewable sources (pu)')
        plt.xlabel('Time')
        plt.ylabel('PG Value (pu)')
        plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

        for ts in self.Grid.Time_series:
           if ts.type  in ['Load']:
               # Assuming ts.data is a pandas Series or a list of data points indexed by time
               time_range = slice(np_start, np_end + 1)
               # Plot ts.data (NumPy array indexed by np.index)
               plt.plot(range(start, end + 1), ts.data[time_range], label=ts.name)
        plt.title('Load (pu)')
        plt.xlabel('Time')
        plt.ylabel('Load Value (pu)')
        plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

        if self.Grid.VarPrice :
            price = [col for col in self.Grid.Time_series_price.columns]
            self.Grid.Time_series_price.loc[start+1:end+1, price].plot(
                kind='line', title='Price in different price_zones')
            plt.xlabel('Time')
            plt.ylabel('Price (Eu/pu)')
            plt.legend(title='Source', bbox_to_anchor=(
                1.05, 1), loc='upper left')
            plt.show()

          

        # Plotting Pg for Slack nodes
        slack_colsPg_ = [col for col in self.Grid.Time_series_res.columns if 'Pg_' in col]
        PG_data= self.Grid.Time_series_res.loc[start:end, slack_colsPg_]
        
        legend_labels = [col.replace('Pg_', '') for col in slack_colsPg_]
        
        PG_data.plot(kind='line', title='Power generation')
        plt.xlabel('Time')
        plt.ylabel('PG Value (pu)')
        plt.legend(title='Node', labels=legend_labels,bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

        # Plotting Qg for Slack nodes
        slack_colsQg_ = [col for col in self.Grid.Time_series_res.columns if 'Qg_' in col]
        QF_data= self.Grid.Time_series_res.loc[start:end, slack_colsQg_]
        
        legend_labels = [col.replace('Qg_', '') for col in slack_colsQg_]
        QF_data.plot(kind='line', title='Reactive power generation')
        plt.xlabel('Time')
        plt.ylabel('QG Value (pu)')
        plt.legend(title='Node',  labels=legend_labels,bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

           
        # Plotting PDC for Converters
        converter_cols = [col for col in self.Grid.Time_series_conv_res.columns if '_P_DC' in col]
        P_DC_data=self.Grid.Time_series_conv_res.loc[start:end, converter_cols]
        
        legend_labels = [col.replace('_P_DC', '') for col in converter_cols]
        P_DC_data.plot(kind='line', title='DC power in each Converter')
        plt.xlabel('Time')
        plt.ylabel('DC power')
        plt.legend(title='Converter',  labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        
        if All ==True:
            # Plotting AC line Loads
            line_cols = [
                col for col in self.Grid.Time_series_line_res.columns if 'AC_Load_' in col]
            line_load_data = self.Grid.Time_series_line_res.loc[start:end, line_cols]
    
            # Remove "AC_" from legend labels
            legend_labels = [col.replace('AC_AC_Load_','') for col in line_cols]
    
            line_load_data.plot(kind='line', title='Power load in each AC Line')
            plt.xlabel('Time')
            plt.ylabel('Line load (pu of power)')
            plt.legend(title='Line', labels=legend_labels,bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
    
            # Plotting DC line Loads
            line_cols = [
                col for col in self.Grid.Time_series_line_res.columns if 'DC_Load_' in col]
            line_load_data = self.Grid.Time_series_line_res.loc[start:end, line_cols]
    
            # Remove "DC_" from legend labels
            legend_labels = [col.replace('DC_Load_', '') for col in line_cols]
    
            line_load_data.plot(kind='line', title='Power load in each  DC Line')
            plt.xlabel('Time')
            plt.ylabel('Line load (pu of power)')
            plt.legend(title='Line', labels=legend_labels,
                       bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
    
    
    