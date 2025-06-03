import pyflow_acdc as pyf


pyf.initialize_pyflowacdc()

S_base = 1000

DC_node_1 = pyf.Node_DC(node_type='Slack', Voltage_0=1,kV_base=525,name='Node_1')
DC_node_2 = pyf.Node_DC(node_type='P', Voltage_0=1,kV_base=525,name='Node_2',Power_load=1)
DC_node_3 = pyf.Node_DC(node_type='P', Voltage_0=1,kV_base=525,name='Node_3')

DC_nodes = [DC_node_1, DC_node_2, DC_node_3]

grid = pyf.Grid(S_base,nodes_DC=DC_nodes)
res= pyf.Results(grid,decimals=5)

r= 0.010575 #Ohm/dist
d12 = 5
d13 = 15
d23 = 60


pyf.add_line_DC(grid,DC_node_1, DC_node_2,R_Ohm_km=r,Length_km=d12,MW_rating=9999,polarity='sm',update_grid=False)
pyf.add_line_DC(grid,DC_node_1, DC_node_3,R_Ohm_km=r,Length_km=d13,MW_rating=9999,polarity='sm',update_grid=False)
pyf.add_line_DC(grid,DC_node_2, DC_node_3,R_Ohm_km=r,Length_km=d23,MW_rating=9999,polarity='sm',update_grid=False)

grid.create_Ybus_DC()
grid.Update_Graph_DC()

pyf.add_RenSource(grid,'Node_1', 2000)
   

model, timing_info, model_res,solver_stats=pyf.Optimal_PF(grid)

# model.pprint()
res.All()

print(model_res)
print(timing_info)
model.obj.display()



