from pymoo.core.problem import ElementwiseProblem
import numpy as np
import pyomo.environ as pyo
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
import time
import matplotlib.pyplot as plt
from .ACDC_OPF_NL_model import analyse_OPF
from .ACDC_OPF import OPF_solve,OPF_obj,obj_w_rule,ExportACDC_NLmodel_toPyflowACDC

__all__ = [
    'transmission_expansion_pymoo'
]


    

class TEPOuterProblem(ElementwiseProblem):
    def __init__(self, grid, weights_def, n_years, Hy, r, pv_set=False, pz=False, time_limit=60):
        t1=time.perf_counter()
        analyse_OPF(grid)
        self.grid = grid
        self._store_TEP_flags()
        n_var, xl, xu, vtype, self.bound_names = self._create_pymoo_bounds()
         
        super().__init__(n_var=n_var, xl=xl, xu=xu, vtype=vtype)  # mix with bools if needed
        
        self.weights_def = weights_def
        self.present_value = Hy * (1 - (1 + r) ** -n_years) / r
        self.pv_set = pv_set
        self.pz = pz
        self.time_limit = time_limit
        self.pyomo_runs = 0
        self.pyomo_time = 0


        self.model = self._build_model()  # built once with TEP=False
        self.t_modelcreate = time.perf_counter() - t1

    def _create_pymoo_bounds(self):
        """Create bounds and variable types for pymoo based on original TEP flags"""
        bounds = []
        xl = []  # lower bounds
        xu = []  # upper bounds
        vtype = []  # variable types
        self.idx_to_object = {}  # NEW: mapping from pymoo index to object info
        idx = 0
        
        # AC expansions (integer variables)
        for l in self.grid.lines_AC_exp:
            if self.original_np_line_opf.get(l.lineNumber, False):
                xl.append(l.np_line)  # minimum: current value
                xu.append(l.np_line_max)  # maximum: max allowed
                vtype.append(int)
                bounds.append(f"AC_exp_{l.lineNumber}")
                self.idx_to_object[idx] = (l.lineNumber, "np_line_AC")
                idx += 1
        
        # DC lines (integer variables)
        for l in self.grid.lines_DC:
            if self.original_np_line_opf_DC.get(l.lineNumber, False):  # Check current flag (should be True for DC lines)
                xl.append(l.np_line)
                xu.append(l.np_line_max)
                vtype.append(int)
                bounds.append(f"DC_line_{l.lineNumber}")
                self.idx_to_object[idx] = (l.lineNumber, "np_line_DC")
                idx += 1
        # Converters (integer variables)
        for c in self.grid.Converters_ACDC:
            if self.original_NUmConvP_opf.get(c.ConvNumber, False):
                xl.append(c.NumConvP)
                xu.append(c.NumConvP_max)
                vtype.append(int)
                bounds.append(f"Conv_{c.ConvNumber}")
                self.idx_to_object[idx] = (c.ConvNumber, "NumConvP_ACDC")
                idx += 1
        # AC repurposing (binary variables)
        for l in self.grid.lines_AC_rec:
            if self.original_rec_line_opf.get(l.lineNumber, False):
                xl.append(0)
                xu.append(1)
                vtype.append(int)  # pymoo uses int for binary
                bounds.append(f"AC_rec_{l.lineNumber}")
                self.idx_to_object[idx] = (l.lineNumber, "rec_line_AC")
                idx += 1
        # Array cable type (integer variables: -1 to max_cable_type)
        for l in self.grid.lines_AC_ct:
            if self.original_array_opf.get(l.lineNumber, False):
                if self.grid.Array_opf: 
                    xl.append(-1)  # -1 means no cable
                    self.idx_to_object[idx] = (l.lineNumber, "Array_ct_AC")
                elif l.active_config < 0:
                    continue
                else:
                    xl.append(0)
                    self.idx_to_object[idx] = (l.lineNumber, "CSS_AC")
                xu.append(len(l._cable_types) - 1)  # max cable type index
                vtype.append(int)
                bounds.append(f"Array_ct_{l.lineNumber}")
                
                idx += 1
        # AC generators (if GPR is enabled)
        if self.grid.GPR:
            for g in self.grid.Generators:
                if self.original_np_gen_opf.get(g.genNumber, False):
                    xl.append(g.np_gen)
                    xu.append(g.np_gen_max)
                    vtype.append(int)
                    bounds.append(f"Gen_{g.genNumber}")
                    self.idx_to_object[idx] = (g.genNumber, "ac_gen")
                    idx += 1
        # DC generators (if GPR is enabled)
        if self.grid.GPR:
            for g in self.grid.Generators_DC:
                if self.original_np_gen_opf_DC.get(g.genNumber_DC, False):
                    xl.append(g.np_gen)
                    xu.append(g.np_gen_max)
                    vtype.append(int)
                    bounds.append(f"Gen_DC_{g.genNumber_DC}")
                    self.idx_to_object[idx] = (g.genNumber_DC, "dc_gen")
                    idx += 1
        return len(xl), xl, xu, vtype, bounds 


    def _store_TEP_flags(self):
        # Store original True values in dictionaries
        self.original_np_gen_opf = {}
        self.original_np_gen_opf_DC = {}

        self.original_np_line_opf = {}
        self.original_rec_line_opf = {}
        self.original_array_opf = {}

        self.original_np_line_opf_DC = {}
        self.original_NUmConvP_opf = {}
        
        for g in self.grid.Generators:
            self.original_np_gen_opf[g.genNumber] = g.np_gen_opf
            g.np_gen_opf = False
        
        # AC expansions
        for l in self.grid.lines_AC_exp:
            self.original_np_line_opf[l.lineNumber] = l.np_line_opf
            l.np_line_opf = False
        
        # AC repurposing
        for l in self.grid.lines_AC_rec:
            self.original_rec_line_opf[l.lineNumber] = l.rec_line_opf
            l.rec_line_opf = False
        
        # Array cable type
        for l in self.grid.lines_AC_ct:
            self.original_array_opf[l.lineNumber] = l.array_opf
            l.array_opf = False
        

        for g in self.grid.Generators_DC:
            self.original_np_gen_opf_DC[g.genNumber_DC] = g.np_gen_opf
            g.np_gen_opf = False

        for l in self.grid.lines_DC:
            self.original_np_line_opf_DC[l.lineNumber] = l.np_line_opf
            l.np_line_opf = False
    
        # Converters
        for c in self.grid.Converters_ACDC:
            self.original_NUmConvP_opf[c.ConvNumber] = c.NUmConvP_opf
            c.NUmConvP_opf = False

    def _restore_TEP_flags(self):
        for g in self.grid.Generators:
            g.np_gen_opf = self.original_np_gen_opf[g.genNumber]

        for l in self.grid.lines_AC_exp:
            l.np_line_opf = self.original_np_line_opf[l.lineNumber]
        for l in self.grid.lines_AC_rec:
            l.rec_line_opf = self.original_rec_line_opf[l.lineNumber]
        for l in self.grid.lines_AC_ct:
            l.array_opf = self.original_array_opf[l.lineNumber]

        for l in self.grid.lines_DC:
            l.np_line_opf = self.original_np_line_opf_DC[l.lineNumber]
       

        for c in self.grid.Converters_ACDC:
            c.NUmConvP_opf = self.original_NUmConvP_opf[c.ConvNumber]
    
    def _build_model(self):
        model = pyo.ConcreteModel()
        model.name = "TEP pymoo OPF"
        
        # Import the OPF builder
        from .ACDC_OPF_NL_model import OPF_create_NLModel_ACDC
        
        # Build with TEP=False so investments are Params
        OPF_create_NLModel_ACDC(model, self.grid, PV_set=self.pv_set, 
                               Price_Zones=self.pz, TEP=False)
        
        obj_OPF = OPF_obj(model,self.grid,self.weights_def)
    
        model.obj = pyo.Objective(rule=obj_OPF, sense=pyo.minimize)
        return model

    

    def _capex_from_model(self,NPV=True):
        capex = 0.0
        def Gen_investments():
            np_gen_TEP = {k: np.float64(pyo.value(v)) for k, v in self.model.np_gen.items()}
            Gen_Inv = 0
            if hasattr(self.model, 'gen_AC') and hasattr(self.model, 'np_gen'):
                for g in self.model.gen_AC:
                    gen = self.grid.Generators[g]
                    if self.original_np_gen_opf.get(g, False):
                        Gen_Inv += (np_gen_TEP[g] - gen.np_gen) * gen.base_cost
            return Gen_Inv

        def AC_Line_investments():
            AC_Inv_lines = 0
            lines_AC_TEP = {k: np.float64(pyo.value(v)) for k, v in self.model.NumLinesACP.items()}
            if hasattr(self.model, 'lines_AC_exp') and hasattr(self.model, 'NumLinesACP'):
                for l in self.model.lines_AC_exp:
                    line = self.grid.lines_AC_exp[l]
                    if self.original_np_line_opf.get(l, False):
                        if NPV:
                            AC_Inv_lines += (lines_AC_TEP[l] - line.np_line) * line.base_cost
                        else:
                            AC_Inv_lines += (lines_AC_TEP[l] - line.np_line) * line.base_cost / line.life_time_hours
            return AC_Inv_lines
        
        def Repurposing_investments():
            Rep_Inv_lines = 0
            lines_AC_REP = {k: np.float64(pyo.value(v)) for k, v in self.model.rec_branch.items()}
            if hasattr(self.model, 'lines_AC_rec') and hasattr(self.model, 'rec_branch'):
                for l in self.model.lines_AC_rec:
                    line = self.grid.lines_AC_rec[l]
                    if self.original_rec_line_opf.get(l, False):
                        if NPV:
                            Rep_Inv_lines += lines_AC_REP[l] * line.base_cost
                        else:
                            Rep_Inv_lines += lines_AC_REP[l] * line.base_cost / line.life_time_hours
            return Rep_Inv_lines
        
        def Cables_investments():
            lines_DC_TEP = {k: np.float64(pyo.value(v)) for k, v in self.model.NumLinesDCP.items()}
            Inv_lines = 0
            if hasattr(self.model, 'lines_DC') and hasattr(self.model, 'NumLinesDCP'):
                for l in self.model.lines_DC:
                    line = self.grid.lines_DC[l]
                    if self.original_np_line_opf_DC.get(l, False):
                        if NPV:
                            Inv_lines += (lines_DC_TEP[l] - line.np_line) * line.base_cost
                        else:
                            Inv_lines += (lines_DC_TEP[l] - line.np_line) * line.base_cost / line.life_time_hours
            return Inv_lines

        def Array_investments():
            Inv_array = 0
            lines_AC_CT = {k: {ct: np.float64(pyo.value(self.model.ct_branch[k, ct])) for ct in self.model.ct_set} for k in self.model.lines_AC_ct}
            if hasattr(self.model, 'lines_AC_ct') and hasattr(self.model, 'ct_branch'):
                for l in self.model.lines_AC_ct:
                    line = self.grid.lines_AC_ct[l]
                    if self.original_array_opf.get(l, False):
                        if NPV:
                            for ct in self.model.ct_set:
                                Inv_array += lines_AC_CT[l, ct] * line.base_cost[ct]
                        else:
                            for ct in self.model.ct_set:
                                Inv_array += lines_AC_CT[l, ct] * line.base_cost[ct] / line.life_time_hours
            return Inv_array
            
        def Converter_investments():
            Inv_conv = 0
            NumConvP_TEP = {k: np.float64(pyo.value(v)) for k, v in self.model.NumConvP.items()}
            if hasattr(self.model, 'conv') and hasattr(self.model, 'NumConvP'):
                for cn in self.model.conv:
                    conv = self.grid.Converters_ACDC[cn]
                    if self.original_NUmConvP_opf.get(cn, False):
                        if NPV:
                            Inv_conv += (NumConvP_TEP[cn] - conv.NumConvP) * conv.base_cost
                        else:
                            Inv_conv += (NumConvP_TEP[cn] - conv.NumConvP) * conv.base_cost / conv.life_time_hours
            return Inv_conv
        def DC_Gen_investments():
            Inv_gen = 0
            np_gen_TEP = {k: np.float64(pyo.value(v)) for k, v in self.model.np_gen_DC.items()}
            if hasattr(self.model, 'gen_DC') and hasattr(self.model, 'np_gen_DC'):
                for g in self.model.gen_DC:
                    gen = self.grid.Generators_DC[g]
                    if self.original_np_gen_opf_DC.get(g, False):
                        Inv_gen += (np_gen_TEP[g] - gen.np_gen) * gen.base_cost
            return Inv_gen

        if self.grid.GPR:
            capex += Gen_investments()      
            capex += DC_Gen_investments()
        if self.grid.TEP_AC:
            capex += AC_Line_investments()     
        if self.grid.REC_AC:
            capex += Repurposing_investments()        
        if self.grid.CT_AC:
            capex += Cables_investments()       
        if self.grid.Array_opf:
            capex += Array_investments()       
        if self.grid.ACmode and self.grid.DCmode:
            capex += Converter_investments()
        
        return capex

    def _update_model_from_vector(self, x):
        for idx, value in enumerate(x):
            obj_id, obj_type = self.idx_to_object[idx]
            
            if obj_type == 'np_line_AC':
                self.model.NumLinesACP[obj_id].set_value(int(value))
            elif obj_type == 'np_line_DC':
                self.model.NumLinesDCP[obj_id].set_value(int(value))
            elif obj_type == 'NumConvP_ACDC':
                self.model.NumConvP[obj_id].set_value(int(value))
            elif obj_type == 'rec_line_AC':
                self.model.rec_branch[obj_id].set_value(bool(value))
            elif obj_type == 'CSS_AC':
                # Set one-hot encoding for cable type
                for ct in self.model.ct_set:
                    self.model.ct_branch[obj_id, ct].set_value(1 if ct == int(value) else 0)
            elif obj_type == 'Array_ct_AC':
                if int(value) == -1:
                    # No cable type selected - set all to 0
                    for ct in self.model.ct_set:
                        self.model.ct_branch[obj_id, ct].set_value(0)
                else:
                    # Cable type selected - one-hot encoding
                    for ct in self.model.ct_set:
                        self.model.ct_branch[obj_id, ct].set_value(1 if ct == int(value) else 0)

            elif obj_type == 'ac_gen':
                self.model.np_gen[obj_id].set_value(int(value))
            elif obj_type == 'dc_gen':
                self.model.np_gen_DC[obj_id].set_value(int(value))
        
    
    def _evaluate(self, x, out, *args, **kwargs):
        try:
            # Suppress Pyomo logging warnings
            import logging
            pyomo_logger = logging.getLogger('pyomo')
            original_level = pyomo_logger.level
            pyomo_logger.setLevel(logging.ERROR) 
            # Refresh Pyomo Params instead of rebuilding
            self._update_model_from_vector(x)
            self.pyomo_runs += 1
            
            results, stats = OPF_solve(self.model, self.grid, solver='ipopt', tee=False, time_limit=self.time_limit, suppress_warnings=True)
            self.pyomo_time += stats['time']
            if results is None:
                out["F"] = 1e24
                return
            capex = self._capex_from_model()
            opex = pyo.value(self.model.obj)
            out["F"] = capex + self.present_value * opex
        except Exception:
            out["F"] = 1e24

    def export_solution_to_grid(self, x):
        """Export the best pymoo solution back to the grid object"""
        
        # Update model with the solution
        self._update_model_from_vector(x)
        results, stats = OPF_solve(self.model, self.grid, solver='ipopt', tee=False, time_limit=self.time_limit, suppress_warnings=True)
            
        # Export the model to grid (reuse existing export function)
        from .ACDC_OPF import ExportACDC_NLmodel_toPyflowACDC
        

        # Get price zones info (you might need to pass this from __init__)
        PZ = getattr(self, 'price_zones', False)
        # Restore original TEP flags
        self._restore_TEP_flags()
        # Export the solved model to grid
        ExportACDC_NLmodel_toPyflowACDC(self.model, self.grid, PZ, TEP=False)
            
        
        
        return self.grid

def transmission_expansion_pymoo(grid,NPV=True,n_years=25,Hy=8760,discount_rate=0.02,ObjRule=None,solver='bonmin',time_limit=300,tee=False,export=True,PV_set=False):
    
            
    analyse_OPF(grid)
    
    weights_def, PZ = obj_w_rule(grid,ObjRule,True)
    # Create problem
    problem = TEPOuterProblem(grid, weights_def, n_years=n_years, Hy=Hy, r=discount_rate)

    # Run optimization
    algorithm = GA(pop_size=50)
    
    res = minimize(problem, algorithm, 
                  ('n_gen', 30), ('f_tol', 1e-6),
                  save_history=True,
                  verbose=True)
    t2 = time.perf_counter()
    # Export best solution to grid
    best_solution = res.X  # Best decision vector
    grid = problem.export_solution_to_grid(best_solution)
    t3 = time.perf_counter()
    # Now grid contains the optimized solution
    print(f"Best objective: {res.F[0]}")
    print(f"Grid now has optimized investments")

    

    print(f"Number of Pyomo runs: {problem.pyomo_runs}")
    print(f"Pyomo time: {problem.pyomo_time}")
    print(f"mean Pyomo time: {problem.pyomo_time / problem.pyomo_runs}")
    val = [e.opt.get("F")[0] for e in res.history]
    plt.plot(np.arange(len(val)), val)
    plt.show()
    timing_info = {
        "create": problem.t_modelcreate,  # Model creation time (negligible for pymoo)
        "solve": res.exec_time,  # Optimization time
        "export": t3-t2,  # Export time (negligible)
    }
    
    solver_stats = {
        'iterations': len(val),
        'best_objective': res.F[0],
        'time': res.exec_time,
        'termination_condition': 'optimal',
        'feasible_solutions': []
    }
    # Return same format as original: model, results, timing_info, solver_stats
    return problem.model, res, timing_info, solver_stats