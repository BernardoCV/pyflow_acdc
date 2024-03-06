# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:23:18 2024

@author: BernardoCastro
"""

import numpy as np
import sys

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


def ACDC_sequential(grid, tol_lim=1e-6, maxIter=20, change_slack2Droop=False, QLimit=True):

    tolerance = 1
    grid.iter_num_seq = 0

    for conv in grid.Converters_ACDC:
        if conv.type != 'PAC':
            AC_node = conv.Node_AC
            DC_node = conv.Node_DC
            DC_node.Pconv = conv.P_DC
            P_DC = conv.P_DC
            conv.P_AC = -P_DC
            AC_node.P_s = conv.P_AC
            s = 1
            
    grid.Update_PQ_AC()

    while tolerance > tol_lim and grid.iter_num_seq < maxIter:
        grid.Ps_AC_new = np.zeros((grid.nn_AC, 1))

        load_flow_AC(grid)

        for conv in grid.Converters_ACDC:
            if conv.type == 'PAC':
                if conv.Node_AC.stand_alone == True:
                    conv.P_AC = -(conv.Node_AC.PGi-conv.Node_AC.PLi)
                    conv.Q_AC = -(conv.Node_AC.QGi-conv.Node_AC.QLi+conv.Node_AC.Q_s_fx)
                    # print('non stand alone MG')
                else:
                    if conv.AC_type == 'Slack':
                        conv.P_AC = conv.Node_AC.P_INJ-(conv.Node_AC.PGi-conv.Node_AC.PLi)
                        conv.Q_AC = conv.Node_AC.Q_INJ-(conv.Node_AC.QGi-conv.Node_AC.QLi+conv.Node_AC.Q_s_fx)
                    if conv.AC_type == 'PV':
                        conv.Q_AC = conv.Node_AC.Q_INJ-(conv.Node_AC.QGi-conv.Node_AC.QLi+conv.Node_AC.Q_s_fx)
                flow_conv_P_AC(grid,conv)

        if QLimit == True:
            for conv in grid.Converters_ACDC:
                Converter_Qlimit(grid,conv)

        if grid.iter_num_seq == 0:
            s = 1
            grid.Check_SlacknDroop(change_slack2Droop)

        s = 1

        grid.Update_PQ_AC()
        grid.Update_P_DC()

        load_flow_DC(grid)

        for conv in grid.Converters_ACDC:

            AC_node = conv.Node_AC
            DC_node = conv.Node_DC

            if conv.AC_type == 'PV':
                conv.Q_AC = AC_node.Q_INJ-(AC_node.QGi-AC_node.QLi+AC_node.Q_s_fx)
            conv.U_s = AC_node.V
            conv.th_s = AC_node.theta
            flow_conv(grid,conv)
            if conv.name == 'CmA1':
                s = 1

        s = 1
        Ps = np.copy(grid.Ps_AC)
        Ps_AC_new = np.copy(grid.Ps_AC_new)
        P_dif = Ps-Ps_AC_new

        tolerance = np.max(abs(P_dif))

        s = 1
        for node in grid.nodes_AC:
            node.P_s = Ps_AC_new[node.nodeNumber]
            node.P_s_new = 0
        grid.Update_PQ_AC()

        # print(f'{iter_num} tolerance reached: {np.round(tolerance,decimals=12)}')
        grid.iter_num_seq += 1

    if grid.iter_num_seq == maxIter:
        if tolerance > tol_lim*100:
            print('')
            print(
                f'Warning  Sequential flow did not converge in less than {maxIter} iterations')
            print(
                f'Lowest tolerance reached: {np.round(tolerance,decimals=6)}')
    grid.Line_AC_calc()
    grid.Line_DC_calc()
    
def Jacobian_DC(grid, V_DC, P):
    grid.slack_bus_number_DC = []
    J = np.zeros((grid.nn_DC, grid.nn_DC))
    V = V_DC

    for i in range(grid.nn_DC):
        m = grid.nodes_DC[i].nodeNumber

        if grid.nodes_DC[i].type != 'Slack':
            for k in range(grid.nn_DC):
                n = grid.nodes_DC[k].nodeNumber
                Y = grid.Ybus_DC[m, n]
                pol = 1

                if m != n:
                    if Y != 0:
                        line = grid.get_lineDC_by_nodes(m, n)
                        pol = line.pol

                    J[m, n] = pol*Y*V[m]*V[n]
                else:
                    J[m, n] = P[m]
                    if grid.nconv != 0:
                        if grid.nodes_DC[k].type == 'Droop':
                            J[m, n] += grid.nodes_DC[k].Droop_rate * V[m]

                    for a in range(grid.nn_DC):
                        if a != m:
                            Ya = grid.Ybus_DC[m, a]
                            if Ya != 0:
                                line = grid.get_lineDC_by_nodes(m, a)
                                pola = line.pol
                            J[m, n] += pola*-Ya*V[m]*V[m]

        else:
            grid.slack_bus_number_DC.append(m)

    grid.J_DC = J

def load_flow_DC(grid, tol_lim=1e-4, maxIter=20):

    iter_num = 0

    P_known = np.copy(grid.P_DC+grid.Pconv_DC)
    V = np.zeros(grid.nn_DC)
    s = 1

    for node in grid.nodes_DC:
        V[node.nodeNumber] = node.V
    tol = 1
    while tol > tol_lim and iter_num < maxIter:
        iter_num += 1

        P = np.zeros((grid.nn_DC, 1))
        P1 = np.zeros((grid.nn_DC, 1))
        Pf = np.zeros((grid.nn_DC, 1))
        pol = 1
        for node in grid.nodes_DC:

            i = node.nodeNumber
            for k in range(grid.nn_DC):
                Y = grid.Ybus_DC[i, k]

                if k != i:
                    if Y != 0:
                        line = grid.get_lineDC_by_nodes(i, k)
                        pol = line.pol
                    Y = -Y
                    P[i] += pol*V[i]*(V[i]-V[k])*Y

        for node in grid.nodes_DC:
            if grid.nconv != 0:
                if node.type == 'Droop':
                    n = node.nodeNumber
                    Droop_change = (node.V_ini-V[n])*node.Droop_rate
                    P_known[n] = np.copy(
                        grid.P_DC[n]+grid.Pconv_DC[n]) + Droop_change
                    s = 1

        # print (P1)
        # print (P)
        # print('------')
        dPa = P_known-P

        Jacobian_DC(grid,V, P)

        if len(grid.slack_bus_number_DC) == 0:
            J_modified = grid.J_DC
        else:
            J_modified = np.delete(
                np.delete(grid.J_DC, grid.slack_bus_number_DC, 0), grid.slack_bus_number_DC, 1)
            dPa = np.delete(dPa, grid.slack_bus_number_DC, 0)

        dV_V = np.linalg.solve(J_modified, dPa)

        # Recall the updated voltage vector into the correct place
        k = 0  # Index for dV vector
        for i in range(grid.nn_DC):
            if grid.nodes_DC[i].type != 'Slack':
                dV = dV_V[k].item()*V[i]
                V[i] += dV
                k += 1  # Move to the next element in dV
        tol = max(abs(dPa))
        # print(f"Iteration {iter_num}, Max Voltage Change: {max(abs(dV))}, tolerance: {tol}")

        if iter_num == maxIter:
            print('')
            print(f'Warning  load flow DC did not converge')
            print(f'Lowest tolerance reached: {np.round(tol,decimals=6)}')

    grid.iter_flow_DC.append(iter_num)

    grid.V_DC = V

    for node in grid.nodes_DC:

        i = node.nodeNumber
        for k in range(grid.nn_DC):
            Y = grid.Ybus_DC[i, k]
            if k != i:
                if Y != 0:
                    line = grid.get_lineDC_by_nodes(i, k)
                    pol = line.pol
                Y = -Y
                Pf[i] += pol*V[i]*(V[i]-V[k])*Y
                grid.nodes_DC[i].V = V[i]
    dPa = P_known-Pf

    for node in grid.nodes_DC:
        n = node.nodeNumber
        node.P_INJ = Pf[n].item()
        node.P = P_known[n].item()-grid.P_DC[n].item()
        if node.type == 'Slack':
            node.P = Pf[n].item()-grid.P_DC[n].item()

    s = 1

    grid.P_DC_INJ = np.vstack([node.P_INJ for node in grid.nodes_DC])

    if grid.nconv != 0:

        for conv in grid.Converters_ACDC:
            n = conv.Node_DC.nodeNumber
            if conv.type == 'Droop':

                conv.P_DC = P_known[n].item()-grid.P_DC[n].item()
                s = 1
            elif conv.type == 'Slack':
                conv.Node_DC.Pconv = Pf[n].item()-grid.P_DC[n].item()
                conv.P_DC = Pf[n].item()-grid.P_DC[n].item()
                s = 1
    grid.Update_P_DC()
    s = 1

def Jacobian_AC(grid, Voltages, Angles):
    grid.slack_bus_number_AC = []
    # DOES NOT ACCEPT PV NODES YET
    V = Voltages
    th = Angles
    # Derivate of P in respect to Theta

    J_11 = np.zeros((grid.nn_AC, grid.nn_AC))

    # here m and n should be modified to include PV nodes  later on

    for i in range(grid.nn_AC):
        m = grid.nodes_AC[i].nodeNumber
        if grid.nodes_AC[i].type != 'Slack':
            for k in range(grid.nn_AC):
                n = grid.nodes_AC[k].nodeNumber
                G = np.real(grid.Ybus_AC[m, n])
                B = np.imag(grid.Ybus_AC[m, n])
                if m == n:
                    # = -Q - B * V^2
                    J_11[i, k] = -grid.Q[m]-V[m]**2*B

                else:
                    # Vm*Vn* (G sin(θ_mn)-B cos(θ_mn)
                    J_11[i, k] = V[m]*V[n] * \
                        (G*np.sin(th[m]-th[n])-B*np.cos(th[m]-th[n]))
        else:
            grid.slack_bus_number_AC.append(m)
    J_11 = np.delete(
        np.delete(J_11, grid.slack_bus_number_AC, 0), grid.slack_bus_number_AC, 1)

    J_12 = np.zeros((grid.nn_AC, grid.npq))

    # here m and n should be modified to include PV nodes  later on

    for i in range(grid.nn_AC):
        m = grid.nodes_AC[i].nodeNumber
        for k in range(grid.npq):
            n = grid.pq_nodes[k].nodeNumber
            G = np.real(grid.Ybus_AC[m, n])
            B = np.imag(grid.Ybus_AC[m, n])
            if m == n:
                # P/V + G * V
                J_12[i, k] = grid.P[m]/V[m] + G*V[m]

            else:
                # V * (G cos θ_mn + B sin θ_mn)
                J_12[i, k] = V[m] * \
                    (G*np.cos(th[m]-th[n])+B*np.sin(th[m]-th[n]))

    J_12 = np.delete(J_12, grid.slack_bus_number_AC, 0)

    J_21 = np.zeros((grid.npq, grid.nn_AC))

    # here m and n should be modified to include PV nodes  later on

    for i in range(grid.npq):
        m = grid.pq_nodes[i].nodeNumber
        for k in range(grid.nn_AC):
            n = grid.nodes_AC[k].nodeNumber
            G = np.real(grid.Ybus_AC[m, n])
            B = np.imag(grid.Ybus_AC[m, n])
            if m == n:
                # P - G V^2
                J_21[i, k] = grid.P[m]-V[m]**2*G

                s = 1
            else:
                # - Vi *Vk *( G sin θ_mn + B cos θ_mn)
                J_21[i, k] = -V[m]*V[n] * \
                    (G*np.cos(th[m]-th[n])+B*np.sin(th[m]-th[n]))
    J_21 = np.delete(J_21, grid.slack_bus_number_AC, 1)

    J_22 = np.zeros((grid.npq, grid.npq))
    # here m and n should be modified to include PV nodes  later on

    for i in range(grid.npq):
        m = grid.pq_nodes[i].nodeNumber
        for k in range(grid.npq):
            n = grid.pq_nodes[k].nodeNumber
            G = np.real(grid.Ybus_AC[m, n])
            B = np.imag(grid.Ybus_AC[m, n])
            if m == n:
                # Q /V - B*V
                J_22[i, k] = grid.Q[m]/V[m]-B*V[m]
                s = 1
            else:
                # V *(G sin θ_mn - B cos θ_mn )
                J_22[i, k] = V[m] * \
                    (G*np.sin(th[m]-th[n])-B*np.cos(th[m]-th[n]))
    grid.J_AC = np.vstack(
        (np.hstack((J_11, J_12)), np.hstack((J_21, J_22))))

    f = 1

def load_flow_AC(grid, tol_lim=1e-4, maxIter=20):

    Pnet = np.copy(grid.P_AC+grid.Ps_AC)
    Qnet = np.copy(grid.Q_AC+grid.Qs_AC)

    angles = np.zeros(grid.nn_AC)
    V = np.zeros(grid.nn_AC)

    # number of different node types

    npv = len(grid.pv_nodes)
    npq = len(grid.pq_nodes)
    nps = len(grid.slack_nodes)

    for node in grid.nodes_AC:
        V[node.nodeNumber] = node.V
        angles[node.nodeNumber] = node.theta

    tol = 1
    iter_num = 0
    while tol > tol_lim and iter_num < maxIter:
        iter_num += 1

        P = np.zeros((grid.nn_AC, 1))
        Q = np.zeros((grid.nn_AC, 1))

        for node in grid.nodes_AC:
            i = node.nodeNumber
            for k in range(grid.nn_AC):
                G = np.real(grid.Ybus_AC[i, k])
                B = np.imag(grid.Ybus_AC[i, k])
                P[i] += V[i]*V[k] * \
                    (G*np.cos(angles[i]-angles[k]) +
                     B*np.sin(angles[i]-angles[k]))
                Q[i] += V[i]*V[k] * \
                    (G*np.sin(angles[i]-angles[k]) -
                     B*np.cos(angles[i]-angles[k]))

        # Power flow vector solve
        grid.P = P
        grid.Q = Q

        # Calculate changes in specified active and reactive power
        dPa = Pnet-P
        dQa = Qnet-Q
        k = 0

        Jacobian_AC(grid,V, angles)

        Q_del = []
        for node in grid.nodes_AC:
            i = node.nodeNumber
            if node.type != 'PQ':
                Q_del.append(i)

        dP = np.delete(dPa, grid.slack_bus_number_AC, axis=0)
        dQ = np.delete(dQa, Q_del, axis=0)

        M = np.vstack((dP, dQ))

        X = np.linalg.solve(grid.J_AC, M)

        # Check for NaN values in the array
        nan_indices = np.isnan(X)

        # Get the indices of NaN values
        nan_indices = np.where(nan_indices)[0]

        if nan_indices.size > 0:
            print("Linear results not avialable for AC PF")
            sys.exit()
        dTh = X[0:(grid.nn_AC-nps)]
        dV = X[grid.nn_AC-nps:]
        # dTh = np.array((1,2,3,4,5))
        # dV= np.array((1,2,3,4,5))

        # Recall the updated voltage vector into the correct place
        k = 0  # Index for dV vector
        for i in range(grid.nn_AC):
            if grid.nodes_AC[i].type != 'Slack':
                s = 1
                grid.nodes_AC[i].theta += dTh[k].item()
                angles[i] += dTh[k].item()
                k += 1  # Move to the next element in dTh
        k = 0  # Index for dV vector
        for i in range(grid.nn_AC):
            if grid.nodes_AC[i].type == 'PQ':
                grid.nodes_AC[i].V += dV[k].item()
                V[i] += dV[k].item()
                k += 1  # Move to the next element in dV

        # for node in grid.nodes:
        #      V[node.nodeNumber]= node.V_iter
        #      angles[node.nodeNumber] = node.theta_iter

        tol = max(abs(M))
        if iter_num == maxIter:
            print('')
            print(f'Warning  load flow AC did not converge')
            print(f'Lowest tolerance reached: {np.round(tol,decimals=6)}')

    grid.iter_flow_AC.append(iter_num)

    grid.V_AC = V
    grid.Theta_V_AC = angles

    grid.voltage_violation = 0
    Diff = np.abs(V-1)

    grid.dif = max(Diff)

    if grid.dif > 0.11:
        grid.voltage_violation = 1

    V_violation = grid.voltage_violation
    Pf = np.zeros((grid.nn_AC, 1))
    Qf = np.zeros((grid.nn_AC, 1))
    for node in grid.nodes_AC:
        i = node.nodeNumber
        for k in range(grid.nn_AC):
            G = np.real(grid.Ybus_AC[i, k])
            B = np.imag(grid.Ybus_AC[i, k])
            Pf[i] += V[i]*V[k] * \
                (G*np.cos(angles[i]-angles[k]) +
                 B*np.sin(angles[i]-angles[k]))
            Qf[i] += V[i]*V[k] * \
                (G*np.sin(angles[i]-angles[k]) -
                 B*np.cos(angles[i]-angles[k]))
    Sf = Pf+1j*Qf

    for node in grid.nodes_AC:
        i = node.nodeNumber
        node.P_INJ = Pf[i].item()
        node.Q_INJ = Qf[i].item()
    grid.P_AC_INJ = np.vstack([node.P_INJ for node in grid.nodes_AC])
    grid.Q_INJ = np.vstack([node.Q_INJ for node in grid.nodes_AC])

def flow_conv_P_AC(grid, conv):
    Us = conv.Node_AC.V.item()
    th_s = conv.Node_AC.theta.item()

    P_AC = conv.P_AC
    Q_AC = conv.Q_AC

    Ztf = conv.Ztf
    Zc = conv.Zc
    Zf = conv.Zf
    Bf = conv.Bf

    Yeq = 1/(Zc+Ztf)
    Us_cart = pol2cartz(Us, th_s)
    Ss_cart = P_AC+1j*Q_AC

    Is = np.conj(Ss_cart/Us_cart)

    Uf_cart = Us_cart+Ztf*Is

    if Zf != 0:
        Ic_cart = Us_cart/Zf+Is*(Zf+Ztf)/Zf
    else:
        Ic_cart = Is
    Uc_cart = Uf_cart+Zc*Ic_cart

    [Uc, th_c] = cartz2pol(Uc_cart)
    [Uf, th_f] = cartz2pol(Uf_cart)
    [Ic, th_Ic] = cartz2pol(Ic_cart)

    Gc = conv.Gc
    Bc = conv.Bc
    Gtf = conv.Gtf
    Btf = conv.Btf

    Sc = Uc_cart*np.conj(Ic_cart)

    if Zf == 0:
        # CHECK THIS
        Ps1 = -Us*Us*Gc+Us*Uc*(Gc*np.cos(th_s-th_c)+Bc*np.sin(th_s-th_c))
        Qs1 = Us*Us*Bc+Us*Uc*(Gc*np.sin(th_s-th_c)-Bc*np.cos(th_s-th_c))

        Pc1 = Uc*Uc*Gc-Us*Uc*(Gc*np.cos(th_s-th_c)-Bc*np.sin(th_s-th_c))
        Qc1 = -Uc*Uc*Bc+Us*Uc*(Gc*np.sin(th_s-th_c)+Bc*np.cos(th_s-th_c))
    else:
        Ps1 = -Us*Us*Gtf+Us*Uf * \
            (Gtf*np.cos(th_s-th_f)+Btf*np.sin(th_s-th_f))
        Qs1 = Us*Us*Btf+Us*Uf*(Gtf*np.sin(th_s-th_f)-Btf*np.cos(th_s-th_f))

        Pc1 = Uc*Uc*Gc-Uf*Uc*(Gc*np.cos(th_f-th_c)-Bc*np.sin(th_f-th_c))
        Qc1 = -Uc*Uc*Bc+Uf*Uc*(Gc*np.sin(th_f-th_c)+Bc*np.cos(th_f-th_c))

        Psf1 = Uf*Uf*Gtf-Uf*Us * \
            (Gtf*np.cos(th_s-th_f)-Btf*np.sin(th_s-th_f))
        Qsf1 = -Uf*Uf*Btf+Uf*Us * \
            (Gtf*np.sin(th_s-th_f)+Btf*np.cos(th_s-th_f))

        Pcf1 = -Uf*Uf*Gc+Uf*Uc*(Gc*np.cos(th_f-th_c)+Bc*np.sin(th_f-th_c))
        Qcf1 = Uf*Uf*Bc+Uf*Uc*(Gc*np.sin(th_f-th_c)-Bc*np.cos(th_f-th_c))

        Qf1 = -Uf*Uf*Bf

    Pc = np.real(Sc)

    Ic1 = np.sqrt(Pc1*Pc1+Qc1*Qc1)/Uc

    # AC to DC
    if conv.P_AC > 0:  # DC to AC
        P_loss = conv.a_conv+conv.b_conv*Ic+conv.c_inver*Ic*Ic
    else:  # AC to DC
        P_loss = conv.a_conv+conv.b_conv*Ic+conv.c_rect*Ic*Ic

    P_DC = -Pc-P_loss

    conv.P_loss = P_loss
    conv.P_DC = P_DC
    conv.U_f = Uf
    conv.U_c = Uc
    conv.Node_DC.Pconv = P_DC
    conv.Node_AC.P_s = P_AC
    conv.Ic = Ic

    if conv.name == 'CmA1':
        s = 1

def Jacobian_conv_notransformer(grid, conv, U_c, Pc, Qc, Ps, Qs):
    J_conv = np.zeros((2, 2))

    # dPc/dTheta_c
    J_conv[0, 0] = -Qc-conv.Bc*U_c*U_c

    # U_C*dPc/dUc
    J_conv[0, 1] = Pc+conv.Gc*U_c*U_c

    # dQs/dThetac
    J_conv[1, 0] = -Ps-conv.Gc*conv.U_s*conv.U_s

    # Uc*dQs/dU_c
    J_conv[1, 1] = Qs-(conv.Bf+conv.Bc)*conv.U_s*conv.U_s

    return J_conv

def Jacobian_conv_no_Filter(grid, conv, U_c, Pc, Qc, Ps, Qs):
    J_conv = np.zeros((2, 2))

    # dPc/dTheta_c
    J_conv[0, 0] = -Qc-conv.Bc*U_c*U_c

    # U_C*dPc/dUc
    J_conv[0, 1] = Pc+conv.Gc*U_c*U_c

    # dQs/dThetac
    J_conv[1, 0] = -Ps-conv.Gc*conv.U_s*conv.U_s

    # Uc*dQs/dU_c
    J_conv[1, 1] = Qs-conv.Bc*conv.U_s*conv.U_s

    return J_conv

def Jacobian_conv(grid, conv, Qcf, Qsf, Pcf, Psf, U_f, U_c, Pc, Qc, Ps, Qs):
    J_conv = np.zeros((4, 4))

    # dPc/dTheta_c
    J_conv[0, 0] = -Qc-conv.Bc*U_c*U_c

    # dPc/dTheta_f
    J_conv[0, 1] = Qc+conv.Bc*U_c*U_c

    # U_C*dPc/dUc
    J_conv[0, 2] = Pc+conv.Gc*U_c*U_c

    # U_f*dPc/dUf
    J_conv[0, 3] = Pc-conv.Gc*U_c*U_c

    # dQs/dThetaf
    J_conv[1, 1] = -Ps-conv.Gtf*conv.U_s*conv.U_s

    # Uf*dQs/dU_f
    J_conv[1, 3] = Qs-conv.Btf*conv.U_s*conv.U_s

    # dF1/dTheta c
    J_conv[2, 0] = Qcf-conv.Bc*U_f*U_f

    # dF1/dTheta f
    J_conv[2, 1] = -Qcf+Qsf+(conv.Bc+conv.Btf)*U_f*U_f

    #Uc *dF1/dUc
    J_conv[2, 2] = Pcf+conv.Gc*U_f*U_f

    # Uf*dF1/dUf
    J_conv[2, 3] = Pcf-Psf-(conv.Gc+conv.Gtf)*U_f*U_f

    # dF2/dTheta c
    J_conv[3, 0] = -Pcf-conv.Gc*U_f

    # dF2/dTheta f
    J_conv[3, 1] = Pcf-Psf+(conv.Gc+conv.Gtf)*U_f*U_f

    #Uc *dF2/dUc
    J_conv[3, 2] = Qcf-conv.Bc*U_f*U_f

    # Uf*dF2/dUf
    J_conv[3, 3] = Qcf-Qsf+(conv.Bc+conv.Btf+2*conv.Bf)*U_f*U_f

    return J_conv

def flow_conv(grid, conv, tol_lim=1e-14, maxIter=20):

    if conv.Bf == 0:
        flow_conv_no_filter(grid,conv, tol_lim, maxIter)

    elif conv.Gtf == 0:

        flow_conv_no_transformer(grid,conv, tol_lim, maxIter)

    else:

        flow_conv_complete(grid,conv, tol_lim, maxIter)

def flow_conv_no_filter(grid, conv, tol_lim, maxIter):

    Ztf = conv.Ztf
    Zc = conv.Zc

    Zeq = Ztf+Zc
    Yeq = 1/Zeq

    Uc = conv.U_c
    Gc = np.real(Yeq)
    Bc = np.imag(Yeq)

    th_c = conv.th_c

    Pc_known = -np.copy(conv.P_DC)
    Qs_known = conv.Q_AC
    Us = conv.U_s.item()
    th_s = conv.th_s.item()

    tol2 = 1

    while tol2 > tol_lim:
        tol = 1
        iter_num = 0
        while tol > tol_lim and iter_num < maxIter:
            iter_num += 1

            Ps = -Us*Us*Gc+Us*Uc * \
                (Gc*np.cos(th_s-th_c)+Bc*np.sin(th_s-th_c))
            Qs = Us*Us*Bc+Us*Uc*(Gc*np.sin(th_s-th_c)-Bc*np.cos(th_s-th_c))

            Pc = Uc*Uc*Gc-Us*Uc*(Gc*np.cos(th_s-th_c)-Bc*np.sin(th_s-th_c))
            Qc = -Uc*Uc*Bc+Us*Uc * \
                (Gc*np.sin(th_s-th_c)+Bc*np.cos(th_s-th_c))

            J_conv = Jacobian_conv_no_Filter(grid,conv, Uc, Pc, Qc, Ps, Qs)

            dPc = Pc_known-Pc
            dQs = Qs_known-Qs

            M = np.array([dPc, dQs])

            X = np.linalg.solve(J_conv, M)

            th_c += X[0].item()

            Uc += X[1].item()*Uc

            tol = max(abs(M))

        Pc = Uc*Uc*Gc-Us*Uc*(Gc*np.cos(th_s-th_c)-Bc*np.sin(th_s-th_c))
        Qc = -Uc*Uc*Bc+Us*Uc*(Gc*np.sin(th_s-th_c)+Bc*np.cos(th_s-th_c))

        if iter_num > maxIter:
            print('')
            print(f'Warning  converter {conv.name} did not converge')
            print(f'Lowest tolerance reached: {np.round(tol,decimals=6)}')

        Ic = np.sqrt(Pc*Pc+Qc*Qc)/Uc

        if conv.P_DC < 0:  # DC to AC
            P_loss = conv.a_conv+conv.b_conv*Ic+conv.c_inver*Ic*Ic
        else:  # AC to DC
            P_loss = conv.a_conv+conv.b_conv*Ic+conv.c_rect*Ic*Ic

        Pc_new = -conv.P_DC-P_loss

        tol2 = abs(Pc_known-Pc_new)

        Pc_known = np.copy(Pc_new)

        if conv.name == 'CmE1':
            s = 1

    Ps = -Us*Us*Gc+Us*Uc*(Gc*np.cos(th_s-th_c)+Bc*np.sin(th_s-th_c))
    Qs = Us*Us*Bc+Us*Uc*(Gc*np.sin(th_s-th_c)-Bc*np.cos(th_s-th_c))

    Pc = Uc*Uc*Gc-Us*Uc*(Gc*np.cos(th_s-th_c)-Bc*np.sin(th_s-th_c))
    Qc = -Uc*Uc*Bc+Us*Uc*(Gc*np.sin(th_s-th_c)+Bc*np.cos(th_s-th_c))

    conv.P_AC = Ps
    conv.Q_AC = Qs
    conv.Node_AC.P_s_new = Ps
    conv.Pc = Pc
    conv.Qc = Qc
    Ps_old = conv.Node_AC.P_s
    conv.P_loss = P_loss
    conv.P_loss_tf = abs(Ps-Pc)
    n = conv.Node_AC.nodeNumber
    conv.Node_AC.P_s_new += Ps
    grid.Ps_AC_new[n] += Ps

    if conv.name == 'CbA1':
        s = 1

def flow_conv_no_transformer(grid, conv, tol_lim, maxIter):
    Uc = conv.U_c
    Gc = conv.Gc

    Bc = conv.Bc
    th_f = conv.th_f
    th_c = conv.th_c
    Gtf = conv.Gtf
    Btf = conv.Btf
    Bf = conv.Bf

    Pc_known = -np.copy(conv.P_DC)
    Qs_known = np.copy(conv.Q_AC)
    Us = conv.U_s.item()
    th_s = conv.th_s.item()

    tol2 = 1

    while tol2 > tol_lim:
        tol = 1
        iter_num = 0
        while tol > tol_lim and iter_num < maxIter:
            iter_num += 1
            Bcf = Bc+Bf

            Ps = -Us*Us*Gc+Us*Uc * \
                (Gc*np.cos(th_s-th_c)+Bc*np.sin(th_s-th_c))
            Qs = Us*Us*Bcf+Us*Uc * \
                (Gc*np.sin(th_s-th_f)-Bc*np.cos(th_s-th_c))

            Pc = Uc*Uc*Gc-Us*Uc*(Gc*np.cos(th_s-th_c)-Bc*np.sin(th_s-th_c))
            Qc = -Uc*Uc*Bc+Us*Uc * \
                (Gc*np.sin(th_s-th_c)+Bc*np.cos(th_s-th_c))

            J_conv = Jacobian_conv_notransformer(grid,conv, Uc, Pc, Qc, Ps, Qs)

            dPc = Pc_known-Pc
            dQs = Qs_known-Qs

            M = np.array([dPc, dQs])

            X = np.linalg.solve(J_conv, M)

            th_c += X[0].item()
            Uc += X[1].item()*Uc

            tol = max(abs(M))

        Pc = Uc*Uc*Gc-Us*Uc*(Gc*np.cos(th_s-th_c)-Bc*np.sin(th_s-th_c))
        Qc = -Uc*Uc*Bc+Us*Uc*(Gc*np.sin(th_s-th_c)+Bc*np.cos(th_s-th_c))

        if iter_num > maxIter:
            print('')
            print(f'Warning  converter {conv.name} did not converge')
            print(f'Lowest tolerance reached: {np.round(tol,decimals=6)}')
            print(f'Lowest tolerance reached: {np.round(tol,decimals=6)}')

        Ic = np.sqrt(Pc*Pc+Qc*Qc)/Uc

        if conv.P_DC < 0:  # DC to AC
            P_loss = conv.a_conv+conv.b_conv*Ic+conv.c_inver*Ic*Ic
        else:  # AC to DC
            P_loss = conv.a_conv+conv.b_conv*Ic+conv.c_rect*Ic*Ic

        Pc_new = -conv.P_DC-P_loss

        tol2 = abs(Pc_known-Pc_new)

        Pc_known = np.copy(Pc_new)

        s = 1

    Ps = -Us*Us*Gc+Us*Uc*(Gc*np.cos(th_s-th_c)+Bc*np.sin(th_s-th_c))
    Qs = Us*Us*Bcf+Us*Uc*(Gc*np.sin(th_s-th_f)-Bc*np.cos(th_s-th_c))

    Pc = Uc*Uc*Gc-Us*Uc*(Gc*np.cos(th_s-th_c)-Bc*np.sin(th_s-th_c))
    Qc = -Uc*Uc*Bc+Us*Uc*(Gc*np.sin(th_s-th_c)+Bc*np.cos(th_s-th_c))

    conv.P_AC = Ps
    conv.Q_AC = Qs
    conv.Node_AC.P_s_new = Ps
    conv.Pc = Pc
    conv.Qc = Qc

    conv.P_loss = P_loss
    conv.P_loss_tf = abs(Ps-Pc)
    conv.Node_AC.P_s_new += Ps
    grid.Ps_AC_new[conv.Node_AC.nodeNumber] += Ps

    if conv.name == 'CbA1':
        s = 1

def flow_conv_complete(grid, conv, tol_lim, maxIter):
    Uc = conv.U_c
    Gc = conv.Gc
    Uf = conv.U_f
    Bc = conv.Bc
    th_f = conv.th_f
    th_c = conv.th_c
    Gtf = conv.Gtf
    Btf = conv.Btf
    Bf = conv.Bf

    Pc_known = -np.copy(conv.P_DC)
    Qs_known = conv.Q_AC
    Us = conv.U_s.item()
    th_s = conv.th_s.item()

    tol2 = 1

    while tol2 > tol_lim:
        tol = 1
        iter_num = 0
        while tol > tol_lim and iter_num < maxIter:
            iter_num += 1

            Ps = -Us*Us*Gtf+Us*Uf * \
                (Gtf*np.cos(th_s-th_f)+Btf*np.sin(th_s-th_f))
            Qs = Us*Us*Btf+Us*Uf * \
                (Gtf*np.sin(th_s-th_f)-Btf*np.cos(th_s-th_f))

            Pc = Uc*Uc*Gc-Uf*Uc*(Gc*np.cos(th_f-th_c)-Bc*np.sin(th_f-th_c))
            Qc = -Uc*Uc*Bc+Uf*Uc * \
                (Gc*np.sin(th_f-th_c)+Bc*np.cos(th_f-th_c))

            Psf = Uf*Uf*Gtf-Uf*Us * \
                (Gtf*np.cos(th_s-th_f)-Btf*np.sin(th_s-th_f))
            Qsf = -Uf*Uf*Btf+Uf*Us * \
                (Gtf*np.sin(th_s-th_f)+Btf*np.cos(th_s-th_f))

            Pcf = -Uf*Uf*Gc+Uf*Uc * \
                (Gc*np.cos(th_f-th_c)+Bc*np.sin(th_f-th_c))
            Qcf = Uf*Uf*Bc+Uf*Uc * \
                (Gc*np.sin(th_f-th_c)-Bc*np.cos(th_f-th_c))

            Qf = -Uf*Uf*Bf

            J_conv = Jacobian_conv(grid,conv, Qcf, Qsf, Pcf, Psf, Uf, Uc, Pc, Qc, Ps, Qs)

            F1 = Pcf-Psf
            F2 = Qcf-Qsf-Qf
            dPc = Pc_known-Pc
            dQs = Qs_known-Qs

            M = np.array([dPc, dQs, -F1, -F2])

            X = np.linalg.solve(J_conv, M)

            th_c += X[0].item()
            th_f += X[1].item()
            Uc += X[2].item()*Uc
            Uf += X[3].item()*Uf

            tol = max(abs(M))

        Pc = Uc*Uc*Gc-Uf*Uc*(Gc*np.cos(th_f-th_c)-Bc*np.sin(th_f-th_c))
        Qc = -Uc*Uc*Bc+Uf*Uc*(Gc*np.sin(th_f-th_c)+Bc*np.cos(th_f-th_c))

        if iter_num > maxIter:
            print('')
            print(f'Warning  converter {conv.name} did not converge')
            print(f'Lowest tolerance reached: {np.round(tol,decimals=6)}')

        Ic = np.sqrt(Pc*Pc+Qc*Qc)/Uc

        if conv.P_DC < 0:  # DC to AC
            P_loss = conv.a_conv+conv.b_conv*Ic+conv.c_inver*Ic*Ic
        else:  # AC to DC
            P_loss = conv.a_conv+conv.b_conv*Ic+conv.c_rect*Ic*Ic

        Pc_new = -conv.P_DC-P_loss

        tol2 = abs(Pc_known-Pc_new)

        Pc_known = np.copy(Pc_new)

    Ps = -Us*Us*Gtf+Us*Uf*(Gtf*np.cos(th_s-th_f)+Btf*np.sin(th_s-th_f))
    Qs = Us*Us*Btf+Us*Uf*(Gtf*np.sin(th_s-th_f)-Btf*np.cos(th_s-th_f))
    # Pc=  Uc*Uc*Gc-Uf*Uc*(Gc*np.cos(th_f-th_c)-Bc*np.sin(th_f-th_c))
    # Qc= -Uc*Uc*Bc+Uf*Uc*(Gc*np.sin(th_f-th_c)+Bc*np.cos(th_f-th_c))
    # CHECK THIs
    conv.P_AC = Ps
    conv.Q_AC = Qs
    conv.Node_AC.P_s_new = Ps
    conv.Pc = Pc
    conv.Qc = Qc

    conv.U_c = Uc
    conv.U_f = Uf

    conv.P_loss = P_loss
    conv.P_loss_tf = abs(Ps-Pc)
    conv.Ic = Ic
    conv.Node_AC.P_s_new += Ps
    grid.Ps_AC_new[conv.Node_AC.nodeNumber] += Ps

def Converter_Qlimit(grid, conv):

    Us = conv.Node_AC.V
    th_s = conv.Node_AC.theta

    conj_Ztf = np.conj(conv.Ztf)
    conj_Zc = np.conj(conv.Zc)
    conj_Zf = np.conj(conv.Zf)

    MVA_max = conv.MVA_max
    Icmax = MVA_max

    Ps = conv.P_AC

    S0 = 0
    S0v = 0
    Y1 = 0
    if conv.Z1 != 0:
        Y1 = 1/conv.Z1

    if conv.Zf != 0:
        r = Us*Icmax*np.abs(conj_Zf/(conj_Zf+conj_Ztf))

        S0 = -Us**2*(1/(conj_Zf+conj_Ztf))
        Y2 = 1/conv.Z2
        S0v = -Us**2*(np.conj(Y1)+np.conj(Y2))
        rVmin = Us*conv.Ucmin*np.abs(Y2)
        rVmax = Us*conv.Ucmax*np.abs(Y2)
    else:
        r = Us*Icmax
        S0v = -Us**2*(1/(conj_Ztf+conj_Zc))
        rVmin = Us*conv.Ucmin*np.abs(1/(conj_Ztf+conj_Zc))
        rVmax = Us*conv.Ucmax*np.abs(1/(conj_Ztf+conj_Zc))

    Q0 = np.imag(S0)
    Q0V = np.imag(S0v)

    Po = np.real(S0)

    sqrt = r**2-(Ps-Po)**2
    if sqrt < 0:
        print(f'Converter {conv.name} is over current capacity')

    Qs_plus = Q0+np.sqrt(r**2-(Ps-Po)**2)
    Qs_minus = Q0-np.sqrt(r**2-(Ps-Po)**2)

    Qs_plusV = Q0V+np.sqrt(rVmax**2-(Ps-Po)**2)
    Qs_minusV = Q0V+np.sqrt(rVmin**2-(Ps-Po)**2)

    Qs_max = min(Qs_plus, Qs_plusV)
    Qs_min = max(Qs_minus, Qs_minusV)

    name = conv.name

    conv.Node_AC.Q_min = Qs_minus
    conv.Node_AC.Q_max = Qs_plus

    AC_node = conv.Node_AC.nodeNumber

    if conv.AC_type == 'PV' or conv.AC_type == 'Slack':
        conv.Node_AC.Q_s = (grid.Q_INJ[AC_node]-grid.Q_AC[AC_node]).item()
        Q_req = conv.Node_AC.Q_s
    else:
        Q_req = conv.Q_AC
        # conv.Node_AC.Q_s= (grid.Q_INJ[AC_node]-grid.Q_AC[AC_node]).item()-Q_req

    if Q_req > Qs_max or Q_req < Qs_min:

        print('-----------')
        print(f'{conv.name}  CONVERTER LIMIT circle REACHED')
        print('-----------')
        if conv.Node_AC.type == 'Slack':
            print(f' Limiting Q from converter')
            print(
                f' External reactive compensation needed at node {conv.Node_AC.name}')
            if Q_req > Qs_plus:
                conv.Node_AC.Q_s = Qs_max
                conv.Node_AC.Q_AC = Q_req-Qs_max
            elif Q_req < Qs_minus:
                conv.Node_AC.Q_s = Qs_min
                conv.Node_AC.Q_AC = Q_req-Qs_min
                s = 1

        else:
            print(
                f'Limiting Q from converter and changing AC node {conv.Node_AC.name} to PQ')
            conv.Node_AC.type = 'PQ'
            if Q_req > Qs_max:
                conv.Node_AC.Q_s = Qs_max
            elif Q_req < Qs_min:
                conv.Node_AC.Q_s = Qs_min
        conv.AC_type = 'PQ'
        grid.npv = len(grid.pv_nodes)
        grid.npq = len(grid.pq_nodes)

    if conv.name == 'CbC2':
        s = 1
