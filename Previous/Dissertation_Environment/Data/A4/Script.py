import sympy as sp 
import numpy as np 
import matplotlib.pyplot as plt
from copy import deepcopy

def calculate_derivatives():
    x, y = sp.symbols('x y', real=True)
    # The First Function

    f1 = (1 * ((x - 1) ** 4)) + (8 * ((y - 1) ** 2))
    df1_x = sp.diff(f1, x)
    df1_y = sp.diff(f1, y)
    print(f1)
    print(df1_x)
    print(df1_y)
    # The second Function

    f2 = sp.Max(x - 1, 0) + (8 * sp.Abs(y - 1))
    df2_x = sp.diff(f2, x)
    df2_y = sp.diff(f2, y)
    print(f2)
    print(df2_x)
    print(df2_y)

def polyak(df,x0,f,epsilon = 1e-8):
    # standard code 
    x = deepcopy(x0)
    n = len(df)
    x_list, f_list, step_list = [deepcopy(x)], [f(*x)],[]

    #report
    for _ in range(150):
        count = 0
        for i in range(n):
            count = sum(df[i](x[i])**2)
        step = f(*x) / (count + epsilon)
        for i in range(n):
            x[i] -= step * df[i](x[i])
        x_list.append(deepcopy(x))
        f_list.append(f(*x))
        step_list.append(step)
    return x_list, f_list, step_list

def RMSprop(f, df, x0, parameters, iterations=100):
    alpha0, beta = parameters
    x = deepcopy(x0)
    n = len(df)
    x_list, f_list ,step_list = [deepcopy(x)], [f(*x)], [[alpha0] * n]
    # defining the constant parameter of epsilon to account for division by zero 
    epsilon = 1e-8
    sums = [0] * n
    alphas = [alpha0] * n
    for _ in range(iterations):
        for i in range(n):
            x[i] -= alphas[i] * df[i](x[i])
            sums[i] = (beta * sums[i]) + ((1 - beta) * (df[i](x[i]) ** 2))
            alphas[i] = alpha0 / ((sums[i] ** 0.5) + epsilon)
        x_list.append(deepcopy(x))
        f_list.append(f(*x))
        step_list.append(deepcopy(alphas))
    return x_list, f_list ,step_list

def HeavyBall(f, df, x0, parameters, iterations=100):
    alpha, beta = parameters
    x = deepcopy(x0)
    n = len(df)
    x_list, f_list, step_list = [deepcopy(x)], [f(*x)], [0]
    # defining the constant parameter of epsilon to account for division by zero 
    epsilon = 1e-8
    z = 0
    for _ in range(iterations):
        
        z = (beta * z) + (alpha * f(*x) / (sum(df[j](x[j]) ** 2 for j in range(n)) + epsilon))
        for i in range(n):
            x[i] -= z * df[i](x[i])
        x_list.append(deepcopy(x))
        f_list.append(f(*x))
        step_list.append(z)
    return x_list, f_list, step_list

def Adam(f,df,x0,parameters,iterations=100):
    alpha,beta1,beta2 = parameters
    x = deepcopy(x0)
    n = len(df)
    x_list, f_list, step_list = [deepcopy(x)], [f(*x)], [[0] * n]
    # defining the constant parameter of epsilon to account for division by zero 
    epsilon = 1e-8
    ms = [0] * n
    vs = [0] * n
    step = [0] * n
    t = 0
    for _ in range(iterations):
        t += 1
        for i in range(n):
            ms[i] = (beta1 * ms[i]) + ((1 - beta1) * df[i](x[i]))
            vs[i] = (beta2 * vs[i]) + ((1 - beta2) * (df[i](x[i]) ** 2))
            m_hat = ms[i] / (1 - (beta1 ** t))
            v_hat = vs[i] / (1 - (beta2 ** t))
            step[i] = alpha * (m_hat / ((v_hat ** 0.5) + epsilon))
            x[i] -= step[i]
        x_list.append(deepcopy(x))
        f_list.append(f(*x))
        step_list.append(deepcopy(step))
    return x_list, f_list, step_list

def B1(f, df, x, fnum):
    alpha0s = [0.001, 0.01, 0.1]
    betas = [0.25, 0.9]
    iterations = 200
    iters = list(range(iterations + 1))
    
    plt.figure(facecolor='black')
    plt.xlabel('Iterations', color='white')
    plt.ylabel(f'$f_{fnum}(x,y)$', color='white')
    plt.title(f'RMSProp for $f_{fnum}(x,y)$', color='white')
    plt.grid(color='gray')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    
    all_values = []
    
    for alpha0 in alpha0s:
        for beta in betas:
            xs, values, steps = RMSprop(f, df, x, [alpha0, beta], iterations=iterations)
            plt.plot(iters, values, label=f'alpha0={alpha0}, beta={beta}')  
            all_values.extend(values)  
    
    
    plt.ylim([min(all_values) - 1, max(all_values) + 1])

    
    plt.legend(facecolor='black', edgecolor='white', fontsize='medium')

    plt.show()

def B2(f, df, x, fnum):
    alphas = [0.01,0.1,1]
    betas = [0.25,0.9]
    iterations = 200
    iters = list(range(iterations + 1))
    legend = []
    contour_data = []
    for alpha in alphas:
        for beta in betas:
            xs, values, steps = HeavyBall(f, df, x, [alpha, beta], iterations=iterations)
            legend.append(f'$\\alpha={alpha},\\, \\beta={beta}$')
            print(f'alpha={alpha}, beta={beta}: final_value={values[-1]}')
            plt.plot(iters, values)
    plt.xlabel('iterations')
    plt.ylabel(f'$f_{fnum}(x,y)$')
    plt.title(f'Heavy Ball for $f_{fnum}(x,y)$')
    if fnum == 1: plt.ylim([0, 200])
    else: plt.ylim([0, 30])
    plt.legend(legend)
    plt.show()

def B3(f, df, x, fnum):
    alphas = [0.01, 0.1, 1]
    beta1s = [0.25, 0.9]
    beta2s = [0.9, 0.999]
    iterations = 200
    iters = list(range(iterations + 1))
    legend = []
    for beta2 in beta2s:
        for alpha in alphas:
            for beta1 in beta1s:
                xs, values, steps = Adam(f, df, x, [alpha, beta1, beta2], iterations=iterations)
                legend.append(f'$\\alpha={alpha},\\, \\beta_1={beta1}$')
                print(f'alpha={alpha}, beta1={beta1}, beta2={beta2}: final_value={values[-1]}')
                plt.figure(1)
                plt.plot(iters, values)
                stepsx = [step[0] for step in steps]
                stepsy = [step[1] for step in steps]
                plt.figure(2)
                plt.plot(iters, stepsx)
                plt.figure(3)
                plt.plot(iters, stepsy)
        plt.figure(1)
        plt.xlabel('iterations')
        plt.ylabel(f'$f_{fnum}(x,y)$')
        plt.title(f'Adam for $f_{fnum}(x,y),\\, \\beta_2={beta2}$')
        plt.legend(legend)
        plt.figure(2)
        plt.xlabel('iterations')
        plt.ylabel('step $x$')
        plt.title(f'Step size of $x$ for $f_{fnum},\\, \\beta_2={beta2}$')
        plt.figure(3)
        plt.xlabel('iterations')
        plt.ylabel('step $y$')
        plt.title(f'Step size of $y$ for $f_{fnum},\\, \\beta_2={beta2}$')
        plt.show()

def Question_B():
    f1 = lambda x, y: 1*(x - 1)**4 + 8*(y - 1)**2
    df1_x = lambda x: 4*(x - 1)**3
    df1_y = lambda y: 16*y - 16
    f2 = lambda x, y: 8*abs(y - 1) + max(0, x - 1)
    df2_x = lambda x: np.heaviside(x - 1, 0)
    df2_y = lambda y: 8*np.sign(y - 1)
    print('(b)(i) f1')
    B1(f1, [df1_x, df1_y], [3, 0], 1)
    print('(b)(i) f2')
    B1(f2, [df2_x, df2_y], [3, 0], 2)
    print('(b)(ii) f1')
    B2(f1, [df1_x, df1_y], [3, 0], 1)
    print('(b)(ii) f2')
    B2(f2, [df2_x, df2_y], [3, 0], 2)
    print('(b)(iii) f1')
    B3(f1, [df1_x, df1_y], [3, 0], 1)
    print('(b)(iii) f2')
    B3(f2, [df2_x, df2_y], [3, 0], 2)


def Question_C():
    f = lambda x: max(x, 0)
    df = lambda x: np.heaviside(x, 0)
    num_iters = 150
    #can be changed to 15000 for last answer
    iters = list(range(num_iters + 1))
    for x0 in [-1, 1, 100]:
        _, values, _ = RMSprop(f, [df], [x0], [0.01, 0.9], iterations=num_iters)
        print(f'RMSProp (x0={x0}): {values[-1]}')
        plt.plot(iters, values)
        _, values, _ = HeavyBall(f, [df], [x0], [1, 0.25], iterations=num_iters)
        print(f'Heavy Ball (x0={x0}): {values[-1]}')
        plt.plot(iters, values)
        _, values, _ = Adam(f, [df], [x0], [0.01, 0.9, 0.999], iterations=num_iters)
        print(f'Adam (x0={x0}): {values[-1]}')
        plt.plot(iters, values)
        plt.xlabel(f'iterations ($x_0={x0}$)')
        plt.ylabel('f(x)')
        plt.legend(['RMSProp', 'Heavy Ball', 'Adam'])
        plt.show()


calculate_derivatives()
Question_B()
Question_C()