from gplearn_modify.genetic import SymbolicRegressor
from gplearn_modify.functions import make_function
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
from gplearn_modify._program import print_formula, printout
import numpy as np
from gplearn_modify.functions import _Function


def sphere_function(X):
    y = np.full_like(X[:, 0], 0)
    for index, sample in enumerate(X):
        # y[index] = (sample * sample).sum(axis=0)
        y[index] = sample.sum(axis=0) ** 2
    return y


def rosenbrock_function(X):
    y = np.full_like(X[:, 0], 0)
    for index, sample in enumerate(X):
        y[index] = ((sample[:-1]**2 - sample[1:])**2 + (sample[:-1] - 1)**2).sum(axis=0)
    return y


def cosine(X):
    y = np.full_like(X[:, 0], 0)
    for index, sample in enumerate(X):
        y[index] = X.shape[1] - X.shape[1] * np.cos(sample).sum(axis=0)  #
    return y


rng = check_random_state(0)
# Training samples
samples, dim = 100, 4
X_train = rng.uniform(-5, 5, samples * dim).reshape(samples, dim)
# y_train = sphere_function(X_train)
# y_train = rosenbrock_function(X_train)
# y_train = cosine(X_train)
# y_train = X_train[:, 0] - np.tan(X_train[:, 0])
# y_train = X_train[:, 0]**2 + np.tan(X_train[:, 1]) + 3000 * np.log(np.abs(X_train[:, 2])) + np.log(np.abs(X_train[:, 3]))
# y_train = X_train[:, 0]**2 + np.tan(X_train[:, 1]) + np.cos(X_train[:, 2]) + np.log(np.abs(X_train[:, 3])) + np.exp(X_train[:, 1])
# y_train = (X_train[:, 0] + X_train[:, 1] + X_train[:, 2] + X_train[:, 3]) / 4
# y_train = X_train[:, 0] + X_train[:, 1]*X_train[:, 2] - X_train[:, 3]*X_train[:, 4]
# y_train = np.sin(X_train[:, 0])/np.sin(X_train[:, 1])
# y_train = np.log(np.abs(X_train[:, 0] + X_train[:, 1] + X_train[:, 2] + X_train[:, 3]))
# y_train = 1000 * (X_train[:, 0]**2 + X_train[:, 1] + X_train[:, 2])
# y_train = 100 * (X_train[:, 0]**2 + X_train[:, 1]**2 + X_train[:, 2]**2 + X_train[:, 3]**2 + X_train[:, 4]**2)
y_train = (X_train[:, 0] + X_train[:, 1] + X_train[:, 2] + X_train[:, 3]) / 4 +\
          X_train[:, 1] * X_train[:, 3] + X_train[:, 0] * X_train[:, 2] + X_train[:, 3]**4 * X_train[:, 0]
# y_train = 3000 * X_train[:, 0]**2 - X_train[:, 1]**2 + X_train[:, 1] + X_train[:, 1] * X_train[:, 2] * X_train[:, 3] + \
#           X_train[:, 1] * X_train[:, 3] + X_train[:, 0] * X_train[:, 2] + X_train[:, 3]**4 * X_train[:, 0]
# y_train = 3000 * (X_train[:, 0]**2 - X_train[:, 1]**2 + X_train[:, 1] + X_train[:, 1] * X_train[:, 2] * X_train[:, 3] +
#                   X_train[:, 1] * X_train[:, 3] + X_train[:, 0] * X_train[:, 2] + X_train[:, 3]**4 * X_train[:, 0])

# , 'add', 'mul', 'sum', 'prod', 'pow', 'mean', 'inv', 'neg', 'abs', 'mean', 'cos', 'sin', 'tan', 'log', 'exp'
# function_set = ['add', 'sub', 'mul', 'div', 'sum', 'prod', 'mean', 'pow', 'exp',
#                 'sin', 'cos', 'tan', 'log', 'abs', 'neg', 'min', 'max', 'sqrt', 'inv']
# function_set = ['add', 'sub', 'mul', 'div', 'sum', 'prod', 'mean', 'pow', 'exp',
#                 'sin', 'cos', 'tan', 'log', 'abs', 'neg', 'min', 'max', 'sqrt']
# function_set = ['add', 'sub', 'mul', 'div', 'sum', 'prod', 'mean', 'pow', 'abs', 'neg']
function_set = ['add', 'sub', 'mul', 'div', 'sum', 'pow', 'mean', 'sin', 'cos', 'tan', 'log', 'sqrt', 'abs', 'neg', 'exp']
est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.1, p_point_mutation=0.05,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0,
                           function_set=function_set, init_depth=(3, 5), mutate_depth=(3, 10),
                           init_method='full', variable_range=(-5.5, 5.5), n_jobs=2)

est_gp.fit(X_train, y_train)
best_program = est_gp._program
print(best_program)  # 输出最优个体
printout(program=best_program.program, max_dim=4)
print_formula(program=best_program.program, max_dim=4, show_operand=True)
print()

X_augmentation = rng.uniform(-5, 5, 100 * 10).reshape(100, 10)
random_state = check_random_state(0)
best_program.execute(X=X_augmentation, random_state=random_state)
printout(program=best_program.program, max_dim=10)
print_formula(program=best_program.program, max_dim=10, show_operand=True)
