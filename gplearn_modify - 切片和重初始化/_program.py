"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import math
import random
from copy import copy, deepcopy

import numpy as np
from sklearn.utils.random import sample_without_replacement

from .functions import _Function, _sum, _prod, _mean, _sigmoid, _protected_power, _protected_sub, _protected_multiply, add2, sub2, mul2
from .functions import _protected_division, _protected_sqrt, _protected_log, _protected_inverse, _protected_exp
from .utils import check_random_state


rng = check_random_state(0)
X_train = rng.uniform(-5, 5, 100 * 10).reshape(100, 10)
check_constant_function = True

default_remaining = [3, 3, 1, 1]
default_total = [0, 0, 0, 0]
aggregate = ['prod', 'mean', 'sum', 'min', 'max']  # aggragate函数名称列表
concatenate = ['add', 'sub', 'mul', 'div', 'sum', 'prod', 'mean']  # 可能会出现主导问题的连接操作符
operator = ['neg', 'inv', 'abs']  # neg/inv/abs
elementary_functions = ['sin', 'cos', 'tan', 'log']  # 基本初等函数名称列表
ignore = [['add', 'sub', 'sum'],  # 加性函数节点
          ['add', 'sub', 'sum', 'mul', 'div', 'prod']]  # 加性和乘性函数节点


def printout(program):
    for node in program:
        if isinstance(node, _Function):
            if node.name in ['sum', 'prod', 'mean']:
                print(f"{node.name}[{node.input_dimension},{node.arity}]", end=' ')  # ,{node.parent_distance},{node.child_distance_list}
            else:
                print(f"{node.name}[{node.input_dimension}]", end=' ')  # [{node.parent_distance},{node.child_distance_list}]
        else:
            print(node, end=' ')
    print()


def print_formula(program, show_operand=False):  # 颜色编码从'\033[31m'到'\033[38m'
    formula_stack = []
    min_priority_stack = []  # 用于子树的内括号判断，记录每个子树的min_priority
    # name_mapping = {'add': '+', 'sub': '-', 'mul': '×', 'div': '/', 'pow': '^'}  # , 'sum': '+', 'prod': '×'
    name_mapping = {'add': '+', 'neg': '-', 'sub': '-', 'mul': '×', 'div': '/', 'pow': '^'}  # , 'sum': '+', 'prod': '×'
    # priority = {'add': 1, 'sub': 2, 'mul': 3, 'div': 4, 'pow': 5}  # 操作符的优先级
    priority = {'add': 1, 'neg': 2, 'sub': 2, 'mul': 3, 'div': 4, 'pow': 5}  # 操作符的优先级
    formula = ''
    last_arity = 0
    last_name = ''
    min_priority = 5  # 用于子树的外括号判断
    for node in program:
        if isinstance(node, _Function):
            formula_stack.append(node.name)
            formula_stack.append(node.arity)
            last_name = node.name
            last_arity = node.arity
        else:
            if show_operand:  # 展示具体的操作数，操作数分为向量切片(tuple)和常数向量list[ndarray]两种
                temp = '\033[36m' + '[' + '\033[0m'
                if isinstance(node, tuple):
                    for i in range(node[0], node[1], node[2]):
                        temp += 'X' + str(i) + ', '
                else:
                    for i in node[0]:  # 遍历ndarray
                        temp += str(i) + ', '
                temp = temp[:-2] + '\033[36m' + ']' + '\033[0m'
                formula_stack.append(temp)
            else:
                formula_stack.append('o')
            # 如果arity已经满足，且中间没有arity数字，说明操作数数目已经满足
            # 同时在该完整子树内进行内括号判断，即根据基本操作符的优先级顺序来添加括号
            while last_arity + 1 <= len(formula_stack) and \
                    formula_stack[-(last_arity + 1)] == last_arity and\
                    formula_stack[-(last_arity + 1)] not in formula_stack[- last_arity:]:
                for i in range(last_arity):  # 移除末尾last_arity个操作数
                    intermediate = formula_stack.pop()
                    if intermediate[0] != '@':  # 不是子树，则不需要内括号判断
                        formula = intermediate + formula
                        if last_name == 'neg':
                            min_priority = 0  # neg对外优先级最低，对内优先级与sub相同
                        elif last_name in name_mapping.keys():
                            min_priority = priority[last_name]
                        else:
                            min_priority = 5
                    else:  # 如果是子树，则需要知道min_priority
                        intermediate = intermediate[1:]  # 去掉第一个特殊字符@
                        min_priority = min_priority_stack.pop()  # 获取最后一个子树的min_priority
                        if last_name in name_mapping.keys():  # 在函数名字-符号映射表内，则判断是否需要添加括号
                            if min_priority < priority[last_name] or \
                                    min_priority == priority[last_name] and min_priority % 2 == 0:  # 相等且为2或4，即减和除
                                formula = '(' + intermediate + ')' + formula
                            else:
                                formula = intermediate + formula
                            if last_name != 'neg':
                                min_priority = priority[last_name]  # 优先级取最低的
                            else:
                                min_priority = 0
                        else:  # 其他函数，无需内括号判断
                            formula = intermediate + formula
                            min_priority = 5
                    if i != last_arity - 1:  # 不是最后一个操作数，则需要加上操作符
                        if last_name in name_mapping.keys():
                            formula = name_mapping[last_name] + formula
                        else:
                            formula = ', ' + formula
                    elif last_name == 'neg':
                        formula = name_mapping[last_name] + formula
                formula_stack.pop()  # 移除函数节点的arity数字
                formula_stack.pop()  # 移除函数节点的函数名字
                # 一个完整子树的外括号判断：abs为||；neg为-，并且判断是否需要添加括号；非基本操作符的函数外括号添加，聚集函数使用{}，其余使用()
                if last_name == 'abs':
                    front = '\033[32m' + '|' + '\033[0m'
                    end = '\033[32m' + '|' + '\033[0m'
                    formula = front + formula + end
                # elif last_name == 'neg':
                #     if min_priority <= 2:
                #         formula = '-(' + formula + ')'
                #     else:
                #         formula = '-' + formula
                #     min_priority = 0
                elif last_name not in name_mapping.keys():  # 若不在函数名字-符号映射表内，则需要加上函数节点的名字
                    if last_name in aggregate:
                        front = '\033[31m' + '{' + '\033[0m'
                        end = '\033[31m' + '}' + '\033[0m'
                        formula = last_name + front + formula + end
                    else:
                        front = '('
                        end = ')'
                        formula = last_name + front + formula + end
                if len(formula_stack) == 0:  # formula为空
                    print(formula)
                    return
                # 找新的最后一个函数节点，更新last_name和last_arity
                for index in range(len(formula_stack)):
                    if not isinstance(formula_stack[- 1 - index], str):  # 不是str，则为arity
                        last_arity = formula_stack[- 1 - index]
                        last_name = formula_stack[- 2 - index]
                        break
                formula = '@' + formula  # @开头表示这是一个子树的文本表示
                formula_stack.append(formula)  # 附加到末尾
                min_priority_stack.append(min_priority)
                formula = ''
                min_priority = 5


# function是_Function对象，若函数为连加或连乘，则需要创建新的实例，以记录不同的arity值。
def new_operator(function, random_state, n_features, output_dimension, remaining):
    new_function = 0
    new_remaining = deepcopy(remaining)
    # total = [0, 0, 0, 0]  # 默认值是[0, 0, 0, 0]
    if function.name == 'sum' or function.name == 'prod' or function.name == 'mean':
        if output_dimension == 1:  # output_dimension=1和arity=1是充分必要条件！
            arity = 1  # arity为1时可求向量的各分量的累加或累乘
        else:  # output_dimension不为1时arity也不为1
            if n_features > 1:
                arity = random_state.randint(2, n_features + 1)  # [2, self.n_features]
            else:
                arity = 2
        if function.name == 'sum':  # 初始名字是sum
            # new_function = _Function(function=_sum, name=f'sum({arity})', arity=arity)
            new_function = _Function(function=_sum, name=f'sum', arity=arity)
            new_remaining[0] -= 1  # 剩余aggregate次数 - 1
            # total[0] += 1
        elif function.name == 'prod':
            # new_function = _Function(function=_prod, name=f'prod({arity})', arity=arity)
            new_function = _Function(function=_prod, name=f'prod', arity=arity)
            new_remaining[0] -= 3  # 剩余aggregate次数 - 3
            # total[0] += 3
        elif function.name == 'mean':
            # new_function = _Function(function=_mean, name=f'mean({arity})', arity=arity)
            new_function = _Function(function=_mean, name=f'mean', arity=arity)
            new_remaining[0] -= 3  # 剩余aggregate次数 - 3
            # total[0] += 3
        # 输入/输出维度设置
        new_function.remaining = new_remaining
        # new_function.total = total
        new_function.output_dimension = output_dimension
        if new_function.arity == 1:  # arity为=1时可以允许输入维度不等于输出维度
            if n_features > 1:
                new_function.input_dimension = random_state.randint(1, n_features) + 1  # [2, self.n_features]，至少为2
            else:
                new_function.input_dimension = 1
        else:
            new_function.input_dimension = new_function.output_dimension
    else:  # 对于其他运算符同样要创建新实例，以记录不同的输入和输出维度
        if function.name == 'add':
            new_function = _Function(function=np.add, name='add', arity=2)
        elif function.name == 'sub':
            new_function = _Function(function=np.subtract, name='sub', arity=2)
        elif function.name == 'mul':
            new_function = _Function(function=np.multiply, name='mul', arity=2)
        elif function.name == 'div':
            new_function = _Function(function=_protected_division, name='div', arity=2)
        elif function.name == 'max':
            new_function = _Function(function=np.maximum, name='max', arity=2)
            new_remaining[0] -= 1  # 剩余aggregate次数 - 1
            # total[0] += 1
        elif function.name == 'min':
            new_function = _Function(function=np.minimum, name='min', arity=2)
            new_remaining[0] -= 1  # 剩余aggregate次数 - 1
            # total[0] += 1
        elif function.name == 'pow':
            new_function = _Function(function=_protected_power, name='pow', arity=2)
            new_remaining[1] -= 1  # 剩余pow次数 - 1
            # total[1] += 1
        elif function.name == 'log':
            new_function = _Function(function=_protected_log, name='log', arity=1)
            new_remaining[2] -= 1  # 基本初等函数次数 - 1
            # total[2] += 1
        elif function.name == 'sin':
            new_function = _Function(function=np.sin, name='sin', arity=1)
            new_remaining[2] -= 1  # 基本初等函数次数 - 1
            # total[2] += 1
        elif function.name == 'cos':
            new_function = _Function(function=np.cos, name='cos', arity=1)
            new_remaining[2] -= 1  # 基本初等函数次数 - 1
            # total[2] += 1
        elif function.name == 'tan':
            new_function = _Function(function=np.tan, name='tan', arity=1)
            new_remaining[2] -= 1  # 基本初等函数次数 - 1
            # total[2] += 1
        elif function.name == 'neg':
            new_function = _Function(function=np.negative, name='neg', arity=1)
            # new_remaining[3] = 1  # 还可以选inv和abs
        elif function.name == 'inv':
            new_function = _Function(function=_protected_inverse, name='inv', arity=1)
            # new_remaining[3] = 2  # 还可以选abs
        elif function.name == 'abs':
            new_function = _Function(function=np.abs, name='abs', arity=1)
            # new_remaining[3] = 3  # neg,inv和abs都不可再选
        elif function.name == 'sqrt':
            new_function = _Function(function=_protected_sqrt, name='sqrt', arity=1)
        elif function.name == 'sig':
            new_function = _Function(function=_sigmoid, name='sig', arity=1)
        elif function.name == 'exp':
            new_function = _Function(function=_protected_exp, name='exp', arity=1)
            new_remaining[3] -= 1  # 剩余exp次数减1
        # 这些运算符的输入维度=输出维度
        new_function.remaining = new_remaining
        # new_function.total = total
        new_function.output_dimension = output_dimension
        new_function.input_dimension = new_function.output_dimension
    return new_function


class _Program(object):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    variable_range : tuple of two floats
        The range of variables to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,  # 函数(对象)集
                 arities,  # arity字典，对应上述函数集
                 init_depth,  # 第一代种群树深度范围
                 init_method,  # grow，full，half and half
                 n_features,  # 输入向量X的维度
                 variable_range,  # 变量的范围
                 metric,  # fitness
                 p_point_replace,  # 点突变的概率
                 parsimony_coefficient,  # 简约系数
                 random_state,  # np的随机数生成器
                 transformer=None,  # sigmoid函数
                 feature_names=None,  # X的各分量名字
                 program=None):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.variable_range = variable_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def build_program(self, random_state, output_dimension=1):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.
        output_dimension: int
            The dimension of program's output

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)  # 从给定的深度范围内随机选定该树的最大深度
        # init_function_set = [add2, sub2, mul2]  # , sum, prod  sum和prod作为根节点的arity只有1，不适合
        # 前两层节点改为从加减乘累加累乘中选择
        # function = random_state.randint(len(init_function_set))
        function = random_state.randint(len(self.function_set))
        function = self.function_set[function]  # 随机挑选一个_Function对象
        current_remaining = default_remaining
        function = new_operator(function, random_state, self.n_features, output_dimension, current_remaining)
        function.depth = 0  # 初始节点的深度为0
        function.parent_distance = 0  # 根节点的该属性为0
        program = [function]
        terminal_stack = [function.arity]
        program = self.set_total(len(program) - 1, program)  # 设置total属性
        next_is_terminal = False
        next_is_function = False
        while terminal_stack:  # 当栈不为空时，重复添加函数或向量
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(choice)
            parent_index = self.find_parent(len(program), program)
            parent = program[parent_index]
            parent_name = parent.name  # 父节点函数名字
            existed_dimension = parent.input_dimension  # 上下层接口一致
            value_range = parent.value_range  # 父节点函数值域
            # 维护父节点的child_distance_list属性
            if not next_is_function and not next_is_terminal:  # 如果不是重复循环，就记录当前点
                parent.child_distance_list.append(len(program) - parent_index)
            # full策略优先选择添加函数
            if (depth < max_depth) and (method == 'full' or choice <= len(self.function_set)) and not next_is_terminal \
                    or next_is_function:
                current_remaining = parent.remaining  # 记录父节点的remaining
                current_function_set = self.clip_function_set(function_set=self.function_set,
                                                              remaining=current_remaining,
                                                              parent_name=parent_name)  # 求约束规则下的函数集
                # if depth > 1:
                #     current_function_set = self.clip_function_set(function_set=self.function_set,
                #                                                   remaining=current_remaining,
                #                                                   parent_name=parent_name)  # 求约束规则下的函数集
                # else:  # 前两层节点改为从加减乘累加累乘中选择
                #     current_function_set = self.clip_function_set(function_set=init_function_set,
                #                                                   remaining=current_remaining,
                #                                                   parent_name=parent_name)  # 求约束规则下的函数集
                if parent_name in ['max', 'min'] and terminal_stack[-1] == 1:  # max和min的第二个操作数
                    if isinstance(program[parent_index + 1], _Function):
                        if program[parent_index + 1].name in current_function_set:
                            current_function_set.remove(program[parent_index + 1].name)
                # 若函数集为空，或当前点pow函数的第二个操作数，则选择terminal
                if (parent_name == 'pow' and terminal_stack[-1] == 1) or len(current_function_set) == 0:
                    next_is_terminal = True  # 不添加函数节点，改为添加terminal
                    if next_is_function:
                        raise ValueError("Loop: next_is_function and next_is_terminal are both True.")
                    continue
                # 在约束函数集中选择函数节点
                function = random_state.randint(len(current_function_set))
                function = current_function_set[function]
                function = new_operator(function, random_state, self.n_features,
                                        existed_dimension, current_remaining)
                function.depth = depth  # 记录函数节点所在深度
                function.parent_distance = parent_index - len(program)  # 父节点相对于自己的距离
                if len(value_range):  # 值域不为空，传递给子节点
                    function.value_range = deepcopy(value_range)
                program.append(function)
                terminal_stack.append(function.arity)
                # 更新当前点以及其所有ancestors的total属性，所以new_operator函数中不需要再对total属性进行处理
                program = self.set_total(len(program) - 1, program)
                next_is_function = False  # 回到正常状态
            else:
                # 生成常数时父节点有value_range，则需要进行主导现象限制  'add', 'sub', 'sum', 'mean', 'max', 'min'
                if parent_name in ['mul', 'div', 'prod']:
                    if len(value_range):
                        level = np.max(np.abs(value_range))  # level > 0
                    else:
                        level = np.max(np.abs(self.variable_range))
                    const_range = (math.ceil(level / 5), math.ceil(level * 5))  # 常数绝对值大小在该范围内即可
                else:
                    const_range = self.variable_range
                assert const_range[1] > const_range[0]
                # 生成当前父节点最后一个子节点时要回溯至第一个sub或div节点，避免生成导致完全抵消的变量节点
                name_list = []
                index_list = []
                has_sub_div = False
                temp_parent = parent
                temp_name = parent_name
                temp_index = parent_index
                prohibit = []
                children = []
                no_cancel = False
                if terminal_stack[-1] == 1:  # 当前父节点的最后一个操作数   and parent_name != 'pow'
                    # 遍历子节点，不加入当前节点，但arity为1的函数节点会导致children为空
                    for c in program[parent_index].child_distance_list[:-1]:
                        if not isinstance(program[parent_index + c], tuple):  # children会比children2少最后一个操作数
                            no_cancel = True
                            break
                        else:  # 记录变量子节点
                            children.append(program[parent_index + c])
                    if not no_cancel:  # 子节点有常数或函数节点就认为不会抵消
                        while temp_parent.parent_distance != 0:  # 一直回溯到根节点
                            index_list.append(temp_index)
                            if temp_name in ['sub', 'div']:
                                has_sub_div = True
                                break
                            else:  # 不是sub和div
                                name_list.append(temp_name)
                                temp_index = temp_index + temp_parent.parent_distance
                                temp_parent = program[temp_index]  # 父节点回溯
                                temp_name = temp_parent.name
                        if temp_name in ['sub', 'div'] and not has_sub_div:  # 若第一个sub或div节点是根节点
                            has_sub_div = True
                            index_list.append(temp_index)
                if has_sub_div:
                    # 有sub和div祖先节点且当前点位于其右子树
                    if len(index_list) == 1 and len(program) != index_list[-1] + 1 or \
                            len(index_list) > 1 and index_list[-2] != index_list[-1] + 1:
                        temp_index = index_list[-1] + 1
                        if program[index_list[-1]].name == 'div':
                            if len(name_list):
                                complete = False
                                name_index = 1
                                while name_list[- name_index] in ['abs', 'neg']:  # 右边跳过abs和neg
                                    name_index += 1
                                    if name_index > len(name_list):  # 等价于右边name_list为空
                                        if isinstance(program[temp_index], tuple):
                                            prohibit.append(program[temp_index])
                                        elif isinstance(program[temp_index], _Function):
                                            while isinstance(program[temp_index], _Function) and \
                                                    program[temp_index].name in ['abs', 'neg']:  # 左边跳过abs和neg
                                                temp_index += 1  # abs和neg是单节点函数，左边是完整子树，不会越界
                                            if isinstance(program[temp_index], tuple):
                                                prohibit.append(program[temp_index])
                                        complete = True
                                        break
                                if not complete:  # complete为False说明右边还有待判断的函数，且不是abs或neg
                                    # 当前temp_index是div的左子节点，所以可以通过这种方式跳过abs和neg
                                    while isinstance(program[temp_index], _Function) and\
                                            program[temp_index].name in ['abs', 'neg']:  # 左边跳过abs和neg
                                        temp_index += 1  # abs和neg是单节点函数，左边是完整子树，不会越界
                                    if isinstance(program[temp_index], _Function) and\
                                            program[temp_index].name == name_list[- name_index]:
                                        # 相同函数名匹配后右边要重新跳过abs和neg
                                        # printout(program)
                                        # print(program[temp_index].name)
                                        name_index += 1
                                        if name_index <= len(name_list):
                                            while name_list[- name_index] in ['abs', 'neg']:  # 右边跳过abs和neg
                                                name_index += 1
                                                if name_index > len(name_list):  # 等价于右边name_list为空
                                                    # complete = True
                                                    break
                                        child_index_stack = [0]  # 记录还未探索的子节点索引
                                        # 这里应该要深度优先遍历完整个子树，但碰到一条符合的之后就停止，这样足矣避免完全抵消
                                        while len(child_index_stack) or complete:  # 若complete=True，则进行最后一次循环
                                            # printout(program)
                                            # print(program[temp_index].name)
                                            # print(name_list)
                                            # print(name_index)
                                            if name_index > len(name_list) and complete:
                                                children2 = []
                                                for c in program[temp_index].child_distance_list:  # 遍历子节点
                                                    # children2要加上对pow的检测
                                                    if not isinstance(program[temp_index + c], tuple):
                                                        if isinstance(program[temp_index], _Function) and \
                                                                program[temp_index].name == 'pow':  # 父节点是pow函数
                                                            # pow函数的最后一个操作数，即指数节点
                                                            if c == program[temp_index].child_distance_list[-1]:
                                                                children2.append(program[temp_index + c])
                                                            else:
                                                                no_cancel = True
                                                                break
                                                        else:
                                                            no_cancel = True
                                                            break
                                                    else:
                                                        children2.append(program[temp_index + c])
                                                if no_cancel:  # 子节点有常数或函数节点就认为不会抵消
                                                    break
                                                if not len(children):  # children为空，说明当前函数arity为1
                                                    # if program[temp_index].name == program[parent_index].name:
                                                    prohibit.append(children2[0])  # 可以避免完全抵消
                                                    break
                                                if program[temp_index].name == program[parent_index].name:
                                                    # 子节点组合匹配，子节点都是变量节点才需要检测抵消
                                                    if program[temp_index].name in ['add', 'mul', 'max', 'min',
                                                                                    'sum', 'prod', 'mean']:
                                                        for c in children:
                                                            if c in children2:  # 在children2中，则将children2中相同的节点去除
                                                                children2.remove(c)
                                                            if c == children[-1] and len(children2) == 1:
                                                                prohibit.append(children2[0])
                                                    else:  # 其他函数则进行序列匹配
                                                        for c_i, c in enumerate(children):
                                                            if c != children2[c_i]:  # # 如果有一个不相同，则不会抵消
                                                                break
                                                            if c == children[-1]:  # 前面都相同，则禁止生成children2的最后一个
                                                                prohibit.append(children2[-1])
                                                else:  # 左右两边函数名不相同，则右边是abs或neg中的一个
                                                    prohibit.append(children2[0])
                                                break
                                            match = False
                                            fence = child_index_stack.pop()  # 获取最新的
                                            temp_children = program[temp_index].child_distance_list
                                            for j, c in enumerate(temp_children):
                                                if j < fence:
                                                    continue
                                                if isinstance(program[temp_index + c], _Function):
                                                    if program[temp_index + c].name in ['abs', 'neg']:
                                                        match = True
                                                        child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                                        child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                                        temp_index += c
                                                        break
                                                    elif name_index <= len(name_list) and\
                                                            program[temp_index + c].name == name_list[- name_index]:
                                                        match = True
                                                        child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                                        child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                                        temp_index += c
                                                        name_index += 1  # 搜索下一个不是abs和neg的函数名字
                                                        # 函数名匹配成功后右边重新跳过abs和neg
                                                        if name_index <= len(name_list):
                                                            while name_list[- name_index] in ['abs', 'neg']:
                                                                name_index += 1
                                                                if name_index > len(name_list):  # 等价于右边name_list为空
                                                                    # 且左节点的子节点中没有abs和neg，则complete=True
                                                                    break
                                                        break
                                            if not match:  # 这个当前点的子节点没有匹配的
                                                # printout(program)
                                                # print(program[temp_index].name)
                                                # print(name_list)
                                                # print(name_index)
                                                if name_index > len(name_list):  # 这表明没有abs或neg函数，则可以停止搜索
                                                    complete = True
                                                else:  # 否则还需要返回至当前点的父节点
                                                    temp_index += program[temp_index].parent_distance
                                                    name_index -= 1  # 搜索上一个名字
                                                    while program[temp_index].name in ['neg', 'abs'] and \
                                                            program[temp_index].parent_distance != 0:
                                                        if len(child_index_stack):
                                                            child_index_stack.pop()  # 去掉neg和abs记录的下一探索点
                                                        else:  # 超出界限说明已经不满足抵消了
                                                            break
                                                        temp_index += program[temp_index].parent_distance
                                                        name_index -= 1  # 搜索上一个名字
                            else:  # 右边name_list为空
                                if isinstance(program[temp_index], tuple):
                                    prohibit.append(program[temp_index])
                                else:
                                    while isinstance(program[temp_index], _Function) and \
                                            program[temp_index].name in ['abs', 'neg']:
                                        temp_index += 1
                                    if isinstance(program[temp_index], tuple):
                                        prohibit.append(program[temp_index])
                        elif len(name_list):  # 与sub之间存在中间函数节点
                            if isinstance(program[temp_index], _Function) and program[temp_index].name == name_list[-1]:
                                child_index_stack = [0]  # 记录还未探索的子节点索引
                                name_index = 1
                                while len(child_index_stack):  # 这里应该要深度优先遍历完整个子树，而不是只遍历一个子支
                                    match = False
                                    # 找到了父节点均相同的子支
                                    if name_index + 1 > len(name_list):
                                        # printout(program)
                                        # print(program[temp_index].name)
                                        children2 = []
                                        for c in program[temp_index].child_distance_list:  # 遍历子节点
                                            # children2要加上对pow的检测
                                            if not isinstance(program[temp_index + c], tuple):
                                                if isinstance(program[temp_index], _Function) and \
                                                        program[temp_index].name == 'pow':  # 父节点是pow函数
                                                    # pow函数的最后一个操作数，即指数节点
                                                    if c == program[temp_index].child_distance_list[-1]:
                                                        children2.append(program[temp_index + c])
                                                    else:
                                                        no_cancel = True
                                                        break
                                                else:
                                                    no_cancel = True
                                                    break
                                            else:
                                                children2.append(program[temp_index + c])
                                        if no_cancel:  # 子节点有常数或函数节点就认为不会抵消
                                            break
                                        if not len(children):  # children为空，说明当前函数arity为1
                                            prohibit.append(children2[0])
                                            break
                                        if program[temp_index].name == program[parent_index].name:
                                            # 子节点组合匹配，子节点都是变量节点才需要检测抵消
                                            if program[temp_index].name in ['add', 'mul', 'max', 'min',
                                                                            'sum', 'prod', 'mean']:
                                                for c in children:
                                                    if c in children2:  # 在children2中，则将children2中相同的节点去除
                                                        children2.remove(c)
                                                    if c == children[-1] and len(children2) == 1:
                                                        prohibit.append(children2[0])
                                            else:  # 其他函数则进行序列匹配
                                                for c_i, c in enumerate(children):
                                                    if c != children2[c_i]:  # # 如果有一个不相同，则不会抵消
                                                        break
                                                    if c == children[-1]:  # 前面都相同，则禁止生成children2的最后一个
                                                        prohibit.append(children2[-1])
                                        break
                                    temp_name = name_list[- name_index - 1]
                                    fence = child_index_stack.pop()  # 获取最新的
                                    temp_children = program[temp_index].child_distance_list
                                    for j, c in enumerate(temp_children):
                                        if j < fence:
                                            continue
                                        if isinstance(program[temp_index + c], _Function) and \
                                                program[temp_index + c].name == temp_name:  # 找到了一个匹配的子节点
                                            child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                            temp_index = temp_index + c
                                            child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                            name_index += 1  # 搜索下一个名字
                                            match = True
                                            break
                                    if not match:  # 这个当前点的子节点没有匹配的，则返回至当前点的父节点
                                        temp_index = temp_index + program[temp_index].parent_distance  # 确保初始值正确
                                        name_index -= 1  # 搜索上一个名字
                        else:  # 与sub之间没有中间函数节点
                            if isinstance(program[temp_index], tuple):
                                prohibit.append(program[temp_index])
                if parent_name in ['max', 'min'] and terminal_stack[-1] == 1:  # max和min的第二个操作数
                    if isinstance(program[parent_index + 1], tuple) and program[parent_index + 1] not in prohibit:
                        prohibit.append(program[parent_index + 1])
                if parent_name == 'pow':
                    if terminal_stack[-1] == 2:  # pow函数的第一个操作数不接收常数
                        terminal = self.generate_a_terminal(random_state, existed_dimension,
                                                            vary=True)  # , prohibit=prohibit
                    else:  # pow的第二个操作数应为整数向量
                        terminal = self.generate_a_terminal(random_state, existed_dimension,
                                                            const_int=True, prohibit=prohibit)
                # arity为1的函数节点不接收常数，其他函数节点最多只有一个常数子节点，prohibit可能导致只能生成常数节点，此时只能改为生成函数节点
                elif parent.constant_num >= min(1, parent.arity - 1):
                    if len(prohibit) and existed_dimension == self.n_features:
                        for item in prohibit:
                            if (item[1] - item[0] - 1) / item[2] + 1 == self.n_features:
                                next_is_function = True
                                break
                    if next_is_function:
                        continue
                    else:
                        terminal = self.generate_a_terminal(random_state, existed_dimension,
                                                            vary=True, prohibit=prohibit)
                else:  # 还没有常数节点，则可以生成常数节点
                    terminal = self.generate_a_terminal(random_state, existed_dimension,
                                                        const_range=const_range, prohibit=prohibit)
                if isinstance(terminal, list):  # 若生成常数节点，则需要维护constant_num属性
                    parent.constant_num += 1
                next_is_terminal = False  # 回到正常状态
                program.append(terminal)
                terminal_stack[-1] -= 1
                temp_range = np.array([0, 0])
                subtree_complete = False
                while terminal_stack[-1] == 0:  # 这里要对self.value_range属性进行维护
                    subtree_complete = True
                    terminal_stack.pop()
                    if not terminal_stack:  # terminal_stack为空时返回program
                        y_pred = self.execute_test(program, X_train)
                        if np.max(y_pred) - np.min(y_pred) <= 1e-8 and check_constant_function:
                            print("-----常数函数(build_program)-----")
                            printout(program)
                            print_formula(program, show_operand=True)
                        return program
                    terminal_stack[-1] -= 1
                    temp_range = np.array([0., 0.])
                    arity = program[parent_index].arity
                    if parent_name == 'mean':
                        for i in range(arity):
                            if isinstance(program[- 1 - i], _Function):
                                span = program[- 1 - i].value_range
                            else:
                                span = np.array(self.variable_range)
                            temp_range += span
                        temp_range = np.array([temp_range[0]/arity, temp_range[1]/arity])
                    elif parent_name in ['add', 'sum']:
                        for i in range(arity):
                            if isinstance(program[- 1 - i], _Function):
                                span = program[- 1 - i].value_range
                            else:
                                span = np.array(self.variable_range)
                            # print(f'add span:{span}')
                            temp_range += span
                            # print(f'add temp_range:{temp_range}')
                    elif parent_name == 'sub':
                        for i in range(arity):
                            if isinstance(program[- 1 - i], _Function):
                                span = program[- 1 - i].value_range
                            else:
                                span = np.array(self.variable_range)
                            # print(f'sub span:{span}')
                            if i == 0:
                                span = np.array([-span[1], -span[0]])
                            temp_range += span
                            # print(f'sub temp_range:{temp_range}')
                    elif parent_name in ['mul', 'prod']:
                        temp_range = [1, 1]
                        for i in range(arity):
                            if isinstance(program[- 1 - i], _Function):
                                span = program[- 1 - i].value_range
                            else:
                                span = np.array(self.variable_range)
                            # print(f'mul span:{span}')
                            temp = np.array([temp_range[0] * span[0],
                                             temp_range[0] * span[1],
                                             temp_range[1] * span[0],
                                             temp_range[1] * span[1]])
                            temp_range = np.array([np.min(temp), np.max(temp)])
                            # print(f'mul temp_range:{temp_range}')
                    elif parent_name in ['div', 'inv']:
                        temp_range = [1, 1]
                        for i in range(arity):
                            if isinstance(program[- 1 - i], _Function):
                                span = program[- 1 - i].value_range
                            else:
                                span = np.array(self.variable_range)
                            if i == 0:
                                if span[0] <= -0.001 and 0.001 <= span[1]:
                                    span = [-1000, 1/span[0], 1/span[1], 1000]
                                elif -0.001 <= span[0] <= 0.001 <= span[1]:
                                    span = [1/span[1], 1000]
                                elif 0.001 <= span[0] or span[1] <= -0.001:
                                    span = [1/span[1], 1/span[0]]
                                elif span[0] <= -0.001 <= span[1] <= 0.001:
                                    span = [-1000, 1/span[0]]
                                elif -0.001 <= span[0] <= span[1] <= 0.001:
                                    span = [1, 1]
                            if len(span) < 4:
                                temp = np.array([temp_range[0] * span[0],
                                                 temp_range[0] * span[1],
                                                 temp_range[1] * span[0],
                                                 temp_range[1] * span[1]])
                            else:
                                temp = []
                                for m in span:
                                    for n in temp_range:
                                        temp.append(m*n)
                                temp = np.array(temp)
                            # print(f'div span:{span}')
                            temp_range = np.array([np.min(temp), np.max(temp)])
                            # print(f'div temp_range:{temp_range}')
                    elif parent_name == 'pow':
                        if isinstance(program[-2], _Function):
                            span = program[-2].value_range
                        else:
                            span = np.array(self.variable_range)
                        span = np.array(span, dtype=np.float64)
                        exponent = program[-1][0][0]
                        # print(f'pow span:{span}, exponent:{exponent}')
                        if exponent > 0:
                            if exponent % 2 == 0:
                                if span[0] >= 0 or span[1] <= 0:
                                    x = np.power(span, exponent)
                                    temp_range = np.array([np.min(x), np.max(x)])
                                else:
                                    temp_range = np.array([0, np.max(np.power(span, exponent))])
                            else:  # 指数>0且为奇数，单调递增
                                temp_range = np.power(span, exponent)
                        else:  # 指数小于0，涉及分式，-0.001和0.001为临界点
                            if exponent % 2 == 0:
                                if span[0] <= -0.001 and 0.001 <= span[1]:
                                    temp_range = np.array([np.min(np.power(span, exponent)), 0.001**exponent])
                                elif -0.001 <= span[0] <= 0.001 <= span[1]:
                                    temp_range = np.power([span[1], 0.001], exponent)
                                elif 0.001 <= span[0] or span[1] <= -0.001:
                                    x = np.power(span, exponent)
                                    temp_range = np.array([np.min(x), np.max(x)])
                                elif span[0] <= -0.001 <= span[1] <= 0.001:
                                    temp_range = np.power([span[0], 0.001], exponent)
                                elif -0.001 <= span[0] <= span[1] <= 0.001:
                                    temp_range = [-1, 1]  # 无效
                            else:  # 指数<0且为奇数，这里为了方便某些情况只管右支或近似估计
                                if span[0] <= -0.001 and 0.001 <= span[1]:
                                    temp_range = np.power([-0.001, 0.001], exponent)
                                elif -0.001 <= span[0] <= 0.001 <= span[1]:
                                    temp_range = np.power([span[1], 0.001], exponent)
                                elif 0.001 <= span[0] or span[1] <= -0.001:
                                    x = np.power(span, exponent)
                                    temp_range = np.array([np.min(x), np.max(x)])
                                elif span[0] <= -0.001 <= span[1] <= 0.001:
                                    temp_range = np.power([-0.001, span[0]], exponent)
                                elif -0.001 <= span[0] <= span[1] <= 0.001:
                                    temp_range = [-1, 1]  # 无效
                        # print(f'pow temp_range:{temp_range}')
                    elif parent_name == 'max':
                        for i in range(arity):
                            if isinstance(program[- 1 - i], _Function):
                                span = program[- 1 - i].value_range
                            else:
                                span = np.array(self.variable_range)
                            if i == 0:
                                temp_range = span
                            else:
                                temp_range = np.array([np.max([temp_range[0], span[0]]),
                                                       np.max([temp_range[1], span[1]])])
                    elif parent_name == 'min':
                        for i in range(arity):
                            if isinstance(program[- 1 - i], _Function):
                                span = program[- 1 - i].value_range
                            else:
                                span = np.array(self.variable_range)
                            if i == 0:
                                temp_range = span
                            else:
                                temp_range = np.array([np.min([temp_range[0], span[0]]),
                                                       np.min([temp_range[1], span[1]])])
                    elif parent_name in ['sin', 'cos', 'tan']:  # 三角函数
                        temp_range = np.array([-1, 1])
                    elif parent_name == 'exp':
                        if isinstance(program[-1], _Function):
                            span = program[-1].value_range
                        else:
                            span = np.array(self.variable_range)
                        temp_range = np.exp(span)
                    elif parent_name == 'neg':
                        if isinstance(program[-1], _Function):
                            span = program[-1].value_range
                        else:
                            span = np.array(self.variable_range)
                        temp_range = np.array([-span[1], -span[0]])
                    elif parent_name == 'log':
                        if isinstance(program[-1], _Function):
                            span = program[-1].value_range
                        else:
                            span = np.array(self.variable_range)
                        if span[0] <= 0 <= span[1]:
                            temp_range = np.log([0.001, np.max(np.abs(span))])
                        elif span[1] <= 0:
                            temp_range = np.log(np.abs([np.min([-0.001, span[1]]), span[0]]))
                        elif span[0] >= 0:
                            temp_range = np.log([np.max([0.001, span[0]]), span[1]])
                    elif parent_name == 'sqrt':
                        if isinstance(program[-1], _Function):
                            span = program[-1].value_range
                        else:
                            span = np.array(self.variable_range)
                        if span[0] <= 0 <= span[1]:
                            temp_range = np.sqrt([0, np.max(np.abs(span))])
                        elif span[1] <= 0:
                            temp_range = np.sqrt(np.abs([span[1], span[0]]))
                        elif span[0] >= 0:
                            temp_range = np.sqrt(span)
                    elif parent_name == 'abs':
                        if isinstance(program[-1], _Function):
                            span = program[-1].value_range
                        else:
                            span = np.array(self.variable_range)
                        if span[0] <= 0 <= span[1]:
                            temp_range = np.array([0, np.max(np.abs(span))])
                        elif span[1] <= 0:
                            temp_range = np.abs([span[1], span[0]])
                        elif span[0] >= 0:
                            temp_range = span
                    else:
                        raise ValueError('No function matched.')
                    # 防止过大的值域估算
                    if temp_range[1] >= 1e3:
                        temp_range[1] = 1e3
                    if temp_range[0] <= -1e3:
                        temp_range[0] = -1e3
                    temp_range = np.array(temp_range, dtype=np.int32)
                    # print(f'temp_range 1:{temp_range}')
                    program[parent_index].value_range = temp_range
                    parent_index = self.find_parent(parent_index, program)  # 找下一个父节点
                    parent_name = program[parent_index].name  # 父节点函数名字
                if subtree_complete:
                    # print(f'temp_range:{temp_range}')
                    parent.value_range = temp_range
        # We should never get here
        return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        current_depth = 0
        for node in self.program:
            if isinstance(node, _Function):
                if current_depth != node.depth:
                    print("depth error: ", end='')
                assert current_depth == node.depth  # 保证深度属性不出错
                terminals.append(node.arity)
                current_depth += 1
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    current_depth -= 1
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '[' + str(node.output_dimension) + ',' + str(node.input_dimension) + ']' + '('
            else:
                if isinstance(node, tuple):  # 变量向量
                    if self.feature_names is None:
                        output += 'X[%s:%s:%s]' % (node[0], node[1], node[2])
                    else:  # 暂不修改
                        output += self.feature_names[node]
                else:  # 常数向量，但是list类型
                    output += '('
                    for num in node[0]:  # 去掉外层list
                        output += '%.3f,' % num
                    output += ')'
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    output += ')'
                    terminals[-1] -= 1
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):  # 变量分量
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:  # 常数
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):  # arity !!!
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:  #
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def execute(self, X):  # X是一个由多个输入向量组成的矩阵
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]  # 单节点没有什么意义
        if isinstance(node, list):  # 常数向量检测，检测np.ndarray类型
            print('constant')
            return np.repeat(node[0][0], X.shape[0])  # 对每个输入向量返回一个实数
        if isinstance(node, tuple):  # 变量向量检测
            print('variable')
            return X[:, node[0]]

        apply_stack = []
        # 输出每个个体
        # print(self.__str__())
        # printout(self.program)
        # print_formula(self.program, show_operand=True)
        for index, node in enumerate(self.program):
            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)
            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:  # 操作数凑齐时开始计算
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = []
                for t in apply_stack[-1][1:]:
                    if isinstance(t, list):  # 常数向量改为list[ndarray]类型，避免了后续的混淆
                        temp = np.repeat(t, X.shape[0], axis=0)  # n_samples x dimension of t
                        # 调整维度顺序，n_samples调整为最后一维，因为sum和prod等aggregate函数是按axis=0来进行计算的
                        temp = np.transpose(temp, axes=(1, 0))  # dimension x n_samples
                        terminals.append(temp)
                    elif isinstance(t, tuple):
                        temp = X[:, t[0]:t[1]:t[2]]  # n_samples x dimension of t
                        # 调整维度顺序，n_samples调整为最后一维，因为sum和prod等aggregate函数是按axis=0来进行计算的
                        temp = np.transpose(temp, axes=(1, 0))  # dimension x n_samples
                        terminals.append(temp)
                    else:  # 中间结果，即np.ndarray类型，无需额外处理
                        terminals.append(t)  # arity x dimension x n_samples
                # 聚集函数要保证不在样本数维度上做聚集计算，arity>1时在各个操作数维度上进行聚集计算，arity=1时在特征数维度上进行聚集计算
                if function.name in ['sum', 'prod', 'mean']:
                    terminals = np.array(terminals)
                    # arity>1时sum和prod保持输入和输出维度相同，arity=1时输入为向量，输出为实数
                    if terminals.ndim > 2 and terminals.shape[0] == 1:
                        # arity为1时去掉操作数维度，输出结果会少一个维度，此时要统一格式，将增加大小为1的特征数维度
                        intermediate_result = function(terminals[0])
                        intermediate_result = intermediate_result.reshape(1, -1)
                        # print(f'aggregate1: {intermediate_result.shape}')
                    else:  # arity>1时与其他函数输出结果的shape相同
                        intermediate_result = function(terminals)
                        # print(f'aggregate2: {intermediate_result.shape}')
                else:
                    intermediate_result = function(*terminals)
                    # print(f'others: {intermediate_result.shape}')
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    # print(f'final:{intermediate_result.shape}')
                    return intermediate_result[0]  # 最后去掉特征数维度，只保留样本数维度
        # We should never get here
        return None

    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def raw_fitness(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.execute(X)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        # print(f'y_train:{y.shape}')
        # print(f'y_pred:{y_pred.shape}')
        raw_fitness = self.metric(y, y_pred, sample_weight)

        return raw_fitness

    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    # 根据remaining，constant_num和parent_name来筛选可交换节点
    def get_subtree(self, random_state, program=None, output_dimension=1,
                    remaining=None, constant_num=0, prohibit=None, no_root=False, min_depth=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.
        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.
        output_dimension : int
            The dimension of subtree's output
        remaining: list
        constant_num: int
        prohibit: list
        no_root: bool
        min_depth: int
        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        indices = []  # 记录(输出)维度等于output_dimension的节点的下标
        for index, node in enumerate(program):
            if output_dimension == self.calculate_dimension(node):  # 满足维度相同要求
                if isinstance(node, _Function):  # 是函数节点，则需要检查是否有remaining要求以及是否有连续嵌套限制
                    if remaining is None:
                        if prohibit is None or len(prohibit) == 0 or node.name not in prohibit:
                            if min_depth is None or self.get_depth(index=index, program=program) >= min_depth:
                                indices.append(index)
                    elif self.subtree_state_larger(remaining, node.total):  # 有remaining要求
                        if prohibit is None or len(prohibit) == 0 or node.name not in prohibit:
                            if min_depth is None or self.get_depth(index=index, program=program) >= min_depth:
                                indices.append(index)
                elif isinstance(node, list):  # 是常数节点，则需要检查是否满足constant_num的约束
                    if constant_num == 0:  # 可以添加常数节点
                        if min_depth is None or self.get_depth(index=index, program=program) >= min_depth:
                            indices.append(index)
                else:  # 变量节点
                    if min_depth is None or self.get_depth(index=index, program=program) >= min_depth:
                        indices.append(index)
        if no_root and 0 in indices:
            indices.remove(0)  # 去除根节点坐标
        if len(indices) == 0:  # 加入remaining后可能会导致没有可交叉部分
            return -1, -1  # -1, -1表示没有符合要求的子树
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(program[index], _Function) else 0.1 for index in indices])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())
        start = indices[start]
        # start = indices[random_state.randint(len(indices))]  # 随机挑选其中一个节点的索引作为起点
        if start != 0:  # 不是根节点parent_index才有意义
            parent_index = self.find_parent(start, program)  # 寻找父节点函数
            # 父节点是pow函数且子树根节点不是pow的第一操作数
            if program[parent_index].name == 'pow' and start != parent_index + 1:
                if remaining is None:  # 仅没有remaining要求可以如此更换
                    # 将子树根节点改为pow函数或其第一个操作数，若父节点pow函数是根节点，则不可以选中
                    if (no_root and parent_index == 0) or random_state.randint(0, 2) == 0:
                        start = parent_index + 1
                    else:
                        start = parent_index
                else:
                    return -1, -1  # -1, -1表示没有符合要求的子树
        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1
        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def crossover(self, donor, random_state):  # 遍历所有可能的维度大小
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        ###
        # 求两个公式所有输出维度的交集
        common_dimensions = set(self.get_output_dimensions()).intersection(set(self.get_output_dimensions(donor)))
        if len(common_dimensions) == 0:  # 交集为空。这种情况理论上不存在，但若交集只有1，那可能只是完全交换
            raise ValueError('Crossover: The intersection of output_dimensions of two trees is empty.')
        # 生成一个不重复随机索引数列
        index_list = random_state.permutation(range(len(common_dimensions)))
        prohibit = []  # 用于限制donor的子树选择
        counter = 0
        for index in index_list:  # 遍历所有可能的输出维度
            dimension = 0
            for i, item in enumerate(common_dimensions):
                if i == index:
                    dimension = item
            # start, end = self.get_subtree(random_state, output_dimension=dimension, min_depth=2)  # 前两层节点不参与突变
            start, end = self.get_subtree(random_state, output_dimension=dimension, no_root=True)  # 根节点不参与突变
            removed = range(start, end)
            if start > 0:  # 不是根节点，说明父节点是函数节点，然后父节点的remaining和constant_num才有意义
                parent_index = self.find_parent(start)  # 需要找到start的父节点的remaining
                parent_name = self.program[parent_index].name
                remaining = self.program[parent_index].remaining
                if self.program[parent_index].arity == 1 or parent_name == 'pow':
                    constant_num = 1  # arity为1的函数节点以及pow函数节点不接收常数
                else:
                    constant_num = self.program[parent_index].constant_num
            elif start == 0:  # start和parent_index相同，即start=0，为根节点，则没有要求
                parent_index = 0
                parent_name = None
                remaining = deepcopy(default_remaining)
                constant_num = 0
            else:  # start小于0，即只有根节点输出维度为1且被no_root避免导致get_subtree返回(-1, -1)，这种情况直接重新找另一个维度
                continue
            if parent_name is not None:  # 避免相同函数或互逆函数连续嵌套
                if parent_name in ['abs', 'neg', 'add', 'sum', 'sqrt']:
                    prohibit.append(parent_name)  # 防止这些函数连续嵌套
                elif parent_name == 'exp':  # 防止exp和log连续嵌套
                    prohibit.append('log')
                elif parent_name == 'log':
                    prohibit.append('exp')
            # 随机选择donor中一个指定输出维度子树，且子树状态兼容，且避免交换后同时存在两个常数子节点
            donor_start, donor_end = self.get_subtree(random_state, program=donor, output_dimension=dimension,
                                                      remaining=remaining, constant_num=constant_num,
                                                      prohibit=prohibit)
            if (donor_start, donor_end) == (-1, -1):  # 没找到符合要求的子树
                continue
            init_depth = self.get_depth(index=start)  # 求self.program子树根节点的深度
            replacement = self.set_depth(init_depth=init_depth, program=donor[donor_start:donor_end])  # 设置donor子树列表的深度
            if isinstance(replacement[0], _Function):  # 根节点是函数节点则更新remaining
                remaining = self.update_remaining(remaining, replacement[0])
                replacement[0].parent_distance = parent_index - start  # donor子树根节点的parent_distance属性也要更新
            replacement = self.set_remaining(remaining=remaining, program=replacement)  # 设置replacement的remaining属性
            donor_removed = list(set(range(len(donor))) - set(range(donor_start, donor_end)))
            assert self.calculate_dimension(replacement[0]) == self.calculate_dimension(self.program[start])  # 保证两者维度一致
            # 更新父节点和子节点相对距离
            program = deepcopy(self.program)  # 使用deepcopy防止共用错误
            offset = (donor_end - donor_start) - (end - start)
            if offset != 0:
                if start != 0:
                    fence = 0  # 从根节点开始更新parent_distance和child_distance_list属性
                    while fence < start:
                        for i, distance in enumerate(program[fence].child_distance_list[::-1]):
                            if fence + distance > start:  # 位于start后的子节点
                                program[fence].child_distance_list[- 1 - i] += offset
                                # 该子节点是函数节点，则更新parent_distance
                                if isinstance(program[fence + distance], _Function):
                                    program[fence + distance].parent_distance -= offset
                            else:  # fence + distance <= start
                                fence = fence + distance
                                break
            # 设置total值
            subtree = program[start: end]
            for i, node in enumerate(subtree):  # 遍历subtree，减去其中函数节点的total值
                program = self.set_total(index=start + i, program=program, subtract=True)
            for i, node in enumerate(replacement):  # 遍历replacement，其中函数节点的total值均归零
                replacement = self.set_total(index=i, program=replacement, subtract=True)
            new_program = program[:start] + replacement + program[end:]
            for i, node in enumerate(replacement):  # 遍历replacement，加上其中函数节点的total值
                new_program = self.set_total(index=start + i, program=new_program)
            # 对突变后的新子树进行sub和div检查，具体方法是对交换过来的新子树的每个变量节点进行父节点回溯，找start以上的第一个sub和div节点
            cancel = False
            no_cancel = False
            for i, item in enumerate(new_program[start:start+len(replacement)]):
                # print(start + i)
                if isinstance(item, tuple):  # 遍历新子树的所有变量节点来检查sub和div抵消
                    name_list = []
                    index_list = []
                    has_sub_div = False
                    temp_index = self.find_parent(start + i, new_program)
                    temp_parent = new_program[temp_index]
                    temp_name = temp_parent.name
                    while temp_parent.parent_distance != 0:  # 一直回溯到根节点
                        index_list.append(temp_index)
                        if temp_name in ['sub', 'div'] and temp_parent.depth < init_depth:
                            has_sub_div = True
                            break
                        else:  # 不是sub和div
                            name_list.append(temp_name)
                            temp_index = temp_index + temp_parent.parent_distance
                            temp_parent = new_program[temp_index]  # 父节点回溯
                            temp_name = temp_parent.name
                    # 若第一个sub或div节点是根节点
                    if temp_name in ['sub', 'div'] and temp_parent.depth < init_depth and not has_sub_div:
                        has_sub_div = True
                        index_list.append(temp_index)
                    if has_sub_div:  # 有start以上的sub和div祖先节点，则检查其另一子树有无相同子支
                        if len(index_list) == 1 and start + i != index_list[-1] + 1 or \
                                len(index_list) > 1 and index_list[-2] != index_list[-1] + 1:  # 当前点位于右子树
                            temp_index = index_list[-1] + 1  # 从左子节点开始
                        else:  # 当前点位于左子树
                            temp_index = index_list[-1] + new_program[index_list[-1]].child_distance_list[-1]  # 从右子节点开始
                        if new_program[index_list[-1]].name == 'div':
                            if len(name_list):
                                complete = False
                                name_index = 1
                                while name_list[- name_index] in ['abs', 'neg']:  # 右边跳过abs和neg
                                    name_index += 1
                                    if name_index > len(name_list):  # 等价于右边name_list为空
                                        if isinstance(new_program[temp_index], tuple):
                                            if new_program[temp_index] == new_program[start + i]:
                                                cancel = True
                                        elif isinstance(new_program[temp_index], _Function):
                                            while isinstance(new_program[temp_index], _Function) and \
                                                    new_program[temp_index].name in ['abs', 'neg']:  # 左边跳过abs和neg
                                                temp_index += 1  # abs和neg是单节点函数，左边是完整子树，不会越界
                                            if isinstance(new_program[temp_index], tuple):
                                                if new_program[temp_index] == new_program[start + i]:
                                                    cancel = True
                                        complete = True
                                        break
                                if not complete:  # complete为False说明右边还有待判断的函数，且不是abs或neg
                                    # 当前temp_index是div的左子节点，所以可以通过这种方式跳过abs和neg
                                    while isinstance(new_program[temp_index], _Function) and\
                                            new_program[temp_index].name in ['abs', 'neg']:  # 左边跳过abs和neg
                                        temp_index += 1  # abs和neg是单节点函数，左边是完整子树，不会越界
                                    if isinstance(new_program[temp_index], _Function) and\
                                            new_program[temp_index].name == name_list[- name_index]:
                                        # 相同函数名匹配后右边要重新跳过abs和neg
                                        name_index += 1
                                        if name_index <= len(name_list):
                                            while name_list[- name_index] in ['abs', 'neg']:  # 右边跳过abs和neg
                                                name_index += 1
                                                if name_index > len(name_list):  # 等价于右边name_list为空
                                                    # complete = True
                                                    break
                                        child_index_stack = [0]  # 记录还未探索的子节点索引
                                        # 这里应该要深度优先遍历完整个子树，但碰到一条符合的之后就停止，这样足矣避免完全抵消
                                        while len(child_index_stack) or complete:  # 若complete=True，则进行最后一次循环
                                            if name_index > len(name_list) and complete:
                                                children = []
                                                for c in new_program[temp_index].child_distance_list:  # 遍历子节点
                                                    if isinstance(new_program[temp_index + c], _Function):
                                                        children.append(new_program[temp_index + c].name)
                                                    elif isinstance(new_program[temp_index + c], tuple):
                                                        children.append(new_program[temp_index + c])
                                                    # 不是pow的指数向量
                                                    elif new_program[temp_index].name != 'pow' or \
                                                            c != new_program[temp_index].child_distance_list[-1]:
                                                        no_cancel = True
                                                parent = self.find_parent(start + i, new_program)  # 找到当前点的父节点
                                                children2 = []
                                                for c in new_program[parent].child_distance_list:  # 遍历子节点
                                                    if isinstance(new_program[parent + c], _Function):
                                                        children2.append(new_program[parent + c].name)
                                                    elif isinstance(new_program[parent + c], tuple):
                                                        children2.append(new_program[parent + c])
                                                    # 不是pow的指数向量
                                                    elif new_program[parent].name != 'pow' or \
                                                            c != new_program[parent].child_distance_list[-1]:
                                                        no_cancel = True
                                                if no_cancel:
                                                    break
                                                if new_program[temp_index].name == new_program[parent].name:
                                                    # 子节点组合匹配
                                                    if new_program[temp_index].name in ['add', 'mul', 'max', 'min',
                                                                                        'sum', 'prod', 'mean']:
                                                        # 只对非常数节点做检测，若子节点组合相同，则cancel=True
                                                        cancel = True
                                                        for c in children2:
                                                            if c not in children:  # 有一个不在
                                                                cancel = False
                                                                break
                                                            else:
                                                                children.remove(c)
                                                    else:  # 其他函数则进行序列匹配，pow的指数不会被children和children2记录，要单独检测
                                                        if new_program[temp_index].name == 'pow':  # pow要对指数进行检测
                                                            cancel = False
                                                            if isinstance(new_program[parent + 1], tuple):
                                                                if new_program[parent + 1] == new_program[temp_index + 1]:
                                                                    right_index1 = temp_index + \
                                                                                   new_program[temp_index].child_distance_list[-1]
                                                                    right_index2 = parent + \
                                                                                   new_program[parent].child_distance_list[-1]
                                                                    if np.array_equal(new_program[right_index1],
                                                                                      new_program[right_index2]):
                                                                        cancel = True
                                                        else:
                                                            cancel = True
                                                            for c_i, c in enumerate(children2):  # 序列匹配
                                                                if c != children[c_i]:
                                                                    cancel = False
                                                                    break
                                                # div左右函数名不相同如何判断抵消？
                                                else:  # 左右函数名不相同，new_program[parent](即右边)一定是neg或abs函数
                                                    cancel = True
                                                    for c_i, c in enumerate(children):
                                                        if c != children2[0]:
                                                            cancel = False
                                                            break
                                                break
                                            match = False
                                            fence = child_index_stack.pop()  # 获取最新的
                                            temp_children = new_program[temp_index].child_distance_list
                                            for j, c in enumerate(temp_children):
                                                if j < fence:
                                                    continue
                                                if isinstance(new_program[temp_index + c], _Function):
                                                    if new_program[temp_index + c].name in ['abs', 'neg']:
                                                        match = True
                                                        child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                                        child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                                        temp_index += c
                                                        break
                                                    elif name_index <= len(name_list) and\
                                                            new_program[temp_index + c].name == name_list[- name_index]:
                                                        match = True
                                                        child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                                        child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                                        temp_index += c
                                                        name_index += 1  # 搜索下一个不是abs和neg的函数名字
                                                        # 函数名匹配成功后右边重新跳过abs和neg
                                                        if name_index <= len(name_list):
                                                            while name_list[- name_index] in ['abs', 'neg']:
                                                                name_index += 1
                                                                if name_index > len(name_list):  # 等价于右边name_list为空
                                                                    # 且左节点的子节点中没有abs和neg，则complete=True
                                                                    break
                                                        break
                                            if not match:  # 这个当前点的子节点没有匹配的
                                                if name_index > len(name_list):  # 这表明没有abs或neg函数，则可以停止搜索
                                                    complete = True
                                                else:  # 否则还需要返回至当前点的父节点
                                                    temp_index += new_program[temp_index].parent_distance
                                                    name_index -= 1  # 搜索上一个名字
                                                    while new_program[temp_index].name in ['neg', 'abs'] and \
                                                            new_program[temp_index].parent_distance != 0:
                                                        if len(child_index_stack):
                                                            child_index_stack.pop()  # 去掉neg和abs记录的下一探索点
                                                        else:  # 超出界限说明已经不满足抵消了
                                                            break
                                                        temp_index += new_program[temp_index].parent_distance
                                                        name_index -= 1  # 搜索上一个名字
                            else:  # 右边name_list为空
                                if isinstance(new_program[temp_index], tuple):
                                    if new_program[temp_index] == new_program[start + i]:
                                        cancel = True
                                else:
                                    while isinstance(new_program[temp_index], _Function) and \
                                            new_program[temp_index].name in ['abs', 'neg']:
                                        temp_index += 1
                                    if isinstance(new_program[temp_index], tuple):
                                        if new_program[temp_index] == new_program[start + i]:
                                            cancel = True
                        elif len(name_list):  # 与sub之间存在中间函数节点
                            if isinstance(new_program[temp_index], _Function) and \
                                    new_program[temp_index].name == name_list[-1]:
                                child_index_stack = [0]  # 记录还未探索的子节点索引
                                complete = False
                                name_index = 1
                                while len(child_index_stack):  # 这里应该要深度优先遍历完整个子树，而不是只遍历一个子支
                                    match = False
                                    # 找到了父节点均相同的子支
                                    if complete or name_index + 1 > len(name_list):
                                        children = []
                                        for c in new_program[temp_index].child_distance_list:  # 遍历子节点
                                            if isinstance(new_program[temp_index + c], _Function):
                                                children.append(new_program[temp_index + c].name)
                                            elif isinstance(new_program[temp_index + c], tuple):
                                                children.append(new_program[temp_index + c])
                                            elif new_program[temp_index].name != 'pow' or \
                                                    c != new_program[temp_index].child_distance_list[-1]:  # 不是pow的指数向量
                                                no_cancel = True
                                        parent = self.find_parent(start + i, new_program)  # 找到当前点的父节点
                                        children2 = []
                                        for c in new_program[parent].child_distance_list:  # 遍历子节点
                                            if isinstance(new_program[parent + c], _Function):
                                                children2.append(new_program[parent + c].name)
                                            elif isinstance(new_program[parent + c], tuple):
                                                children2.append(new_program[parent + c])
                                            elif new_program[parent].name != 'pow' or \
                                                    c != new_program[parent].child_distance_list[-1]:  # 不是pow的指数向量
                                                no_cancel = True
                                        if no_cancel:
                                            break
                                        if new_program[temp_index].name == new_program[parent].name:
                                            # 子节点组合匹配
                                            if new_program[temp_index].name in ['add', 'mul', 'max', 'min',
                                                                                'sum', 'prod', 'mean']:
                                                # 只对非常数节点做检测，若子节点组合相同，则cancel=True
                                                cancel = True
                                                for c in children2:
                                                    if c not in children:  # 有一个不在
                                                        cancel = False
                                                        break
                                                    else:
                                                        children.remove(c)
                                            else:  # 其他函数则进行序列匹配，pow的指数不会被children和children2记录，要单独检测
                                                if new_program[temp_index].name == 'pow':  # pow要对指数进行检测
                                                    cancel = False
                                                    if isinstance(new_program[parent + 1], tuple):
                                                        if new_program[parent + 1] == new_program[temp_index + 1]:
                                                            right_index1 = temp_index + \
                                                                           new_program[temp_index].child_distance_list[
                                                                               -1]
                                                            right_index2 = parent + \
                                                                           new_program[parent].child_distance_list[-1]
                                                            if np.array_equal(new_program[right_index1],
                                                                              new_program[right_index2]):
                                                                cancel = True
                                                else:
                                                    cancel = True
                                                    for c_i, c in enumerate(children2):  # 序列匹配
                                                        if c != children[c_i]:
                                                            cancel = False
                                                            break
                                        break
                                    temp_name = name_list[- name_index - 1]
                                    fence = child_index_stack.pop()  # 获取最新的
                                    temp_children = new_program[temp_index].child_distance_list
                                    for j, c in enumerate(temp_children):
                                        if j < fence:
                                            continue
                                        if isinstance(new_program[temp_index + c], _Function) and \
                                                new_program[temp_index + c].name == temp_name:  # 找到了一个匹配的子节点
                                            child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                            if len(child_index_stack) + 1 == len(name_list):  # 完全匹配成功
                                                complete = True
                                            temp_index = temp_index + c
                                            child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                            name_index += 1  # 搜索下一个名字
                                            match = True
                                            break
                                    if not match:  # 这个当前点的子节点没有匹配的，则返回至当前点的父节点
                                        temp_index = temp_index + new_program[temp_index].parent_distance  # 确保初始值正确
                                        name_index -= 1  # 搜索上一个名字
                        else:  # 与sub之间没有中间函数节点，则temp_index和start + i分别为sub或div的左右子节点
                            if isinstance(new_program[temp_index], tuple) and \
                                    new_program[temp_index] == new_program[start + i]:
                                cancel = True
                    if not cancel:  # 若没有抵消，则可以突变
                        break
            if cancel:  # 有sub和div的抵消，则重新突变
                counter += 1
                if counter >= 6:  # 尝试次数超过5次就停止，不交叉突变
                    return deepcopy(self.program), 0, 0
                continue
            y_pred = self.execute_test(new_program, X_train)
            if np.max(y_pred) - np.min(y_pred) <= 1e-8 and check_constant_function:
                print("-----常数函数(crossover)-----")
                printout(self.program)
                printout(replacement)
                printout(new_program)
                print_formula(new_program, show_operand=True)
            temp_constant_num = 0
            for i in new_program[parent_index].child_distance_list:  # 重新对start父节点的常数子节点进行计数
                if isinstance(new_program[parent_index + i], list):
                    temp_constant_num += 1
            new_program[parent_index].constant_num = temp_constant_num
            return new_program, removed, donor_removed
        return deepcopy(self.program), 0, 0

    def subtree_mutation(self, random_state):  # 子树突变
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # 挑选一个随机输出维度子树
        output_dimensions = self.get_output_dimensions()
        # 生成一个不重复随机索引数列
        index_list = random_state.permutation(range(len(output_dimensions)))
        prohibit = []
        counter = 0
        for index in index_list:
            output_dimension = 0
            for i, item in enumerate(output_dimensions):
                if i == index:
                    output_dimension = item
            # hoist突变可以对整个树进行裁剪[，前两层节点不参与突变]
            start, end = self.get_subtree(random_state, output_dimension=output_dimension)  # , min_depth=2
            if (start, end) != (-1, -1):
                break
            subtree = self.program[start:end]
            if start > 0:  # 不是根节点，说明父节点是函数节点，然后父节点的remaining和constant_num才有意义
                parent_index = self.find_parent(start)  # 需要找到start的父节点的remaining
                parent_name = self.program[parent_index].name
                remaining = self.program[parent_index].remaining
                if self.program[parent_index].arity == 1 or parent_name == 'pow':
                    constant_num = 1  # arity为1的函数节点以及pow函数节点不接收常数
                else:
                    constant_num = self.program[parent_index].constant_num
            elif start == 0:  # start和parent_index相同，即start=0，为根节点，则没有要求
                parent_index = 0
                parent_name = None
                remaining = deepcopy(default_remaining)
                constant_num = 0  # 0表示可以生成常数
            else:  # start小于0，即只有根节点输出维度为1且被no_root避免导致get_subtree返回(-1, -1)，这种情况直接重新找另一个维度
                continue
            if parent_name is not None:  # 避免相同函数或互逆函数连续嵌套
                if parent_name in ['abs', 'neg', 'add', 'sum', 'sqrt']:
                    prohibit.append(parent_name)  # 防止这些函数连续嵌套
                elif parent_name == 'exp':  # 防止exp和log连续嵌套
                    prohibit.append('log')
                elif parent_name == 'log':
                    prohibit.append('exp')
            sub_start, sub_end = self.get_subtree(random_state, program=subtree,
                                                  output_dimension=self.calculate_dimension(self.program[start]),
                                                  remaining=remaining, constant_num=constant_num,
                                                  prohibit=prohibit, no_root=True)  # 不能选择根节点，否则没有变化
            if (sub_start, sub_end) == (-1, -1):  # 没找到符合要求的子树
                continue
            init_depth = self.get_depth(start, self.program)
            hoist = self.set_depth(init_depth, subtree[sub_start:sub_end])  # 设置hoist各点的depth属性
            if isinstance(hoist[0], _Function):  # 根节点是函数节点则更新remaining
                remaining = self.update_remaining(remaining, hoist[0])
                hoist[0].parent_distance = parent_index - start  # hoist子树根节点的parent_distance属性也要更新
            hoist = self.set_remaining(remaining, hoist)  # 设置hoist各点的remaining属性
            # 更新父节点和子节点相对距离
            program = deepcopy(self.program)
            offset = (sub_end - sub_start) - (end - start)
            if offset != 0:
                if start != 0:
                    fence = 0  # 从根节点开始
                    while fence < start:
                        for i, distance in enumerate(program[fence].child_distance_list[::-1]):
                            if fence + distance > start:  # 位于start后的子节点
                                if isinstance(program[fence], _Function):
                                    program[fence].child_distance_list[- 1 - i] += offset
                                if isinstance(program[fence + distance], _Function):  # 该子节点是函数节点，则更新parent_distance
                                    program[fence + distance].parent_distance -= offset
                            else:  # fence + distance <= start
                                fence = fence + distance
                                break
            # 修改被hoist的部分的ancestors的total值
            for i, node in enumerate(subtree):  # 遍历subtree，减去其中函数节点的total值
                program = self.set_total(index=start + i, program=program, subtract=True)  # 对program变量逐步更新
            for i, node in enumerate(hoist):  # 遍历hoist，减去其中函数节点的total值
                hoist = self.set_total(index=i, program=hoist, subtract=True)  # hoist的total值归零
            new_program = program[:start] + hoist + program[end:]
            for i, node in enumerate(hoist):  # 遍历hoist，加上其中函数节点的total值
                new_program = self.set_total(index=start + i, program=new_program)
            # 对突变后的新子树进行sub和div检查，具体方法是对交换过来的新子树的每个变量节点进行父节点回溯，找start以上的第一个sub和div节点
            cancel = False
            no_cancel = False
            for i, item in enumerate(new_program[start:start + len(hoist)]):  # 遍历新子树的所有变量节点
                # print(start + i)
                if isinstance(item, tuple):
                    name_list = []
                    index_list = []
                    has_sub_div = False
                    temp_index = self.find_parent(start + i, new_program)
                    temp_parent = new_program[temp_index]
                    temp_name = temp_parent.name
                    while temp_parent.parent_distance != 0:  # 一直回溯到根节点
                        index_list.append(temp_index)
                        if temp_name in ['sub', 'div'] and temp_parent.depth < init_depth:
                            has_sub_div = True
                            break
                        else:  # 不是sub和div
                            name_list.append(temp_name)
                            temp_index = temp_index + temp_parent.parent_distance
                            temp_parent = new_program[temp_index]  # 父节点回溯
                            temp_name = temp_parent.name
                    # 若第一个sub或div节点是根节点
                    if temp_name in ['sub', 'div'] and temp_parent.depth < init_depth and not has_sub_div:
                        has_sub_div = True
                        index_list.append(temp_index)
                    if has_sub_div:  # 有start以上的sub和div祖先节点，则检查其另一子树有无相同子支
                        if len(index_list) == 1 and start + i != index_list[-1] + 1 or \
                                len(index_list) > 1 and index_list[-2] != index_list[-1] + 1:  # 当前点位于右子树
                            temp_index = index_list[-1] + 1  # 从左子节点开始
                        else:  # 当前点位于左子树
                            temp_index = index_list[-1] + new_program[index_list[-1]].child_distance_list[-1]  # 从右子节点开始
                        if new_program[index_list[-1]].name == 'div':
                            if len(name_list):
                                complete = False
                                name_index = 1
                                while name_list[- name_index] in ['abs', 'neg']:  # 右边跳过abs和neg
                                    name_index += 1
                                    if name_index > len(name_list):  # 等价于右边name_list为空
                                        if isinstance(new_program[temp_index], tuple):
                                            if new_program[temp_index] == new_program[start + i]:
                                                cancel = True
                                        elif isinstance(new_program[temp_index], _Function):
                                            while isinstance(new_program[temp_index], _Function) and \
                                                    new_program[temp_index].name in ['abs', 'neg']:  # 左边跳过abs和neg
                                                temp_index += 1  # abs和neg是单节点函数，左边是完整子树，不会越界
                                            if isinstance(new_program[temp_index], tuple):
                                                if new_program[temp_index] == new_program[start + i]:
                                                    cancel = True
                                        complete = True
                                        break
                                if not complete:  # complete为False说明右边还有待判断的函数，且不是abs或neg
                                    # 当前temp_index是div的左子节点，所以可以通过这种方式跳过abs和neg
                                    while isinstance(new_program[temp_index], _Function) and \
                                            new_program[temp_index].name in ['abs', 'neg']:  # 左边跳过abs和neg
                                        temp_index += 1  # abs和neg是单节点函数，左边是完整子树，不会越界
                                    if isinstance(new_program[temp_index], _Function) and \
                                            new_program[temp_index].name == name_list[- name_index]:
                                        # 相同函数名匹配后右边要重新跳过abs和neg
                                        name_index += 1
                                        if name_index <= len(name_list):
                                            while name_list[- name_index] in ['abs', 'neg']:  # 右边跳过abs和neg
                                                name_index += 1
                                                if name_index > len(name_list):  # 等价于右边name_list为空
                                                    # complete = True
                                                    break
                                        child_index_stack = [0]  # 记录还未探索的子节点索引
                                        # 这里应该要深度优先遍历完整个子树，但碰到一条符合的之后就停止，这样足矣避免完全抵消
                                        while len(child_index_stack) or complete:  # 若complete=True，则进行最后一次循环
                                            if name_index > len(name_list) and complete:
                                                children = []
                                                for c in new_program[temp_index].child_distance_list:  # 遍历子节点
                                                    if isinstance(new_program[temp_index + c], _Function):
                                                        children.append(new_program[temp_index + c].name)
                                                    elif isinstance(new_program[temp_index + c], tuple):
                                                        children.append(new_program[temp_index + c])
                                                    # 不是pow的指数向量
                                                    elif new_program[temp_index].name != 'pow' or \
                                                            c != new_program[temp_index].child_distance_list[-1]:
                                                        no_cancel = True
                                                parent = self.find_parent(start + i, new_program)  # 找到当前点的父节点
                                                children2 = []
                                                for c in new_program[parent].child_distance_list:  # 遍历子节点
                                                    if isinstance(new_program[parent + c], _Function):
                                                        children2.append(new_program[parent + c].name)
                                                    elif isinstance(new_program[parent + c], tuple):
                                                        children2.append(new_program[parent + c])
                                                    # 不是pow的指数向量
                                                    elif new_program[parent].name != 'pow' or \
                                                            c != new_program[parent].child_distance_list[-1]:
                                                        no_cancel = True
                                                if no_cancel:
                                                    break
                                                if new_program[temp_index].name == new_program[parent].name:
                                                    # 子节点组合匹配
                                                    if new_program[temp_index].name in ['add', 'mul', 'max', 'min',
                                                                                        'sum', 'prod', 'mean']:
                                                        # 只对非常数节点做检测，若子节点组合相同，则cancel=True
                                                        cancel = True
                                                        for c in children2:
                                                            if c not in children:  # 有一个不在
                                                                cancel = False
                                                                break
                                                            else:
                                                                children.remove(c)
                                                    else:  # 其他函数则进行序列匹配，pow的指数不会被children和children2记录，要单独检测
                                                        if new_program[temp_index].name == 'pow':  # pow要对指数进行检测
                                                            cancel = False
                                                            if isinstance(new_program[parent + 1], tuple):
                                                                if new_program[parent + 1] == new_program[temp_index + 1]:
                                                                    right_index1 = temp_index + \
                                                                                   new_program[temp_index].child_distance_list[-1]
                                                                    right_index2 = parent + \
                                                                                   new_program[parent].child_distance_list[-1]
                                                                    if np.array_equal(new_program[right_index1],
                                                                                      new_program[right_index2]):
                                                                        cancel = True
                                                        else:
                                                            cancel = True
                                                            for c_i, c in enumerate(children2):  # 序列匹配
                                                                if c != children[c_i]:
                                                                    cancel = False
                                                                    break
                                                # div左右函数名不相同如何判断抵消？
                                                else:  # 左右函数名不相同，new_program[parent](即右边)一定是neg或abs函数
                                                    cancel = True
                                                    for c_i, c in enumerate(children):
                                                        if c != children2[0]:
                                                            cancel = False
                                                            break
                                                break
                                            match = False
                                            fence = child_index_stack.pop()  # 获取最新的
                                            temp_children = new_program[temp_index].child_distance_list
                                            for j, c in enumerate(temp_children):
                                                if j < fence:
                                                    continue
                                                if isinstance(new_program[temp_index + c], _Function):
                                                    if new_program[temp_index + c].name in ['abs', 'neg']:
                                                        match = True
                                                        child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                                        child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                                        temp_index += c
                                                        break
                                                    elif name_index <= len(name_list) and \
                                                            new_program[temp_index + c].name == name_list[- name_index]:
                                                        match = True
                                                        child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                                        child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                                        temp_index += c
                                                        name_index += 1  # 搜索下一个不是abs和neg的函数名字
                                                        # 函数名匹配成功后右边重新跳过abs和neg
                                                        if name_index <= len(name_list):
                                                            while name_list[- name_index] in ['abs', 'neg']:
                                                                name_index += 1
                                                                if name_index > len(name_list):  # 等价于右边name_list为空
                                                                    # 且左节点的子节点中没有abs和neg，则complete=True
                                                                    break
                                                        break
                                            if not match:  # 这个当前点的子节点没有匹配的
                                                if name_index > len(name_list):  # 这表明没有abs或neg函数，则可以停止搜索
                                                    complete = True
                                                else:  # 否则还需要返回至当前点的父节点
                                                    temp_index += new_program[temp_index].parent_distance
                                                    name_index -= 1  # 搜索上一个名字
                                                    while new_program[temp_index].name in ['neg', 'abs'] and \
                                                            new_program[temp_index].parent_distance != 0:
                                                        if len(child_index_stack):
                                                            child_index_stack.pop()  # 去掉neg和abs记录的下一探索点
                                                        else:  # 超出界限说明已经不满足抵消了
                                                            break
                                                        temp_index += new_program[temp_index].parent_distance
                                                        name_index -= 1  # 搜索上一个名字
                            else:  # 右边name_list为空
                                if isinstance(new_program[temp_index], tuple):
                                    if new_program[temp_index] == new_program[start + i]:
                                        cancel = True
                                else:
                                    while isinstance(new_program[temp_index], _Function) and \
                                            new_program[temp_index].name in ['abs', 'neg']:
                                        temp_index += 1
                                    if isinstance(new_program[temp_index], tuple):
                                        if new_program[temp_index] == new_program[start + i]:
                                            cancel = True
                        elif len(name_list):  # 与sub之间存在中间函数节点
                            if isinstance(new_program[temp_index], _Function) and \
                                    new_program[temp_index].name == name_list[-1]:
                                child_index_stack = [0]  # 记录还未探索的子节点索引
                                complete = False
                                name_index = 1
                                while len(child_index_stack):  # 这里应该要深度优先遍历完整个子树，而不是只遍历一个子支
                                    match = False
                                    # 找到了父节点均相同的子支
                                    if complete or name_index + 1 > len(name_list):
                                        children = []
                                        for c in new_program[temp_index].child_distance_list:  # 遍历子节点
                                            if isinstance(new_program[temp_index + c], _Function):
                                                children.append(new_program[temp_index + c].name)
                                            elif isinstance(new_program[temp_index + c], tuple):
                                                children.append(new_program[temp_index + c])
                                            elif new_program[temp_index].name != 'pow' or \
                                                    c != new_program[temp_index].child_distance_list[-1]:  # 不是pow的指数向量
                                                no_cancel = True
                                        parent = self.find_parent(start + i, new_program)  # 找到当前点的父节点
                                        children2 = []
                                        for c in new_program[parent].child_distance_list:  # 遍历子节点
                                            if isinstance(new_program[parent + c], _Function):
                                                children2.append(new_program[parent + c].name)
                                            elif isinstance(new_program[parent + c], tuple):
                                                children2.append(new_program[parent + c])
                                            elif new_program[parent].name != 'pow' or \
                                                    c != new_program[parent].child_distance_list[-1]:  # 不是pow的指数向量
                                                no_cancel = True
                                        if no_cancel:
                                            break
                                        if new_program[temp_index].name == new_program[parent].name:
                                            # 子节点组合匹配
                                            if new_program[temp_index].name in ['add', 'mul', 'max', 'min',
                                                                                'sum', 'prod', 'mean']:
                                                # 只对非常数节点做检测，若子节点组合相同，则cancel=True
                                                cancel = True
                                                for c in children2:
                                                    if c not in children:  # 有一个不在
                                                        cancel = False
                                                        break
                                                    else:
                                                        children.remove(c)
                                            else:  # 其他函数则进行序列匹配，pow的指数不会被children和children2记录，要单独检测
                                                if new_program[temp_index].name == 'pow':  # pow要对指数进行检测
                                                    cancel = False
                                                    if isinstance(new_program[parent + 1], tuple):
                                                        if new_program[parent + 1] == new_program[temp_index + 1]:
                                                            right_index1 = temp_index + \
                                                                           new_program[temp_index].child_distance_list[
                                                                               -1]
                                                            right_index2 = parent + \
                                                                           new_program[parent].child_distance_list[-1]
                                                            if np.array_equal(new_program[right_index1],
                                                                              new_program[right_index2]):
                                                                cancel = True
                                                else:
                                                    cancel = True
                                                    for c_i, c in enumerate(children2):  # 序列匹配
                                                        if c != children[c_i]:
                                                            cancel = False
                                                            break
                                        break
                                    temp_name = name_list[- name_index - 1]
                                    fence = child_index_stack.pop()  # 获取最新的
                                    temp_children = new_program[temp_index].child_distance_list
                                    for j, c in enumerate(temp_children):
                                        if j < fence:
                                            continue
                                        if isinstance(new_program[temp_index + c], _Function) and \
                                                new_program[temp_index + c].name == temp_name:  # 找到了一个匹配的子节点
                                            child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                            if len(child_index_stack) + 1 == len(name_list):  # 完全匹配成功
                                                complete = True
                                            temp_index = temp_index + c
                                            child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                            name_index += 1  # 搜索下一个名字
                                            match = True
                                            break
                                    if not match:  # 这个当前点的子节点没有匹配的，则返回至当前点的父节点
                                        temp_index = temp_index + new_program[temp_index].parent_distance  # 确保初始值正确
                                        name_index -= 1  # 搜索上一个名字
                        else:  # 与sub之间没有中间函数节点，则temp_index和start + i分别为sub或div的左右子节点
                            if isinstance(new_program[temp_index], tuple) and \
                                    new_program[temp_index] == new_program[start + i]:
                                cancel = True
                    # if has_sub_div:  # 有start以上的sub和div祖先节点，则检查其另一子树有无相同子支
                    #     if len(index_list) == 1 and start + i != index_list[-1] + 1 or \
                    #             len(index_list) > 1 and index_list[-2] != index_list[-1] + 1:  # 当前点位于右子树
                    #         temp_index = index_list[-1] + 1  # 从左子节点开始
                    #     else:  # 当前点位于左子树
                    #         temp_index = index_list[-1] + new_program[index_list[-1]].child_distance_list[-1]  # 从右子节点开始
                    #     # 检测div抵消时忽略neg和abs
                    #     if program[index_list[-1]].name == 'div':
                    #         for name in name_list:  # 先去掉name_list中的neg和abs
                    #             if name in ['abs', 'neg']:
                    #                 name_list.remove(name)
                    #     # 与sub或div节点之间存在中间函数节点
                    #     if len(name_list) or new_program[index_list[-1]].name == 'div':
                    #         if isinstance(new_program[temp_index], _Function) and \
                    #                 (len(name_list) > 0 and new_program[temp_index].name == name_list[-1] or
                    #                  new_program[index_list[-1]].name == 'div' and
                    #                  new_program[temp_index].name in ['abs', 'neg']):
                    #             child_index_stack = [0]  # 记录还未探索的子节点索引
                    #             name_index = 1
                    #             complete = False
                    #             while len(child_index_stack):  # 这里应该要深度优先遍历完整个子树，而不是只遍历一个子支
                    #                 match = False
                    #                 # if complete or len(child_index_stack) + 1 > len(name_list):  # 找到了父节点均相同的子支
                    #                 if complete or name_index + 1 > len(name_list):  # 找到了父节点均相同的子支
                    #                     children = []
                    #                     for c in new_program[temp_index].child_distance_list:  # 遍历子节点
                    #                         if isinstance(new_program[temp_index + c], _Function):
                    #                             children.append(new_program[temp_index + c].name)
                    #                         elif isinstance(new_program[temp_index + c], tuple):
                    #                             children.append(new_program[temp_index + c])
                    #                         elif new_program[temp_index].name != 'pow' or \
                    #                                 c != new_program[temp_index].child_distance_list[-1]:  # 不是pow的指数向量
                    #                             no_cancel = True
                    #                         # elif new_program[temp_index].name == 'pow':  # 要记录pow的指数向量
                    #                         #     if c == new_program[temp_index].child_distance_list[-1]:
                    #                         #         children.append(new_program[temp_index + c])
                    #                     parent = self.find_parent(start + i, new_program)  # 找到当前点的父节点
                    #                     children2 = []
                    #                     for c in new_program[parent].child_distance_list:  # 遍历子节点
                    #                         if isinstance(new_program[parent + c], _Function):
                    #                             children2.append(new_program[parent + c].name)
                    #                         elif isinstance(new_program[parent + c], tuple):
                    #                             children2.append(new_program[parent + c])
                    #                         elif new_program[parent].name != 'pow' or \
                    #                                 c != new_program[parent].child_distance_list[-1]:  # 不是pow的指数向量
                    #                             no_cancel = True
                    #                         # elif new_program[parent].name == 'pow':
                    #                         #     if c == new_program[parent].child_distance_list[-1]:
                    #                         #         children2.append(new_program[parent + c])
                    #                     if no_cancel:
                    #                         break
                    #                     if new_program[temp_index].name == new_program[parent].name:
                    #                         # 子节点组合匹配
                    #                         if new_program[temp_index].name in ['add', 'mul', 'max', 'min',
                    #                                                             'sum', 'prod', 'mean']:
                    #                             # 只对非常数节点做检测，若子节点组合相同，则cancel=True
                    #                             cancel = True
                    #                             for c in children2:
                    #                                 if c not in children:  # 有一个不在
                    #                                     cancel = False
                    #                                     break
                    #                                 else:
                    #                                     children.remove(c)
                    #                         else:  # 其他函数则进行序列匹配，pow的指数不会被children和children2记录，要单独检测
                    #                             if new_program[temp_index].name == 'pow':  # pow要对指数进行检测
                    #                                 cancel = False
                    #                                 if isinstance(new_program[parent + 1], tuple):
                    #                                     if new_program[parent + 1] == new_program[temp_index + 1]:
                    #                                         right_index1 = temp_index + \
                    #                                                        new_program[temp_index].child_distance_list[
                    #                                                            -1]
                    #                                         right_index2 = parent + \
                    #                                                        new_program[parent].child_distance_list[-1]
                    #                                         if np.array_equal(new_program[right_index1],
                    #                                                           new_program[right_index2]):
                    #                                             cancel = True
                    #                             else:
                    #                                 cancel = True
                    #                                 for c_i, c in enumerate(children2):  # 序列匹配
                    #                                     if c != children[c_i]:
                    #                                         cancel = False
                    #                                         break
                    #                     else:  # 左右函数名不相同，new_program[temp_index]是neg或abs函数
                    #                         # if new_program[parent].name in ['add', 'mul', 'sum', 'prod', 'mean']:
                    #                         cancel = True
                    #                         for c_i, c in enumerate(children2):
                    #                             if c != children[0]:
                    #                                 cancel = False
                    #                                 break
                    #                     break
                    #                 # temp_name = name_list[- len(child_index_stack) - 1]
                    #                 temp_name = name_list[- name_index - 1]
                    #                 fence = child_index_stack.pop()  # 获取最新的
                    #                 children = new_program[temp_index].child_distance_list
                    #                 for j, c in enumerate(children):
                    #                     if j < fence:
                    #                         continue
                    #                     if isinstance(new_program[temp_index + c], _Function) and \
                    #                             (new_program[temp_index + c].name == temp_name or
                    #                              new_program[index_list[-1]].name == 'div' and
                    #                              new_program[temp_index].name in ['abs', 'neg']):  # 找到了一个匹配的子节点
                    #                         child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                    #                         if len(child_index_stack) == len(name_list):  # 完全匹配成功
                    #                             complete = True
                    #                         temp_index = temp_index + c
                    #                         child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                    #                         name_index += 1
                    #                         match = True
                    #                         break
                    #                 if not match:  # 这个当前点的子节点没有匹配的，则返回至当前点的父节点
                    #                     temp_index = temp_index + new_program[temp_index].parent_distance
                    #                     name_index -= 1
                    #                     if new_program[index_list[-1]].name == 'div':  # 只有div才忽略neg和abs
                    #                         while new_program[temp_index].name in ['neg', 'abs'] and \
                    #                                 new_program[temp_index].parent_distance != 0:  # 父节点是neg或abs都直接回溯
                    #                             child_index_stack.pop()  # 去掉neg和abs记录的下一探索点
                    #                             temp_index = temp_index + new_program[temp_index].parent_distance
                    #                             name_index -= 1
                    #         elif isinstance(new_program[temp_index], tuple) and len(name_list) == 0:
                    #             if new_program[temp_index] == new_program[start + i]:
                    #                 cancel = True
                    #     else:  # 与sub或div之间没有中间函数节点，则temp_index和start + i分别为sub或div的左右子节点
                    #         if isinstance(new_program[temp_index], tuple) and \
                    #                 new_program[temp_index] == new_program[start + i]:
                    #             cancel = True
                    if not cancel:  # 没有抵消则可以突变
                        break
            if cancel:  # 有sub和div的抵消，则重新突变
                counter += 1
                if counter >= 6:  # 尝试次数超过5次就停止，不进行突变
                    return deepcopy(self.program), 0
                continue
            # Determine which nodes were removed for plotting
            removed = list(set(range(start, end)) - set(range(start + sub_start, start + sub_end)))
            y_pred = self.execute_test(new_program, X_train)
            if np.max(y_pred) - np.min(y_pred) <= 1e-8 and check_constant_function:
                print("-----常数函数(hoist)-----")
                printout(self.program)
                printout(hoist)
                printout(new_program)
                print_formula(new_program, show_operand=True)
            temp_constant_num = 0
            for i in new_program[parent_index].child_distance_list:  # 重新对start父节点的常数子节点进行计数
                if isinstance(new_program[parent_index + i], list):
                    temp_constant_num += 1
            new_program[parent_index].constant_num = temp_constant_num
            return new_program, removed
        return deepcopy(self.program), 0

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # program = copy(self.program)
        program = deepcopy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]  # 返回满足条件的索引数组
        for node in mutate:  # 突变节点索引列表
            if isinstance(program[node], _Function):
                if program[node].name in ['sum', 'prod', 'pow', 'mean']:  # 不能与pow互换，因为会导致nan值的出现
                    continue  # 这几个函数节点暂时不进行点突变
                else:
                    if node != 0:
                        parent_index = self.find_parent(node, program)
                        remaining = program[parent_index].remaining  # 根据父节点的remaining来约束突变范围
                        parent_name = program[parent_index].name
                    else:
                        remaining = deepcopy(default_remaining)
                        parent_name = None
                    function_set = self.clip_function_set(remaining, self.arities[program[node].arity], no_pow=True,
                                                          parent_name=parent_name, total=program[node].total)
                    if len(function_set) != 0:  # 有可突变的函数才突变
                        origin = program[node]
                        replacement = len(function_set)
                        replacement = random_state.randint(replacement)
                        replacement = function_set[replacement]
                        replacement = new_operator(replacement, random_state,
                                                   self.n_features, origin.output_dimension,
                                                   remaining)  # 更换合适的运算符，当前节点的total值同样需要维护
                        # 要继承depth，parent_distance，child_distance_list等所有属性
                        replacement.depth = origin.depth
                        replacement.value_range = deepcopy(origin.value_range)
                        replacement.parent_distance = origin.parent_distance
                        replacement.child_distance_list = deepcopy(origin.child_distance_list)
                        replacement.constant_num = origin.constant_num

                        # 将origin替换为replacement，还需要更新ancestors的total值
                        program = self.set_total(node, program, subtract=True)  # 减去当前点以及其所有ancestors的原先函数的total值
                        replacement.total = deepcopy(origin.total)
                        program[node] = replacement
                        program = self.set_total(node, program, subtract=False)  # 然后再加上新的replacement的total增值
                        remaining = self.update_remaining(remaining, program[node])
                        program[node:] = self.set_remaining(remaining, program[node:])

                        cancel = False
                        if replacement.name in ['sub', 'div']:  # 点突变如果突变出sub或者div，也要避免抵消
                            left_child_index, right_child_index = node + 1, node + program[node].child_distance_list[-1]
                            cancel = self.check_sub_div_cancel(program=program, left_child_index=left_child_index,
                                                               right_child_index=right_child_index)  # 有任何抵消都不进行突变
                        elif node != 0:  # 非根节点突变出其他的函数也要回溯父节点进行sub和div抵消检测
                            sub_div_index = 0
                            has_sub_div = False
                            parent_index = self.find_parent(node, program)
                            temp_index = parent_index
                            temp_parent = program[temp_index]
                            temp_name = temp_parent.name
                            while temp_parent.parent_distance != 0:  # 一直回溯到根节点
                                if temp_name in ['sub', 'div']:
                                    sub_div_index = temp_index
                                    has_sub_div = True
                                    break
                                else:  # 不是sub和div
                                    temp_index = temp_index + temp_parent.parent_distance
                                    temp_parent = program[temp_index]  # 父节点回溯
                                    temp_name = temp_parent.name
                            if temp_name in ['sub', 'div'] and not has_sub_div:  # 若第一个sub或div节点是根节点
                                has_sub_div = True
                                sub_div_index = temp_index
                            if has_sub_div:
                                left_child_index = sub_div_index + 1
                                right_child_index = sub_div_index + program[sub_div_index].child_distance_list[-1]
                                cancel = self.check_sub_div_cancel(program=program, left_child_index=left_child_index,
                                                                   right_child_index=right_child_index)
                        if cancel:  # 如果出现了抵消，则替换回原来的节点
                            program = self.set_total(node, program, subtract=True)  # 减去当前点以及其所有ancestors的原先函数的total值
                            origin.total = deepcopy(replacement.total)
                            program[node] = origin
                            program = self.set_total(node, program, subtract=False)  # 然后再加上origin的total增值
                            remaining = self.update_remaining(remaining, program[node])
                            program[node:] = self.set_remaining(remaining, program[node:])
            else:  # 变量向量或常数向量，要加入对sub和div抵消的检测，若抵消则重新突变
                existed_dimension = self.calculate_dimension(program[node])  # 计算该点的维度
                parent_index = self.find_parent(node, program)  # 通过找父节点来找该点维度和父节点名字
                parent_name = program[parent_index].name

                name_list = []
                index_list = []
                has_sub_div = False
                no_cancel = False
                temp_index = parent_index
                temp_name = parent_name
                temp_parent = program[parent_index]
                prohibit = []
                # 一直回溯到根节点
                while temp_parent.parent_distance != 0:
                    index_list.append(temp_index)
                    if temp_name in ['sub', 'div']:
                        has_sub_div = True
                        break
                    else:  # 不是sub和div
                        name_list.append(temp_name)
                        temp_index = temp_index + temp_parent.parent_distance
                        temp_parent = program[temp_index]  # 父节点回溯
                        temp_name = temp_parent.name
                # 若第一个sub或div节点是根节点
                if temp_name in ['sub', 'div'] and not has_sub_div:
                    has_sub_div = True
                    index_list.append(temp_index)
                if has_sub_div:
                    # 有sub和div祖先节点且当前点位于其右子树
                    if len(index_list) == 1 and node != index_list[-1] + 1 or \
                            len(index_list) > 1 and index_list[-2] != index_list[-1] + 1:
                        temp_index = index_list[-1] + 1  # 从左子节点开始
                    else:  # 当前点位于左子树
                        temp_index = index_list[-1] + program[index_list[-1]].child_distance_list[-1]  # 从右子节点开始
                    if program[index_list[-1]].name == 'div':
                        if len(name_list):
                            complete = False
                            name_index = 1
                            while name_list[- name_index] in ['abs', 'neg']:  # 右边跳过abs和neg
                                name_index += 1
                                if name_index > len(name_list):  # 等价于右边name_list为空
                                    if isinstance(program[temp_index], tuple):
                                        prohibit.append(program[temp_index])
                                    elif isinstance(program[temp_index], _Function):
                                        while isinstance(program[temp_index], _Function) and \
                                                program[temp_index].name in ['abs', 'neg']:  # 左边跳过abs和neg
                                            temp_index += 1  # abs和neg是单节点函数，左边是完整子树，不会越界
                                        if isinstance(program[temp_index], tuple):
                                            prohibit.append(program[temp_index])
                                    complete = True
                                    break
                            if not complete:  # complete为False说明右边还有待判断的函数，且不是abs或neg
                                # 当前temp_index是div的左子节点，所以可以通过这种方式跳过abs和neg
                                while isinstance(program[temp_index], _Function) and \
                                        program[temp_index].name in ['abs', 'neg']:  # 左边跳过abs和neg
                                    temp_index += 1  # abs和neg是单节点函数，左边是完整子树，不会越界
                                if isinstance(program[temp_index], _Function) and \
                                        program[temp_index].name == name_list[- name_index]:
                                    # 相同函数名匹配后右边要重新跳过abs和neg
                                    name_index += 1
                                    if name_index <= len(name_list):
                                        while name_list[- name_index] in ['abs', 'neg']:  # 右边跳过abs和neg
                                            name_index += 1
                                            if name_index > len(name_list):  # 等价于右边name_list为空
                                                # complete = True
                                                break
                                    child_index_stack = [0]  # 记录还未探索的子节点索引
                                    # 这里应该要深度优先遍历完整个子树，但碰到一条符合的之后就停止，这样足矣避免完全抵消
                                    while len(child_index_stack) or complete:  # 若complete=True，则进行最后一次循环
                                        # if name_index + 1 > len(name_list):
                                        if name_index > len(name_list) and complete:  # and complete是为了让temp_index跳过abs和neg
                                            children = []
                                            for c in program[temp_index].child_distance_list:  # 遍历子节点
                                                if isinstance(program[temp_index + c], _Function):
                                                    children.append(program[temp_index + c].name)
                                                elif isinstance(program[temp_index + c], tuple):
                                                    children.append(program[temp_index + c])
                                                elif program[temp_index].name != 'pow' or \
                                                        c != program[temp_index].child_distance_list[-1]:  # 不是pow的指数向量
                                                    no_cancel = True
                                            children2 = []
                                            current = 0
                                            for c_i, c in enumerate(program[parent_index].child_distance_list):  # 遍历子节点
                                                if parent_index + c == node:  # 跳过当前点
                                                    current = c_i
                                                    continue
                                                if isinstance(program[parent_index + c], _Function):
                                                    children2.append(program[parent_index + c].name)
                                                elif isinstance(program[parent_index + c], tuple):
                                                    children2.append(program[parent_index + c])
                                                elif program[parent_index].name != 'pow' or \
                                                        c != program[parent_index].child_distance_list[-1]:  # 不是pow的指数向量
                                                    no_cancel = True
                                            if no_cancel:
                                                break
                                            if not len(children2):
                                                if isinstance(children[0], tuple):
                                                    prohibit.append(children[0])
                                                break
                                            if program[temp_index].name == program[parent_index].name:
                                                # 子节点组合匹配
                                                if program[temp_index].name in ['add', 'mul', 'max', 'min',
                                                                                'sum', 'prod', 'mean']:
                                                    # 只对非常数节点做检测，若子节点组合相同，则cancel=True
                                                    for c_i, c in enumerate(children2):
                                                        # if c_i == current:
                                                        #     continue
                                                        if c not in children:  # 有一个不在
                                                            break
                                                        else:
                                                            children.remove(c)
                                                    if len(children) == 1:
                                                        if isinstance(children[0], tuple):
                                                            prohibit.append(children[0])
                                                else:  # 其他函数则进行序列匹配，pow的指数不会被children和children2记录，要单独检测
                                                    if program[temp_index].name == 'pow':  # pow要对指数进行检测
                                                        if node != parent_index + 1:  # 当前点是指数向量节点
                                                            if isinstance(children[0], tuple):
                                                                if children[0] == children2[0]:
                                                                    right_index = temp_index + \
                                                                                  program[
                                                                                      temp_index].child_distance_list[
                                                                                      -1]
                                                                    prohibit.append(program[right_index])
                                                        else:  # 当前点是pow的左子节点
                                                            right_index1 = temp_index + \
                                                                           program[temp_index].child_distance_list[-1]
                                                            right_index2 = parent_index + \
                                                                           program[parent_index].child_distance_list[-1]
                                                            # 指数向量相同，则避免生成相同的左变量子节点
                                                            if np.array_equal(program[right_index1],
                                                                              program[right_index2]):
                                                                if isinstance(program[temp_index + 1], tuple):
                                                                    prohibit.append(program[temp_index + 1])
                                                    else:  # 序列匹配中需要跳过当前点
                                                        for c_i, c in enumerate(children2):  # 序列匹配
                                                            if c_i < current:
                                                                if c != children[0]:
                                                                    break
                                                                else:
                                                                    children.pop(0)
                                                            else:
                                                                if c != children[1]:
                                                                    break
                                                                else:
                                                                    children.pop(1)
                                                        if len(children) == 1 and isinstance(children[0], tuple):
                                                            prohibit.append(children[0])
                                            else:  # 只有一边是neg或abs
                                                if isinstance(children[0], tuple):
                                                    prohibit.append(children[0])
                                            break
                                        match = False
                                        fence = child_index_stack.pop()  # 获取最新的
                                        temp_children = program[temp_index].child_distance_list
                                        for j, c in enumerate(temp_children):
                                            if j < fence:
                                                continue
                                            if isinstance(program[temp_index + c], _Function):
                                                if program[temp_index + c].name in ['abs', 'neg']:
                                                    match = True
                                                    child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                                    child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                                    temp_index += c
                                                    break
                                                elif name_index <= len(name_list) and \
                                                        program[temp_index + c].name == name_list[- name_index]:
                                                    match = True
                                                    child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                                    child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                                    temp_index += c
                                                    name_index += 1  # 搜索下一个不是abs和neg的函数名字
                                                    # 函数名匹配成功后右边重新跳过abs和neg
                                                    if name_index <= len(name_list):
                                                        while name_list[- name_index] in ['abs', 'neg']:
                                                            name_index += 1
                                                            if name_index > len(name_list):  # 等价于右边name_list为空
                                                                # 且左节点的子节点中没有abs和neg，则complete=True
                                                                break
                                                    break
                                        if not match:  # 这个当前点的子节点没有匹配的
                                            if name_index > len(name_list):  # 这表明没有abs或neg函数，则可以停止搜索
                                                complete = True
                                            else:  # 否则还需要返回至当前点的父节点
                                                temp_index += program[temp_index].parent_distance
                                                name_index -= 1  # 搜索上一个名字
                                                while program[temp_index].name in ['neg', 'abs'] and \
                                                        program[temp_index].parent_distance != 0:
                                                    if len(child_index_stack):
                                                        child_index_stack.pop()  # 去掉neg和abs记录的下一探索点
                                                    else:  # 超出界限说明已经不满足抵消了
                                                        break
                                                    temp_index += program[temp_index].parent_distance
                                                    name_index -= 1  # 搜索上一个名字
                        else:  # name_list为空
                            if node == parent_index + 1:  # 当前点是父节点的左子节点
                                another = parent_index + program[parent_index].child_distance_list[-1]
                            else:
                                another = parent_index + 1
                            if isinstance(program[another], tuple):
                                prohibit.append(program[another])
                            else:
                                while isinstance(program[another], _Function) and \
                                        program[another].name in ['abs', 'neg']:
                                    another += 1
                                if isinstance(program[another], tuple):
                                    prohibit.append(program[another])
                    elif len(name_list):  # 与sub之间存在中间函数节点
                        if isinstance(program[temp_index], _Function) and program[temp_index].name == name_list[-1]:
                            child_index_stack = [0]  # 记录还未探索的子节点索引
                            name_index = 1
                            while len(child_index_stack):  # 这里应该要深度优先遍历完整个子树，而不是只遍历一个子支
                                match = False
                                # 找到了父节点均相同的子支
                                if name_index + 1 > len(name_list):
                                    children = []
                                    for c in program[temp_index].child_distance_list:  # 遍历子节点
                                        if isinstance(program[temp_index + c], _Function):
                                            children.append(program[temp_index + c].name)
                                        elif isinstance(program[temp_index + c], tuple):
                                            children.append(program[temp_index + c])
                                        elif program[temp_index].name != 'pow' or \
                                                c != program[temp_index].child_distance_list[-1]:  # 不是pow的指数向量
                                            no_cancel = True
                                    children2 = []
                                    current = 0
                                    for c_i, c in enumerate(program[parent_index].child_distance_list):  # 遍历子节点
                                        if parent_index + c == node:  # 跳过当前点
                                            current = c_i
                                            continue
                                        if isinstance(program[parent_index + c], _Function):
                                            children2.append(program[parent_index + c].name)
                                        elif isinstance(program[parent_index + c], tuple):
                                            children2.append(program[parent_index + c])
                                        elif program[parent_index].name != 'pow' or \
                                                c != program[parent_index].child_distance_list[-1]:  # 不是pow的指数向量
                                            no_cancel = True
                                    if no_cancel:
                                        break
                                    if not len(children2):
                                        if isinstance(children[0], tuple):
                                            prohibit.append(children[0])
                                        break
                                    # 子节点组合匹配
                                    if program[temp_index].name in ['add', 'mul', 'max', 'min',
                                                                    'sum', 'prod', 'mean']:
                                        # 只对非常数节点做检测，若子节点组合相同，则cancel=True
                                        for c_i, c in enumerate(children2):
                                            # if c_i == current:
                                            #     continue
                                            if c not in children:  # 有一个不在
                                                break
                                            else:
                                                children.remove(c)
                                        if len(children) == 1:
                                            if isinstance(children[0], tuple):
                                                prohibit.append(children[0])
                                    else:  # 其他函数则进行序列匹配，pow的指数不会被children和children2记录，要单独检测
                                        if program[temp_index].name == 'pow':  # pow要对指数进行检测
                                            if node != parent_index + 1:  # 当前点是指数向量节点
                                                if isinstance(children[0], tuple):
                                                    if children[0] == children2[0]:
                                                        right_index = temp_index + \
                                                                      program[temp_index].child_distance_list[-1]
                                                        prohibit.append(program[right_index])
                                            else:  # 当前点是pow的左子节点
                                                right_index1 = temp_index + \
                                                               program[temp_index].child_distance_list[-1]
                                                right_index2 = parent_index + \
                                                               program[parent_index].child_distance_list[-1]
                                                # 指数向量相同，则避免生成相同的左变量子节点
                                                if np.array_equal(program[right_index1], program[right_index2]):
                                                    if isinstance(program[temp_index + 1], tuple):
                                                        prohibit.append(program[temp_index + 1])
                                        else:  # 序列匹配中需要跳过当前点
                                            for c_i, c in enumerate(children2):  # 序列匹配
                                                if c_i < current:
                                                    if c != children[0]:
                                                        break
                                                    else:
                                                        children.pop(0)
                                                else:
                                                    if c != children[1]:
                                                        break
                                                    else:
                                                        children.pop(1)
                                            if len(children) == 1 and isinstance(children[0], tuple):
                                                prohibit.append(children[0])
                                    break
                                temp_name = name_list[- name_index - 1]
                                fence = child_index_stack.pop()  # 获取最新的
                                temp_children = program[temp_index].child_distance_list
                                for j, c in enumerate(temp_children):
                                    if j < fence:
                                        continue
                                    if isinstance(program[temp_index + c], _Function) and \
                                            program[temp_index + c].name == temp_name:  # 找到了一个匹配的子节点
                                        child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                        temp_index = temp_index + c
                                        child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                        name_index += 1  # 搜索下一个名字
                                        match = True
                                        break
                                if not match:  # 这个当前点的子节点没有匹配的，则返回至当前点的父节点
                                    temp_index = temp_index + program[temp_index].parent_distance  # 确保初始值正确
                                    name_index -= 1  # 搜索上一个名字
                    else:  # 与sub之间没有中间函数节点
                        if node == parent_index + 1:  # 当前点是父节点的左子节点
                            right = parent_index + program[parent_index].child_distance_list[-1]
                            if isinstance(program[right], tuple):
                                prohibit.append(program[right])
                        elif isinstance(program[parent_index + 1], tuple):  # 是右子节点，则避免生成与左子节点相同的点
                            prohibit.append(program[parent_index + 1])
                if parent_name in ['max', 'min']:
                    if node == parent_index + 1:  # 左子节点
                        index = parent_index + program[parent_index].child_distance_list[-1]  # 右子节点
                        if isinstance(program[index], tuple) and program[index] not in prohibit:
                            prohibit.append(program[index])
                    else:  # 右子节点
                        if isinstance(program[parent_index + 1], tuple) and program[parent_index + 1] not in prohibit:
                            prohibit.append(program[parent_index + 1])
                # 不是根节点，parent_index才有意义
                if node != 0 and parent_name == 'pow' and node != parent_index + 1:  # 若当前节点是pow的指数向量
                    terminal = self.generate_a_terminal(random_state, existed_dimension,
                                                        const_int=True, prohibit=prohibit)
                # 父节点是arity为1的函数节点或pow函数节点，或者父节点constant_num已经为1且要突变的不是常数节点，则子节点不能突变为常数
                elif program[parent_index].arity == 1 or parent_name == 'pow' or \
                        program[parent_index].constant_num >= 1 and not isinstance(program[node], list):
                    no_mutate = False
                    if len(prohibit) and existed_dimension == self.n_features:
                        for item in prohibit:
                            if isinstance(item, tuple) and (item[1] - item[0] - 1) / item[2] + 1 == self.n_features:
                                no_mutate = True  # 既不能突变为向量切片也不能突变为常数节点，则该点不突变
                                break
                    if no_mutate:
                        continue
                    terminal = self.generate_a_terminal(random_state, existed_dimension, vary=True, prohibit=prohibit)
                else:  # arity>1的函数节点
                    terminal = self.generate_a_terminal(random_state, existed_dimension, prohibit=prohibit)
                program[node] = terminal
                if node != 0:  # 当前点不是根节点，则需要维护当前点的父节点的constant_num属性
                    temp_constant_num = 0
                    for i in program[parent_index].child_distance_list:  # 重新对start父节点的常数子节点进行计数
                        if isinstance(program[parent_index + i], list):
                            temp_constant_num += 1
                    program[parent_index].constant_num = temp_constant_num
        y_pred = self.execute_test(program, X_train)
        if np.max(y_pred) - np.min(y_pred) <= 1e-8 and check_constant_function:
            print("-----常数函数(point)-----")
            printout(self.program)
            printout(program)
            print_formula(program, show_operand=True)
        return program, list(mutate)

    # 辅助函数
    def get_output_dimensions(self, program=None):
        if program is None:
            program = self.program
        output_dimensions = np.full(self.n_features, 0)
        for index, node in enumerate(program):  # 遍历该公式，找到所有不同输出维度的节点
            output_dimensions[self.calculate_dimension(node) - 1] += 1
        result = []
        for index, item in enumerate(output_dimensions):
            if item != 0:  # 有这个输出维度的节点
                result.append(index + 1)  # 记录该输出维度
        return result

    def get_depth(self, index, program=None):  # 给定的program和index，找到index对应节点(函数，常数或变量)的深度
        if program is None:
            program = self.program
        if index == 0:
            return 0
        if len(program) < index:
            raise ValueError("Get_depth: The length of program is smaller than the value of the index.")
        if isinstance(program[0], _Function):  # 初始深度应为program[0]的深度
            current_depth = program[0].depth
        else:
            current_depth = 0
        terminal_stack = []
        for i, node in enumerate(program):
            if i == index:  # 索引相同，找到了要找的点，返回当前深度
                return current_depth
            if isinstance(node, _Function):
                terminal_stack.append(node.arity)
                current_depth += 1
            else:
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    current_depth -= 1
                    terminal_stack.pop()
                    if not terminal_stack:  # terminal_stack为空时仍未找到，返回None
                        return None  # We should never get here
                    terminal_stack[-1] -= 1
        return current_depth  # 没找到该点，说明该点是最后一个点，那返回最新的current_depth即可

    def set_depth(self, init_depth, program=None):  # 给定根节点深度和program，设置program这两个属性
        if program is None:
            program = self.program
        if len(program) == 1:
            return deepcopy(program)
        current_depth = init_depth
        terminal_stack = []
        new_program = deepcopy(program)  # 深复制，避免修改共用数组导致错误
        for index, node in enumerate(program):
            if isinstance(node, _Function):
                new_program[index].depth = current_depth  # 不能返回或修改原数组，应该返回新的数组，避免修改共用数组导致错误
                terminal_stack.append(node.arity)
                current_depth += 1
            else:
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    current_depth -= 1
                    terminal_stack.pop()
                    if not terminal_stack:  # terminal_stack为空时返回program
                        return new_program
                    terminal_stack[-1] -= 1

    def set_remaining(self, remaining, program=None):  # 给定根节点remaining和program，设置program中所有点的remaining值
        if program is None:
            program = self.program
        new_program = deepcopy(program)  # 深复制，避免修改共用数组导致错误
        if isinstance(program[0], _Function):  # 函数节点
            new_program[0].remaining = remaining
        else:
            return new_program
        terminal_stack = []
        for index, node in enumerate(program):
            if isinstance(node, _Function):
                terminal_stack.append(node.arity)
                if index == 0:  # 第一个点已经设置了
                    continue
                parent_index = self.find_parent(index, new_program)
                parent_remaining = new_program[parent_index].remaining
                current_remaining = self.update_remaining(parent_remaining, node)
                new_program[index].remaining = current_remaining
            else:
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:  # terminal_stack为空时返回program
                        return new_program
                    terminal_stack[-1] -= 1
        return None  # We should never get here

    def set_total(self, index, program=None, subtract=False):  # 给定当前点索引以及program，设置当前点以及其所有ancestors的total值
        if program is None:
            program = self.program
        new_program = deepcopy(program)  # 深复制，避免修改共用数组导致错误
        if isinstance(new_program[index], _Function):  # 当前节点是函数节点才会需要修改ancestors的total值
            parent_index = self.find_parent(index, new_program)  # 找父节点，根节点会返回自己，然后会修改自己的total值，而其他节点不会
            if parent_index != index:  # 其他节点也修改自身的total值，与根节点保持一致
                new_program[index].total = self.update_total(new_program[index].total, new_program[index], subtract)
            # 给定节点total和当前函数节点，更新节点total值
            new_program[parent_index].total = self.update_total(new_program[parent_index].total,
                                                                new_program[index], subtract)  # 更新父节点total属性
            while parent_index != 0:  # 只有parent_index到0才会停止
                parent_index = self.find_parent(parent_index, new_program)  # 找父节点的父节点
                new_program[parent_index].total = self.update_total(new_program[parent_index].total,
                                                                    new_program[index], subtract)  # 更新父节点total属性
        return new_program

    # const_int指定生成pow函数需要的整数向量，vary指定生成变量向量，const_range长度不为0则需要根据const_range来生成常数。
    def generate_a_terminal(self, random_state, output_dimension, const_int=False, vary=False,
                            const_range=(), const=False, prohibit=()):
        if len(prohibit) and output_dimension == self.n_features:
            for item in prohibit:
                if isinstance(item, tuple) and (item[1] - item[0] - 1) / item[2] + 1 == self.n_features:
                    if not vary:
                        const = True  # 禁止了最大维度，则只能生成常数
                    else:
                        print('Cancel error.')
                    break
        if vary and const:
            raise ValueError('Parameter "vary" and "const" cannot be both True.')
        if not const_int:
            if not vary:
                # [0, self.n_features]，self.n_features表示选择常数向量
                if not const:
                    terminal = random_state.randint(self.n_features + 1)
                else:  # const为True生成常数
                    terminal = self.n_features
            else:  # 若vary为True，则不生成常数向量
                terminal = random_state.randint(self.n_features)
            if terminal == self.n_features:  # 常数向量，但需要外面套一层[]，使其类型变为list，避免execute函数中类型混淆
                terminal = []
                if len(const_range):  # 有大小要求则按大小要求来
                    # print(f'const_range:{const_range}')
                    for i in range(output_dimension):
                        result = random_state.randint(const_range[0], const_range[1])
                        if random_state.randint(0, 2) == 0:
                            result = -result
                        while abs(result) < 0.001:  # 不能过于接近0
                            result = random_state.randint(const_range[0], const_range[1])
                            if random_state.randint(0, 2) == 0:
                                result = -result
                        terminal.append(result)
                else:  # 随机生成时采用科学记数法
                    for i in range(output_dimension):
                        mantissa = random_state.randint(1, 12)  # 10表示π，11表示e
                        exponent = random_state.randint(-1, 4)  # 数量级范围是[-1, 3]
                        if mantissa == 10:
                            mantissa = np.pi
                        elif mantissa == 11:
                            mantissa = np.e
                        if random_state.randint(0, 2) == 0:
                            result = mantissa * 10 ** exponent
                        else:
                            result = - mantissa * 10 ** exponent
                        terminal.append(result)
                terminal = [np.array(terminal, dtype=np.float64)]  # float64数据类型，统一类型
            else:  # 随机生成向量切片，已知维度->切片
                # 随机生成合法步长，[1, ]
                same = True
                start, end, step = 0, 0, 0
                counter = 0
                while same:  # 生成与prohibit相同的变量节点时重新生成
                    same = False
                    counter += 1
                    if self.n_features > 1:
                        if output_dimension > 1:
                            # step = random_state.randint(math.floor(self.n_features / output_dimension)) + 1
                            step = 1  # 切片步长只能是1，但起点不固定
                            start = random_state.randint(self.n_features - (output_dimension - 1) * step)  # 随机选择切片起点
                            end = start + (output_dimension - 1) * step + 1  # 计算切片终点，由于左闭右开，需要+1
                        else:  # output_dimension == 1
                            step = 1
                            start = random_state.randint(self.n_features)  # 随机选择切片起点
                            end = start + 1
                    else:
                        step = 1
                        start = 0
                        end = 1
                    if len(prohibit):
                        for item in prohibit:
                            if (start, end, step) == item:  # 如果有相同的就重新生成
                                same = True
                                break
                    if counter >= 10:
                        print("Endless loop error from variables.")
                        break
                terminal = (start, end, step)  # 记录该切片
        else:  # 生成pow需要的指数整数向量
            terminal = random_state.randint(2, 5, output_dimension)
            neg = random_state.uniform(size=output_dimension)
            terminal = [np.array(np.where(neg > 0.5, terminal, -terminal))]  # 一半的概率指数变为相反数，范围{-4,-3,-2,2,3,4}
            counter = 0
            if len(prohibit):
                while np.array_equal(terminal[0], prohibit[0][0]):
                    counter += 1
                    terminal = random_state.randint(2, 5, output_dimension)
                    neg = random_state.uniform(size=output_dimension)
                    terminal = [np.array(np.where(neg > 0.5, terminal, -terminal))]  # 一半的概率指数变为相反数，范围{-4,-3,-2,2,3,4}
                    if counter >= 10:
                        print("Endless loop error from pow.")
                        break
        return terminal

    # 给定program和当前点索引，返回当前点的父节点的索引，调用了get_depth函数
    def find_parent(self, current_index, program=None):
        if current_index == 0:
            return 0
        if program is None:
            program = self.program
        current_depth = self.get_depth(current_index, program)
        if current_depth is None:
            print(current_index, end=" ")
        assert current_depth is not None
        for index in range(1, current_index + 1):  # [1, current_index]，寻找父节点函数
            if isinstance(program[current_index - index], _Function):  # 函数
                if program[current_index - index].depth == current_depth - 1:  # 若该函数节点为当前节点的父节点
                    parent_index = current_index - index  # 记录父节点函数索引
                    return parent_index

    # 给定父节点remaining，父节点名字，当前点total值和function_set，给出合法的function_set
    def clip_function_set(self, remaining, function_set=None, no_pow=False, parent_name=None, total=None):
        if function_set is None:
            function_set = self.function_set  # _Function对象列表
        if total is None:
            total = deepcopy(default_total)
        prohibit = []
        if remaining[0] <= 0 or total[0] >= default_remaining[0]:  # aggregate次数为0
            for name in aggregate:
                prohibit.append(name)
        elif remaining[0] < 3 or total[0] > 0:  # aggregate次数不足以支持prod和mean
            prohibit.append('prod')
            prohibit.append('mean')
        if remaining[1] <= 0 or total[1] >= default_remaining[1] or no_pow:  # pow次数为0或指定没有pow函数
            prohibit.append('pow')
        if remaining[2] <= 0 or total[2] >= default_remaining[2]:  # 基本初等函数次数为0
            for name in elementary_functions:
                prohibit.append(name)
        if remaining[3] <= 0 or total[3] >= default_remaining[3]:  # 剩余exp次数为0
            prohibit.append('exp')
        if parent_name is not None:  # 避免相同函数或互逆函数连续嵌套
            if parent_name in ['abs', 'neg', 'add', 'sum', 'sqrt']:
                prohibit.append(parent_name)  # 防止这些函数连续嵌套
            elif parent_name == 'exp':  # 防止exp和log连续嵌套
                prohibit.append('log')
            elif parent_name == 'log':
                prohibit.append('exp')
        if len(prohibit) == 0:
            return function_set
        new_function_set = []
        for item in function_set:
            if item.name not in prohibit:  # 不在禁止范围内
                new_function_set.append(item)
        return new_function_set  # 返回约束范围后的函数集

    @staticmethod
    def calculate_dimension(node):
        if isinstance(node, _Function):  # 函数
            return node.output_dimension
        elif isinstance(node, tuple):  # 变量向量
            return math.ceil((node[1] - node[0]) / node[2])
        elif isinstance(node, list):  # 常数向量
            return len(node[0])
        return None  # We should never get here

    @staticmethod  # 给定当前点父节点的remaining，以及当前点的函数节点，返回更新后当前点的remaining值
    def update_remaining(remaining, function):
        name = function.name
        new_remaining = deepcopy(remaining)
        if name in ['sum', 'min', 'max']:
            new_remaining[0] -= 1
        elif name in ['mean', 'prod']:
            new_remaining[0] -= 3
        elif name == 'pow':
            new_remaining[1] -= 1
        elif name in ['sin', 'cos', 'tan', 'log']:
            new_remaining[2] -= 1  # 基本初等函数次数减1
        elif name == 'exp':
            new_remaining[3] -= 1
        return new_remaining

    @staticmethod  # 给定父节点total值，以及当前函数节点，更新父节点的total值
    def update_total(total, function, subtract=False):
        name = function.name
        new_total = deepcopy(total)
        if subtract:
            if name in ['sum', 'min', 'max']:
                new_total[0] -= 1
            elif name in ['mean', 'prod']:
                new_total[0] -= 3
            elif name == 'pow':
                new_total[1] -= 1
            elif name in ['sin', 'cos', 'tan', 'log']:
                new_total[2] -= 1  # 基本初等函数次数减1
            elif name == 'exp':
                new_total[3] -= 1
        else:
            if name in ['sum', 'min', 'max']:
                new_total[0] += 1
            elif name in ['mean', 'prod']:
                new_total[0] += 3
            elif name == 'pow':
                new_total[1] += 1
            elif name in ['sin', 'cos', 'tan', 'log']:
                new_total[2] += 1  # 基本初等函数次数加1
            elif name == 'exp':
                new_total[3] += 1
        return new_total

    def check_total(self, program=None):  # 给定program，检测根节点total值是否正确
        if program is None:
            program = self.program
        if len(program) == 1:
            return True
        total = [0, 0, 0, 0]
        terminal_stack = []
        for index, node in enumerate(program):  # 遍历该树来统计相应点然后与根节点total值比较
            if isinstance(node, _Function):
                total = self.update_total(total, node)  # 统计该点
                terminal_stack.append(node.arity)
            else:
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:  # terminal_stack为空时返回program
                        return total == program[0].total
                    terminal_stack[-1] -= 1

    # left_child_index和right_child_index是sub或div的左右子节点在program中的索引
    @staticmethod
    def check_sub_div_cancel(program, left_child_index, right_child_index):
        cancel = False
        root = left_child_index - 1
        left_child, right_child = program[left_child_index], program[right_child_index]
        if isinstance(left_child, tuple) and left_child == right_child:  # 变量节点相同
            cancel = True
        # 两边进行深度优先搜索
        elif isinstance(left_child, _Function) and isinstance(right_child, _Function):
            if left_child.name == right_child.name or \
                    program[root].name == 'div' and \
                    (left_child.name in ['abs', 'neg'] or right_child.name in ['abs', 'neg']):
                left_child_index_stack = [0]  # 记录左子树还未探索的子节点索引
                right_child_index_stack = [0]  # 记录右子树还未探索的子节点索引
                # 这里应该要深度优先遍历完整个子树，而不是只遍历一个子支
                while len(left_child_index_stack) and len(right_child_index_stack):
                    match = False
                    left_fence = left_child_index_stack.pop()  # 获取最新的
                    right_fence = right_child_index_stack.pop()  # 获取最新的
                    for l_index, l_child_distance in enumerate(left_child.child_distance_list):
                        if l_index < left_fence:
                            continue
                        l_child_index = left_child_index + l_child_distance
                        l_child = program[l_child_index]
                        for r_index, r_child_distance in enumerate(right_child.child_distance_list):
                            if r_index < right_fence:
                                continue
                            r_child_index = right_child_index + r_child_distance
                            r_child = program[r_child_index]
                            if isinstance(l_child, tuple) and l_child == r_child:  # 变量节点相同
                                cancel = True
                                break
                            elif isinstance(l_child, _Function) and isinstance(r_child, _Function):
                                if program[root].name == 'div':
                                    while isinstance(l_child, _Function) and l_child.name in ['neg', 'abs']:
                                        left_child_index_stack.append(1)  # neg和abs是单节点函数
                                        l_child_index = l_child_index + 1
                                        l_child = program[l_child_index]
                                    while isinstance(r_child, _Function) and r_child.name in ['neg', 'abs']:
                                        right_child_index_stack.append(1)  # neg和abs是单节点函数
                                        r_child_index = r_child_index + 1
                                        r_child = program[r_child_index]
                                    if isinstance(l_child, _Function) and isinstance(r_child, _Function):
                                        if l_child.name == r_child.name:
                                            left_child_index = l_child_index
                                            left_child = l_child
                                            left_child_index_stack.append(l_index + 1)
                                            left_child_index_stack.append(0)

                                            right_child_index = r_child_index
                                            right_child = r_child
                                            right_child_index_stack.append(r_index + 1)
                                            right_child_index_stack.append(0)
                                            match = True
                                            break
                                    elif isinstance(l_child, tuple) and l_child == r_child:  # 变量节点相同
                                        cancel = True
                                        break
                                elif l_child.name == r_child.name:
                                    left_child_index = l_child_index
                                    left_child = l_child
                                    left_child_index_stack.append(l_index + 1)
                                    left_child_index_stack.append(0)

                                    right_child_index = r_child_index
                                    right_child = r_child
                                    right_child_index_stack.append(r_index + 1)
                                    right_child_index_stack.append(0)
                                    match = True
                                    break
                        if match or cancel:
                            break
                    if cancel:  # 发现存在抵消，则停止搜索，并禁止此次突变
                        break
                    if not match:  # 没找到匹配的就回溯到父节点继续搜索
                        left_child_index += left_child.parent_distance
                        left_child = program[left_child_index]
                        right_child_index += right_child.parent_distance
                        right_child = program[right_child_index]
                        if program[root].name == 'div':  # 只有div才忽略neg和abs
                            while left_child.name in ['neg', 'abs']:
                                left_child_index += left_child.parent_distance
                                left_child = program[left_child_index]
                                if len(left_child_index_stack):
                                    left_child_index_stack.pop()  # 去掉neg和abs记录的下一探索点
                                else:
                                    break
                            while right_child.name in ['neg', 'abs']:
                                right_child_index += right_child.parent_distance
                                right_child = program[right_child_index]
                                if len(right_child_index_stack):
                                    right_child_index_stack.pop()  # 去掉neg和abs记录的下一探索点
                                else:
                                    break
            # if left_child.name == right_child.name:
            #     left_child_index_stack = [0]  # 记录左子树还未探索的子节点索引
            #     right_child_index_stack = [0]  # 记录右子树还未探索的子节点索引
            #     # 这里应该要深度优先遍历完整个子树，而不是只遍历一个子支
            #     while len(left_child_index_stack) and len(right_child_index_stack):
            #         match = False
            #         left_fence = left_child_index_stack.pop()  # 获取最新的
            #         right_fence = right_child_index_stack.pop()  # 获取最新的
            #         for l_index, l_child_distance in enumerate(left_child.child_distance_list):
            #             if l_index < left_fence:
            #                 continue
            #             l_child_index = left_child_index + l_child_distance
            #             l_child = program[l_child_index]
            #             for r_index, r_child_distance in enumerate(right_child.child_distance_list):
            #                 if r_index < right_fence:
            #                     continue
            #                 r_child_index = right_child_index + r_child_distance
            #                 r_child = program[r_child_index]
            #                 if isinstance(l_child, tuple) and l_child == r_child:  # 变量节点相同
            #                     cancel = True
            #                     break
            #                 elif isinstance(l_child, _Function) and isinstance(r_child, _Function):
            #                     if l_child.name == r_child.name:
            #                         left_child_index = l_child_index
            #                         left_child = l_child
            #                         left_child_index_stack.append(l_index + 1)
            #                         left_child_index_stack.append(0)
            #
            #                         right_child_index = r_child_index
            #                         right_child = r_child
            #                         right_child_index_stack.append(r_index + 1)
            #                         right_child_index_stack.append(0)
            #                         match = True
            #                         break
            #             if match or cancel:
            #                 break
            #         if cancel:  # 发现存在抵消，则停止搜索，并禁止此次突变
            #             break
            #         if not match:  # 没找到匹配的就回溯到父节点继续搜索
            #             left_child_index = left_child_index + left_child.parent_distance
            #             left_child = program[left_child_index]
            #             right_child_index = right_child_index + right_child.parent_distance
            #             right_child = program[right_child_index]
        return cancel

    @staticmethod
    def subtree_state_larger(remaining, total):  # 比较remaining和total，当第二个被第一个dominate时返回true
        temp = [remaining[i] - total[i] for i in range(len(remaining))]
        return temp[0] >= 0 and temp[1] >= 0 and temp[2] >= 0 and temp[3] >= 0  # 次数都有剩余，则可以兼容

    # @staticmethod
    # def check_prohibit(parent_name, node):  # 给定父节点名字和当前节点，检查是否会导致违规的连续嵌套
    #     if not isinstance(node, _Function):  # 当前节点不是函数节点则没有限制，返回True
    #         return True
    #     elif parent_name == 'exp' and node.name == 'log':
    #         return False
    #     elif parent_name == 'log' and node.name == 'exp':
    #         return False
    #     elif parent_name == node.name and parent_name in ['abs', 'neg', 'add', 'sum', 'sqrt']:
    #         return False
    #     elif parent_name in elementary_functions and isinstance(node, list):  # 父节点是基本初等函数，则子节点不可以只是常数
    #         return False
    #     return True

    @staticmethod
    def execute_test(program, X):  # X是一个由多个输入向量组成的矩阵
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = program[0]  # 单节点没有什么意义
        if isinstance(node, list):  # 常数向量检测，检测np.ndarray类型
            print('constant')
            return np.repeat(node[0][0], X.shape[0])  # 对每个输入向量返回一个实数
        if isinstance(node, tuple):  # 变量向量检测
            print('variable')
            return X[:, node[0]]
        apply_stack = []
        # 输出每个个体
        # print(self.__str__())
        # printout(program)
        # print_formula(program, show_operand=True)
        for index, node in enumerate(program):
            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)
            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:  # 操作数凑齐时开始计算
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = []
                for t in apply_stack[-1][1:]:
                    if isinstance(t, list):  # 常数向量改为list[ndarray]类型，避免了后续的混淆
                        temp = np.repeat(t, X.shape[0], axis=0)  # n_samples x dimension of t
                        # 调整维度顺序，n_samples调整为最后一维，因为sum和prod等aggregate函数是按axis=0来进行计算的
                        # print(f'{temp}{temp.shape}')
                        temp = np.transpose(temp, axes=(1, 0))  # dimension x n_samples
                        # print(f'{temp}{temp.shape}')
                        terminals.append(temp)
                    elif isinstance(t, tuple):
                        temp = X[:, t[0]:t[1]:t[2]]  # n_samples x dimension of t
                        # 调整维度顺序，n_samples调整为最后一维，因为sum和prod等aggregate函数是按axis=0来进行计算的
                        # print(f'{temp}{temp.shape}')
                        temp = np.transpose(temp, axes=(1, 0))  # dimension x n_samples
                        # print(f'{temp}{temp.shape}')
                        terminals.append(temp)
                    else:  # 中间结果，即np.ndarray类型，无需额外处理
                        terminals.append(t)  # arity x dimension x n_samples
                # for t in apply_stack[-1][1:]:
                #     if isinstance(t, list):  # 常数向量改为list[ndarray]类型，避免了后续的混淆
                #         temp = np.repeat(t, X.shape[0], axis=0).squeeze()  # n_samples x dimension of t
                #         print(temp.shape)
                #         if len(temp.shape) == 2:  # 有两个维度
                #             # 调整维度顺序，n_samples调整为最后一维，因为sum和prod等aggregate函数是按axis=0来进行计算的
                #             temp = np.transpose(temp, axes=(1, 0))
                #         terminals.append(temp)
                #     elif isinstance(t, tuple):
                #         temp = X[:, t[0]:t[1]:t[2]].squeeze()  # n_samples x dimension of t
                #         print(temp.shape)
                #         if len(temp.shape) == 2:  # 有两个维度，没有两个维度则说明
                #             # 调整维度顺序，n_samples调整为最后一维，因为sum和prod等aggregate函数是按axis=0来进行计算的
                #             temp = np.transpose(temp, axes=(1, 0))
                #         terminals.append(temp)
                #     else:  # 中间结果，即np.ndarray类型，无需额外处理
                #         terminals.append(t)
                # 聚集函数要保证不在样本数维度上做聚集计算，arity>1时在各个操作数维度上进行聚集计算，arity=1时在特征数维度上进行聚集计算
                # print(f'{function.name}[{function.arity}]')
                if function.name in ['sum', 'prod', 'mean']:
                    terminals = np.array(terminals)
                    # print(f'{terminals}{terminals.shape}')
                    # arity>1时sum和prod保持输入和输出维度相同，arity=1时输入为向量，输出为实数
                    # if function.name != 'mean' or np.array(terminals).ndim > 1:
                    #     intermediate_result = function(np.array(terminals).squeeze())
                    # else:
                    #     intermediate_result = function(np.array(terminals))
                    if terminals.ndim > 2 and terminals.shape[0] == 1:  # arity为1时去掉操作数维度
                        intermediate_result = function(terminals[0])
                    else:  # arity>1时
                        intermediate_result = function(terminals)
                    # print(intermediate_result)
                else:
                    # print(f'{terminals}')
                    intermediate_result = function(*terminals)
                # print(intermediate_result)
                if len(intermediate_result) == 0:
                    print(index)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    if len(intermediate_result.shape):
                        return intermediate_result.squeeze()
                    else:
                        # print(type(np.array(intermediate_result, dtype=np.float64)))
                        return np.array(intermediate_result, dtype=np.float64)
        # We should never get here
        return None

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)

