# 类 rmobo 继承自 mobo
# 重写 solve 方法

import numpy as np
from mobo.factory import get_acquisition
from mobo.mobo import MOBO
from pymoo.factory import get_performance_indicator
from .surrogate_problem import SurrogateProblem
from .utils import Timer, find_pareto_front, calc_hypervolume

class RMOBO(MOBO):
    '''
    Robin MOBO 2
    '''
    config = {
        'surrogate': 'gp',
        'acquisition': 'ei',
        'solver': 'nsga2',
        'selection': 'uncertainty',
    }

    def __init__(self, problem, n_iter, ref_point, framework_args):
        super().__init__(problem, n_iter, ref_point, framework_args)
        framework_args['surrogate']['n_var'] = self.n_var # for surrogate fitting
        framework_args['surrogate']['n_obj'] = self.n_obj # for surroagte fitting
        framework_args['solver']['n_obj'] = self.n_obj # for MOEA/D-EGO
        kwargs = framework_args['acquisition']
        self.acquisition1 = get_acquisition('ei')(**kwargs) # acquisition function for phase 1
        self.acquisition2 = get_acquisition('pi')(**kwargs) # acquisition function for phase 2
        self.phase1_ratio = 1.0 # ratio of phase 1 iterations to total iterations

    def solve(self, X_init, Y_init):
        '''
        Solve the real multi-objective problem from initial data (X_init, Y_init)
        '''
        # determine reference point from data if not specified by arguments
        if self.ref_point is None:
            self.ref_point = np.max(Y_init, axis=0)
        self.selection.set_ref_point(self.ref_point)

        self._update_status(X_init, Y_init)

        # 计算phase1和phase2的迭代次数
        n_phase1_iter = int(self.n_iter * self.phase1_ratio)
        n_phase2_iter = self.n_iter - n_phase1_iter

        global_timer = Timer()

        for i in range(self.n_iter):
            print('========== Iteration %d ==========' % i)

            timer = Timer()

            if i < n_phase1_iter:
                # phase 1: maximize EI
                self.acquisition = self.acquisition1
            else:
                # phase 2: maximize PI
                self.acquisition = self.acquisition2

            # data normalization
            self.transformation.fit(self.X, self.Y)
            X, Y = self.transformation.do(self.X, self.Y)

            # build surrogate models
            self.surrogate_model.fit(X, Y)
            timer.log('Surrogate model fitted')

            # define acquisition functions
            self.acquisition.fit(X, Y)

            # solve surrogate problem
            surr_problem = SurrogateProblem(self.real_problem, self.surrogate_model, self.acquisition, self.transformation)
            solution = self.solver.solve(surr_problem, X, Y)
            timer.log('Surrogate problem solved')

            # batch point selection
            self.selection.fit(X, Y)
            X_next, self.info = self.selection.select(solution, self.surrogate_model, self.status, self.transformation)
            timer.log('Next sample batch selected')

            # update dataset
            Y_next = self.real_problem.evaluate(X_next)
            if self.real_problem.n_constr > 0: Y_next = Y_next[0]
            self._update_status(X_next, Y_next)
            timer.log('New samples evaluated')

            # statistics
            global_timer.log('Total runtime', reset=False)
            print('Total evaluations: %d, hypervolume: %.4f, igd: %.4f\n' % (self.sample_num, self.status['hv'], self.status['igd']))
            
            # return new data iteration by iteration
            yield X_next, Y_next
    
    def __str__(self):
        n_phase1_iter = int(self.n_iter * self.phase1_ratio)
        n_phase2_iter = self.n_iter - n_phase1_iter
        return \
            '========== Framework Description ==========\n' + \
            f'# algorithm: {self.__class__.__name__}\n' + \
            f'# surrogate: {self.surrogate_model.__class__.__name__}\n' + \
            f'# acquisition1: {self.acquisition1.__class__.__name__} {n_phase1_iter} iterations\n' + \
            f'# acquisition2: {self.acquisition2.__class__.__name__} {n_phase2_iter} iterations\n' + \
            f'# solver: {self.solver.__class__.__name__}\n' + \
            f'# selection: {self.selection.__class__.__name__}\n'

