import shutil
import sys
import os
import glob
import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import ioh
from modcma.modularcmaes import ModularCMAES
from dataclasses import dataclass, fields

@dataclass
class TrackedParameters:
    sigma: float = 0
    t: int = 0
    d_norm: float = 0
    d_mean: float = 0
    ps_norm: float = 0
    ps_mean: float = 0
    pc_norm: float = 0
    pc_mean: float = 0
    lambda_: int = 0
    ps_ratio: float = 0
    ps_squared: float = 0


    def update(self, parameters):
        self.sigma = parameters.sigma
        self.t = parameters.t
        self.lambda_ = parameters.lambda_

        for attr in ('D', 'ps', 'pc'):
            setattr(self, f'{attr}_norm'.lower(),
                    np.linalg.norm(getattr(parameters, attr)))
            setattr(self, f'{attr}_mean'.lower(),
                    np.mean(getattr(parameters, attr)))

        self.ps_squared = np.sum(parameters.ps**2)
        self.ps_ratio = np.sqrt(self.ps_squared) / parameters.chiN


class TrackedCMAES(ModularCMAES):
    def __init__(self, tracked_parameters=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracked_parameters = tracked_parameters
        if self.tracked_parameters is not None:
            self.tracked_parameters.update(self.parameters)

    def step(self):
        res = super().step()
        if self.tracked_parameters is not None:
            self.tracked_parameters.update(self.parameters)
        return res


dim = 10
budget_factor = 2500
reps = 10
algo = 'exp-inc'

for id in range(1,2):
    
    problem = ioh.get_problem(
        fid=id+1,
        instance=1,
        dimension=dim,
        problem_class=ioh.ProblemClass.BBOB
    )

    trigger = ioh.logger.trigger.OnImprovement()

    logger = ioh.logger.Analyzer(
        triggers=[trigger],
        folder_name=f'./data/{algo}-fid{id+1}-{dim}D',
        root=os.getcwd(),
        algorithm_name=f'{algo}',
        store_positions=False)
    
    tracked_parameters = TrackedParameters()
    logger.watch(tracked_parameters, [
        x.name for x in fields(tracked_parameters)])

    problem.attach_logger(logger)

    for rep in range(reps):

        np.random.seed(rep)
        cma = TrackedCMAES(
            tracked_parameters,
            problem,
            dim,
            lambda_=4,
            budget = dim*budget_factor,
            pop_size_adaptation=f'{algo}',
            min_lambda_=8,
            max_lambda_=50
            ).run()

        print(f'fid={id+1}/24, rep={rep+1}/{reps}, best y={problem.state.current_best.y}')

        problem.reset()
        logger.close()
