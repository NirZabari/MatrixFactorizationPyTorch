from ray import tune
import wandb
from enum import Enum, auto


def get_search_space():
    search_space = {
        "learning_rate": tune.loguniform(0.0001, 0.1),
        "latent_dimension": tune.choice(list(range(2, 100))),
        "batch_size": tune.randint(1, 512),
    }
    return search_space


def get_metrics(analysis_results):
    columns = analysis_results.results_df.columns
    return [c for c in columns if c.startswith('_metric')]


def dump_analysis(analysis_results, wb_run):
    metrics = get_metrics(analysis_results)
    analysis_results.default_metric = '_metric/accuracy'
    analysis_results.default_mode = 'max'

    best_result = analysis_results.best_result  # Get best trial's last results

    # Get a dataframe with the last results for each trial
    df_results = analysis_results.results_df
    wb_run.log({"df_results": wandb.Table(data=df_results)})

    best_run_info = {**best_result['_metric'], **best_result['config']}
    return best_run_info


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class RayTuneModes(AutoName):
    ASHS = auto()
    MEDIAN_STOPPING_RULE = auto()
    BAYESIAN_OPTIMIZATION = auto()
    AX = auto()
    OPTUNA = auto()


def get_raytune_args(search_space, mode, num_samples=2):
    from ray.tune.schedulers import AsyncHyperBandScheduler, MedianStoppingRule
    from ray.tune.suggest.bayesopt import BayesOptSearch
    from ray.tune.suggest.ax import AxSearch
    from ray.tune.suggest.optuna import OptunaSearch

    args = {
        'config': search_space,
        'num_samples': num_samples,
        'max_failures': 4,
        'resources_per_trial': {'cpu': 1,
                                # the number of cpu determins the number of concurrent trials
                                'gpu': 1,
                                },
    }

    if mode == RayTuneModes.ASHS:
        scheduler = AsyncHyperBandScheduler(grace_period=5, max_t=100)
        args['scheduler'] = scheduler
    elif mode == RayTuneModes.MEDIAN_STOPPING_RULE:
        scheduler = MedianStoppingRule(grace_period=5)
        args['scheduler'] = scheduler
    elif mode == RayTuneModes.BAYESIAN_OPTIMIZATION:
        bayesopt = BayesOptSearch(metric='_metric/accuracy', mode='max')
        args['search_alg'] = bayesopt
    elif mode == RayTuneModes.AX:
        ax_search = AxSearch(metric="score")
        args['search_alg'] = ax_search
    elif mode == RayTuneModes.OPTUNA:
        optuna_search = OptunaSearch(metric='_metric/accuracy', mode='max')
        args['search_alg'] = optuna_search
    return args


def run_hpo(train_func, mode, num_samples):
    wb_run = wandb.init(project='mf_hpo', save_code=True)
    wb_run.log({'mode': str(mode)})
    search_space = get_search_space()
    ray_args = get_raytune_args(search_space=search_space, mode=mode, num_samples=num_samples)
    analysis_results = tune.run(
        train_func,
        metric='_metric/accuracy',
        mode='max',
        **ray_args
    )

    best_run_info = dump_analysis(analysis_results=analysis_results,
                                  wb_run=wb_run)
    best_run_info['mode'] = str(mode)
    wb_run.log({'best_run_info': best_run_info})
    wb_run.finish()
    print("finished HPO run")
    print("finished running")
