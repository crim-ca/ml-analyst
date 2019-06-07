import numpy as np
import argparse
import os, errno, sys
from joblib import Parallel, delayed
from typing import List
import utils

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def run_analysis(**params) -> List[int]:
    """
<class 'dict'>: {'random_state': 25779, 'learners': ['LogisticRegression', 'SVC'], 'preps': ['RobustScaler'], 'model_dir': 'ml/random_search_preprocessing/', 'dataset': 'iris', 'results_path': 'results/iris/'}
    :param params: a variable structure with parameters:
        random_state: int
        learners: a list of functions, found on the 'ml' directory
        preps: a list of preprocessers
        model_dir: directory where the model exists
        dataset: name of the dataset
        results_path: where the analysis will be written
    :return: a list of error codes. 0 is success.
    """
    if 'results_path' not in params:
        raise RuntimeError("'results_path' needed to run the analysis")
    # make the results_path directory if it doesn't exit
    os.makedirs(params['results_path'], exist_ok=True)
    # initialize output files
    preps = ",".join(params['preps'])
    for ml in params['learners']:
        # write headers
        if preps:
            save_file = params['results_path'] + '-'.join(preps.split(',')) + '_' + ml + '.csv'
        else:
            save_file = params['results_path'] + '/' + params['dataset'] + '_' + ml + '.csv'

        feat_file = save_file.split('.')[0] + '.imp_score'
        roc_file = save_file.split('.')[0] + '.roc'

        with open(save_file.split('.')[0] + '.imp_score', 'w') as out:
            out.write('preprocessor\tprep-parameters\talgorithm\talg-parameters\tseed\tfeature\tscore\n')

        with open(save_file.split('.')[0] + '.roc', 'w') as out:
            out.write('preprocessor\tprep-parameters\talgorithm\talg-parameters\tseed\tfpr\ttpr\tauc\n')

        with open(save_file, 'w') as out:
            if preps:
                out.write(
                    'dataset\tpreprocessor\tprep-parameters\talgorithm\talg-parameters\tseed\taccuracy\tf1_macro\tbal_accuracy\troc_auc\n')
            else:
                out.write('dataset\talgorithm\tparameters\taccuracy\tf1_macro\tseed\tbal_accuracy\troc_auc\n')

    # write run commands
    all_commands = []
    job_info = []
    for t in range(params['n_trials']):
        random_state = np.random.randint(2 ** 15 - 1)
        for ml in params['learners']:
            if preps:
                save_file = params['results_path'] + '-'.join(preps.split(',')) + '_' + ml + '.csv'
            else:
                save_file = params['results_path'] + '/' + params['dataset'] + '_' + ml + '.csv'

            if preps:
                all_commands.append('{PYTHON_EXEC} {PATH}/{ML}.py {DATASET} {SAVEFILE} {N_COMBOS} {RS} {PREP} {LABEL}'. \
                                    format(PYTHON_EXEC=sys.executable,
                                           PATH=params['model_dir'],
                                           ML=ml,
                                           DATASET=params['input_file'],
                                           SAVEFILE=save_file,
                                           N_COMBOS=params['n_combos'],
                                           RS=random_state,
                                           PREP=preps,
                                           LABEL=params['label']))
            elif params['search'] == 'random':
                all_commands.append('{PYTHON_EXEC} {PATH}/{ML}.py {DATASET} {SAVEFILE} {N_COMBOS} {RS}'. \
                                    format(PYTHON_EXEC=sys.executable,
                                           PATH=params['model_dir'],
                                           ML=ml,
                                           DATASET=params['input_file'],
                                           SAVEFILE=save_file,
                                           N_COMBOS=params['n_combos'],
                                           RS=random_state))
            else:
                all_commands.append('{PYTHON_EXEC} {PATH}/{ML}.py {DATASET} {SAVEFILE} {RS}'. \
                                    format(PYTHON_EXEC=sys.executable,
                                           PATH=params['model_dir'],
                                           ML=ml,
                                           DATASET=params['input_file'],
                                           SAVEFILE=save_file,
                                           RS=random_state))
            job_info.append({'ml': ml, 'dataset': params['dataset'], 'results_path': params['results_path']})

    if params['lsf']:  # bsub commands
        for i, run_cmd in enumerate(all_commands):
            job_name = job_info[i]['ml'] + '_' + job_info[i]['dataset']
            out_file = job_info[i]['results_path'] + job_name + '_%J.out'

            bsub_cmd = ('bsub -o {OUT_FILE} -n {N_CORES} -J {JOB_NAME} -q {QUEUE} '
                        '-R "span[hosts=1] rusage[mem={M}]" -M {M} ').format(OUT_FILE=out_file,
                                                                             JOB_NAME=job_name,
                                                                             QUEUE=params['queue'],
                                                                             N_CORES=params['n_jobs'],
                                                                             M=params['m'])

            bsub_cmd += '"' + run_cmd + '"'
            print(bsub_cmd)
            os.system(bsub_cmd)  # submit jobs
    else:  # run locally
        print("Will locally run these:\n")
        for run_cmd in all_commands:
            print(run_cmd)
#         all_commands[-1] = "this fails"
        return Parallel(n_jobs=params['n_jobs'])(delayed(os.system)(run_cmd) for run_cmd in all_commands)


def args2params(args):
    """Parses command-line arguments into 'params' structure."""

    root_dir = os.path.dirname(__file__)
    params = {}
    if args.RANDOM_STATE:
        params['random_state'] = args.RANDOM_STATE
    else:
        params['random_state'] = np.random.randint(2**15 - 1)

    params['learners'] = utils.remove_duplicates([ml for ml in args.LEARNERS.split(',')])  # learners
    params['preps'] = utils.remove_duplicates([prep for prep in args.PREP.split(',')])  # pre-processers

    params['search'] = args.SEARCH
    if params['search'] == 'random':
        if args.PREP:
            params['model_dir'] = 'ml/random_search_preprocessing/'
        else:
            params['model_dir'] = 'ml/random_search/'
    else:
        params['model_dir'] = 'ml/grid_search/'
    params['model_dir'] = os.path.join(root_dir, params['model_dir'])
    if not os.path.isdir(params['model_dir']):
        raise RuntimeError("Models' directory '%s' does not exist" % (params['model_dir']))

    params['input_file'] = args.INPUT_FILE if os.path.isabs(args.INPUT_FILE) else os.path.join(root_dir, args.INPUT_FILE)
    params['dataset'] = params['input_file'].split('/')[-1].split('.csv')[0]
    params['results_path'] = '/'.join([args.RDIR, params['dataset']]) + '/'
    params['n_trials'] = args.N_TRIALS
    params['n_combos'] = args.N_COMBOS
    params['label'] = args.LABEL
    params['lsf'] = args.LSF
    params['queue'] = args.QUEUE
    params['n_jobs'] = args.N_JOBS
    params['m'] = args.M
    return params


def args_with_default_params():
    """Returns an 'args' structure, filled up with all default values."""
    return get_parser().parse_known_args()[0]


def get_parser():
    """Parser for user input."""
    parser = argparse.ArgumentParser(description="An analyst for quick ML applications.",
                                     add_help=False)
    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to analyze; ensure that the '
                        'target/label column is labeled as "class".')
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    # parameters with default values:
    parser.add_argument('-ml', action='store', dest='LEARNERS',default=None,type=str,
            help='Comma-separated list of ML methods to use (should correspond to a py file name '
                 'in ml/)')
    parser.add_argument('-prep', action='store', dest='PREP', default=None, type=str,
            help = 'Comma-separated list of preprocessors to apply to data')
    parser.add_argument('--lsf', action='store_true', dest='LSF', default=False,
            help='Run on an LSF HPC (using bsub commands)')
    parser.add_argument('-metric',action='store', dest='METRIC', default='f1_macro', type=str,
            help='Metric to compare algorithms')
    parser.add_argument('-k',action='store', dest='K', default=5, type=int,
            help='Number of folds for cross validation')
    parser.add_argument('-search',action='store',dest='SEARCH',default='random',choices=['grid','random'],
            help='Hyperparameter search strategy')
    parser.add_argument('--r',action='store_true',dest='REGRESSION',default=False,
            help='Run regression instead of classification.')
    parser.add_argument('-n_jobs',action='store',dest='N_JOBS',default=4,type=int,
            help='Number of parallel jobs')
    parser.add_argument('-n_trials',action='store',dest='N_TRIALS',default=1,type=int,
            help='Number of parallel jobs')
    parser.add_argument('-n_combos',action='store',dest='N_COMBOS',default=4,type=int,
            help='Number of hyperparameters to try')
    parser.add_argument('-rs',action='store',dest='RANDOM_STATE',default=None,type=int,
            help='random state')
    parser.add_argument('-label',action='store',dest='LABEL',default='class',type=str,
            help='Name of class label column')
    parser.add_argument('-m',action='store',dest='M',default=4096,type=int,
                        help='LSF memory request and limit (MB)')
    parser.add_argument('-results',action='store',dest='RDIR',default='results',type=str,
                        help='results directory')
    parser.add_argument('-q',action='store',dest='QUEUE',default='moore_normal',type=str,
                        help='results directory')
    return parser


if __name__ == '__main__':
    run_analysis(**args2params(get_parser().parse_args()))
