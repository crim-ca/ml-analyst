import numpy as np
import argparse
import os, errno, sys
from joblib import Parallel, delayed

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="An analyst for quick ML applications.",
                                     add_help=False)
    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to analyze; ensure that the '
                        'target/label column is labeled as "class".')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
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


    args = parser.parse_args()
     
    if args.RANDOM_STATE:
        random_state = args.RANDOM_STATE
    else:
        random_state = np.random.randint(2**15 - 1)

    learners = [ml for ml in args.LEARNERS.split(',')]  # learners

    if args.SEARCH == 'random':
        if args.PREP:
            model_dir = 'ml/random_search_preprocessing/'
        else:
            model_dir = 'ml/random_search/'
    else:
        model_dir= 'ml/grid_search/' 

    dataset = args.INPUT_FILE.split('/')[-1].split('.csv')[0]
    RANDOM_STATE = args.RANDOM_STATE
    results_path = '/'.join([args.RDIR, dataset]) + '/'
    # make the results_path directory if it doesn't exit 
    try:
        os.makedirs(results_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    # initialize output files
    for ml in learners:
        #write headers
        if args.PREP:
            save_file = results_path + '-'.join(args.PREP.split(',')) + '_' + ml + '.csv'  
        else:
            save_file = results_path + '/' + dataset + '_' + ml + '.csv'  

        feat_file =  save_file.split('.')[0]+'.imp_score'        
        roc_file =  save_file.split('.')[0]+'.roc'        
        
        with open(save_file.split('.')[0] + '.imp_score','w') as out:
            out.write('preprocessor\tprep-parameters\talgorithm\talg-parameters\tseed\tfeature\tscore\n')
         
        with open(save_file.split('.')[0] + '.roc','w') as out:
            out.write('preprocessor\tprep-parameters\talgorithm\talg-parameters\tseed\tfpr\ttpr\tauc\n')
   
        with open(save_file,'w') as out:
            if args.PREP:
                out.write('dataset\tpreprocessor\tprep-parameters\talgorithm\talg-parameters\tseed\taccuracy\tf1_macro\tbal_accuracy\troc_auc\n')
            else:
                out.write('dataset\talgorithm\tparameters\taccuracy\tf1_macro\tseed\tbal_accuracy\troc_auc\n')
        
    # write run commands
    all_commands = []
    job_info=[]
    for t in range(args.N_TRIALS):
        random_state = np.random.randint(2**15-1)
        for ml in learners:
            if args.PREP:
                save_file = results_path + '-'.join(args.PREP.split(',')) + '_' + ml + '.csv'  
            else:
                save_file = results_path + '/' + dataset + '_' + ml + '.csv'  
            
            if args.PREP: 
                all_commands.append('{PYTHON_EXEC} {PATH}/{ML}.py {DATASET} {SAVEFILE} {N_COMBOS} {RS} {PREP} {LABEL}'.\
                                    format(PYTHON_EXEC=sys.executable,
                                           PATH=model_dir,
                                           ML=ml,
                                           DATASET=args.INPUT_FILE,
                                           SAVEFILE=save_file,
                                           N_COMBOS=args.N_COMBOS,
                                           RS=random_state,
                                           PREP=args.PREP,
                                           LABEL=args.LABEL))
            elif args.SEARCH == 'random':
                all_commands.append('{PYTHON_EXEC} {PATH}/{ML}.py {DATASET} {SAVEFILE} {N_COMBOS} {RS}'.\
                                    format(PYTHON_EXEC=sys.executable,
                                           PATH=model_dir,
                                           ML=ml,
                                           DATASET=args.INPUT_FILE,
                                           SAVEFILE=save_file,
                                           N_COMBOS=args.N_COMBOS,
                                           RS=random_state))
            else:
                all_commands.append('{PYTHON_EXEC} {PATH}/{ML}.py {DATASET} {SAVEFILE} {RS}'.\
                                    format(PYTHON_EXEC=sys.executable,
                                           PATH=model_dir,
                                           ML=ml,
                                           DATASET=args.INPUT_FILE,
                                           SAVEFILE=save_file,
                                           RS=random_state))
            job_info.append({'ml':ml,'dataset':dataset,'results_path':results_path})

    if args.LSF:    # bsub commands
        for i,run_cmd in enumerate(all_commands):
            job_name = job_info[i]['ml'] + '_' + job_info[i]['dataset']
            out_file = job_info[i]['results_path'] + job_name + '_%J.out'

            bsub_cmd = ('bsub -o {OUT_FILE} -n {N_CORES} -J {JOB_NAME} -q {QUEUE} '
                       '-R "span[hosts=1] rusage[mem={M}]" -M {M} ').format(OUT_FILE=out_file,
                                             JOB_NAME=job_name,
                                             QUEUE=args.QUEUE,
                                             N_CORES=args.N_JOBS,
                                             M=args.M)
            
            bsub_cmd +=  '"' + run_cmd + '"'
            print(bsub_cmd)
            os.system(bsub_cmd)     # submit jobs 
    else:   # run locally
        print("Will locally run these:\n")
        for run_cmd in all_commands:
            print(run_cmd)
        Parallel(n_jobs=args.N_JOBS)(delayed(os.system)(run_cmd) for run_cmd in all_commands )
