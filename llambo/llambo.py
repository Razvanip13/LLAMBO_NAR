import time
import json 


# Import extended versions
from llambo.discriminative_sm import LLM_DIS_SM
from llambo.generative_sm import LLM_GEN_SM
from llambo.aquisition_function import LLM_ACQ # Corrected typo aquisition->acquisition
from llambo.rate_limiter import RateLimiter
from llambo.warping import NumericalTransformer
from llambo.gemma_adapter import GemmaLocalAdapter # Import the adapter
import pandas as pd
import pprint


class LLAMBO:
    def __init__(self,
                 task_context: dict,  # dictionary describing task (see above)
                 sm_mode,  # either 'generative' or 'discriminative'
                 n_candidates,  # number of candidate points to sample at each iteration
                 n_templates,  # number of templates for LLM queries
                 n_gens,    # number of generations for LLM, set at 5
                 alpha,    # alpha for LLM, recommended to be -0.2
                 n_initial_samples, # number of initial samples to evaluate
                 n_trials,   # number of trials to run,
                 init_f,        # function to generate initial configurations
                 bbox_eval_f,       # bbox function to evaluate a point
                 # chat_engine removed, adapter handles model choice
                 llm_adapter: GemmaLocalAdapter, # <-- ADD ADAPTER ARGUMENT
                 top_pct=None,      # only used for generative SM, top percentage of points to consider for generative SM
                 use_input_warping=False,       # whether to use input warping
                 prompt_setting=None,    # ablation on prompt design for SM and ACQ
                 sm_acquisition_strategy='EI', # 'EI' or 'UCB' for surrogate model's selection
                 sm_ucb_kappa=1.96,          # Kappa value if UCB is used for SM
                 shuffle_features=False     # whether to shuffle features in prompt generation
                 ):
        self.task_context = task_context
        assert sm_mode in ['generative', 'discriminative']
        assert top_pct is None if sm_mode == 'discriminative' else top_pct is not None
        self.model_name = task_context['model']
        self.lower_is_better = task_context['lower_is_better']
        lower_is_better = self.lower_is_better
        self.n_candidates = n_candidates
        self.n_template = n_templates
        self.n_gens = n_gens
        self.alpha = alpha
        self.n_initial_samples = n_initial_samples
        self.n_trials = n_trials
        self.llm_query_cost = []    # list of cost for LLM calls in EACH TRIAL
        self.llm_query_time = []    # list of time taken for LLM calls in EACH TRIAL

        self.llm_adapter = llm_adapter # <-- STORE ADAPTER
        assert self.llm_adapter is not None, "An LLM Adapter instance must be provided."

        assert type(shuffle_features) == bool, 'shuffle_features should be a boolean'
        assert type(use_input_warping) == bool, 'use_input_warping should be a boolean'

        self.init_f = init_f
        self.bbox_eval_f = bbox_eval_f

        if use_input_warping:
            warping_transformer = NumericalTransformer(task_context['hyperparameter_constraints'])
        else:
            warping_transformer = None

        rate_limiter = RateLimiter(max_tokens=100000, time_frame=60, max_requests=720)
        
        print('='*150)
        print(f'[Search settings]: ' + '\n\t'
              f'n_candidates: {n_candidates}, n_templates: {n_templates}, n_gens: {n_gens}, ' + '\n\t'
              f'alpha: {alpha}, n_initial_samples: {n_initial_samples}, n_trials: {n_trials}, ' + '\n\t'
              f'using warping: {use_input_warping}, prompt_setting: {prompt_setting}, ' + '\n\t'
              f'sm_acquisition_strategy: {sm_acquisition_strategy}, sm_ucb_kappa: {sm_ucb_kappa}, ' + '\n\t'
              f'shuffle_features: {shuffle_features}')
        print(f'[Task]: ' + '\n\t'
              f'task type: {task_context["task"]}, sm: {sm_mode}, lower is better: {lower_is_better}')
        print(f'Hyperparameter search space: ')
        pprint.pprint(task_context['hyperparameter_constraints'])
        print('='*150)

        # initialize surrogate model and acquisition function
        if sm_mode == 'generative':
            # Pass the adapter to the extended GEN_SM
            self.surrogate_model = LLM_GEN_SM(task_context, n_gens, lower_is_better, top_pct,
                                              n_templates=n_templates, rate_limiter=None, # Rate limiter might be unused now
                                              llm_adapter=self.llm_adapter)
        else:
            # Pass the adapter to the extended DIS_SM
            self.surrogate_model = LLM_DIS_SM(task_context, n_gens, lower_is_better,
                                              n_templates=n_templates, rate_limiter=rate_limiter,
                                              warping_transformer=warping_transformer,
                                              # chat_engine removed
                                              prompt_setting=prompt_setting,
                                              shuffle_features=shuffle_features,
                                              acquisition_strategy=sm_acquisition_strategy,
                                              ucb_kappa=sm_ucb_kappa,
                                              llm_adapter=self.llm_adapter,
                                              verbose=True)

        # Pass the adapter to the extended ACQ
        self.acq_func = LLM_ACQ(task_context, n_candidates, n_templates, lower_is_better,
                                rate_limiter=rate_limiter, warping_transformer=warping_transformer,
                                # chat_engine removed
                                prompt_setting=prompt_setting,
                                shuffle_features=shuffle_features,
                                llm_adapter=self.llm_adapter) # I included the adapter. Every component should rely on the adapter


        self.sm_acquisition_strategy = sm_acquisition_strategy

    def _initialize(self):
        '''Initialize the optimization loop.'''
        start_time = time.time()
        # generate initial configurations
        init_configs = self.init_f(self.n_initial_samples)

        assert isinstance(init_configs, list), 'init_f() should return a list of configs (dictionaries)'
        for item in init_configs:
            assert isinstance(item, dict), 'init_f() should return a list of configs (dictionaries)'

        init_configs = pd.DataFrame(init_configs)
        assert init_configs.shape[0] == self.n_initial_samples, 'init_f() should return n_initial_samples number of configs'

        # create empty pandas dataframe for observed function values
        self.observed_fvals = pd.DataFrame()
        self.observed_configs = pd.DataFrame()

        for index, _ in init_configs.iterrows():
            one_config = init_configs.iloc[[index]]
            one_config, one_result = self._evaluate_config(one_config)

            if self.observed_fvals.empty:
                self.observed_fvals = one_result
            else:
                self.observed_fvals = pd.concat([self.observed_fvals, one_result], axis=0, ignore_index=True)

            if self.observed_configs.empty:
                self.observed_configs = one_config
            else:
                self.observed_configs = pd.concat([self.observed_configs, one_config], axis=0, ignore_index=True)

        print(f'[Initialization] COMPLETED: {self.observed_fvals.shape[0]} points evaluated...')
        end_time = time.time()

        time_taken = end_time - start_time
        return 0, time_taken

    def load_observed_configs(self, path):
        df = pd.read_csv(path)

        acc_column_index = df.columns.get_loc('acc')
        self.observed_configs = df.iloc[:, :acc_column_index]
        self.observed_fvals = df.iloc[:, acc_column_index:]


    def load_llm_query_info(self, path): 
        
        with open(path, 'r') as f:
            data = json.load(f)

        self.llm_query_cost = data['llm_query_cost_breakdown']
        self.llm_query_time = data['llm_query_time_breakdown']


    def load_attention_info(self, path): 
        
        with open(path, 'r') as f:
            data = json.load(f)

        self.acq_func.acq_attention_results = data['acq_attention_results']
        self.surrogate_model.sm_attention_results = data['sm_attention_results']


    def _evaluate_config(self, config):
        # can support batch mode in the future
        assert config.shape[0] == 1, 'batch mode not supported yet'
        config = config.to_dict('records')[0]


        eval_config, eval_results = self.bbox_eval_f(config)

        assert isinstance(eval_config, dict), 'bbox_eval_f() should return the evaluated config as a dictinoary'
        assert isinstance(eval_results, dict), 'bbox_eval_f() should return bbox evaluation results as a dictionary'
        assert 'score' in eval_results.keys(), 'score must be a key in results returned'

        return pd.DataFrame([eval_config]), pd.DataFrame([eval_results])

    def _update_observations(self, new_config, new_fval):
        '''Update the observed configurations and function values.'''
        # append new observations
        self.observed_configs = pd.concat([self.observed_configs, new_config], axis=0, ignore_index=True)
        self.observed_fvals = pd.concat([self.observed_fvals, new_fval], axis=0, ignore_index=True)

    def optimize(self, test_metric='generalization_score',save_res_dir=None, seed=None, resume=False):
        '''Run the optimization loop.'''

        print(f"Running the following SM strategy: {self.sm_acquisition_strategy}")
        time.sleep(10)


        # initialize
        cost, query_time = self._initialize()
        self.llm_query_cost.append(cost)
        self.llm_query_time.append(query_time)

        if self.lower_is_better:
            self.best_fval = self.observed_fvals['score'].min()
            best_gen_fval = self.observed_fvals[test_metric].min()
        else:
            self.best_fval = self.observed_fvals['score'].max()
            best_gen_fval = self.observed_fvals[test_metric].max()

        print(f'[Initialization] COMPLETED: best fval: {self.best_fval:.4f}, best generalization fval: {best_gen_fval:.4f}')
        print('='*150)


        self.acq_func.acq_attention_filepath = f'{save_res_dir}/{seed}acq_attention_vectors.jsonl'

        if resume == True: 
            print(f"Loading the previous checkpoint")
            self.load_observed_configs(f'{save_res_dir}/{seed}.csv')
            self.load_llm_query_info(f'{save_res_dir}/{seed}_search_info.json')
            self.load_attention_info(f'{save_res_dir}/{seed}_search_info.json')
            self.n_trials = self.n_trials - len(self.observed_configs) # we take into account the number of trials left

            print(f"Number of trials left: {self.n_trials}")
            time.sleep(5)
        else: 
            # reset the file for writing attention 
            with open(self.acq_func.acq_attention_filepath, 'w') as f_jsonl:
                pass 

        # optimization loop
        for trial_id in range(1, self.n_trials+1):
            trial_cost = 0
            trial_query_time = 0

            
            if trial_id%5 == 0: 
                self.acq_func.got_attention_vector = False # we collect the attention vector for that step
            
            start_time = time.time()
            # get candidate point


            count = 0 
            while count<5:
                candidate_points, cost, time_taken = self.acq_func.get_candidate_points(self.observed_configs, self.observed_fvals[['score']], alpha=self.alpha)
                trial_cost += cost
                trial_query_time += time_taken

                if len(candidate_points) > 0: 
                    break
                else: 
                    print('='*150)
                    print("No candidates found yet")
                    print(candidate_points)
                    print(candidate_points)
                    print('='*150)
                    time.sleep(10)
                count+=1 

            print('='*150)
            print('EXAMPLE POINTS PROPOSED')
            print(candidate_points)
            print('='*150)


            # select candidate point
            sel_candidate_point, cost, time_taken = self.surrogate_model.select_query_point(self.observed_configs, 
                                                                           self.observed_fvals[['score']], 
                                                                           candidate_points)
            trial_cost += cost
            trial_query_time += time_taken

            self.llm_query_cost.append(trial_cost)
            self.llm_query_time.append(trial_query_time)

            print('='*150)
            print('SELECTED CANDIDATE POINT')
            print(sel_candidate_point)
            print('='*150)

            # evaluate candidate point
            sel_candidate_point, sel_candidate_fval = self._evaluate_config(sel_candidate_point)

            # update observations
            self._update_observations(sel_candidate_point, sel_candidate_fval)
            
            print('='*150)
            print('UPDATED OBSERVATIONS')
            print(self.observed_configs)
            print(self.observed_fvals)
            print('='*150)

            print("Saving the current observations")
            search_history = pd.concat([self.observed_configs, self.observed_fvals], axis=1)
            search_history.to_csv(f'{save_res_dir}/{seed}.csv', index=False)

            # save search info
            search_info = {
                'llm_query_cost_breakdown': self.llm_query_cost,
                'llm_query_time_breakdown': self.llm_query_time,
                'llm_query_cost': sum(self.llm_query_cost),
                'llm_query_time': sum(self.llm_query_time),
                'acq_attention_results': self.acq_func.acq_attention_results, # Adapter now only stores ACQ results
                'sm_attention_results': self.surrogate_model.sm_attention_results # Get SM results directly
            }
            with open(f'{save_res_dir}/{seed}_search_info.json', 'w') as f:
                json.dump(search_info, f)


            end_time = time.time()
            time_taken = end_time - start_time

            current_fval_cv = sel_candidate_fval['score'].values[0]
            current_fval_gen = sel_candidate_fval[test_metric].values[0]

            if self.lower_is_better:
                if current_fval_cv < self.best_fval:
                    self.best_fval = current_fval_cv
                    best_found = True
                else:
                    best_found = False
            else:
                if current_fval_cv > self.best_fval:
                    self.best_fval = current_fval_cv
                    best_found = True
                else:
                    best_found = False

            if best_found:
                print(f'[Trial {trial_id} completed, time taken: {time_taken:.2f}s] best fval (cv): {self.best_fval:.4f}, current fval (cv): {current_fval_cv:.4f}. Generalization fval: {current_fval_gen:.4f} NEW BEST FVAL FOUND!!')
            else: 
                print(f'[Trial {trial_id} completed, time taken: {time_taken:.2f}s] best fval (cv): {self.best_fval:.4f}, current fval (cv): {current_fval_cv:.4f}. Generalization fval: {current_fval_gen:.4f}.')
            print('='*150)

        # returns history of observed configurations and function values
        return self.observed_configs, self.observed_fvals