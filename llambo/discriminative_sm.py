# llambo/discriminative_sm.py
import os
import time
# import openai # No longer needed
# import asyncio # Removed async/await
import re
import numpy as np
from scipy.stats import norm
# from aiohttp import ClientSession # No longer needed
from llambo.rate_limiter import RateLimiter
# Ensure this import is correct and doesn't use openai
from llambo.discriminative_sm_utils import gen_prompt_tempates
# from .gemma_adapter import GemmaLocalAdapter # Import adapter

# Remove OpenAI config

class LLM_DIS_SM:
    # Add llm_adapter argument
    def __init__(self, task_context, n_gens, lower_is_better,
                 bootstrapping=False, n_templates=1,
                 use_recalibration=False,
                 rate_limiter=None, warping_transformer=None,
                 verbose=False, chat_engine=None,
                 prompt_setting=None, shuffle_features=False, # Acquisition strategy for SM
                 acquisition_strategy='EI', ucb_kappa=1.96,  # Default to EI, common kappa for UCB (95% CI)
                 llm_adapter=None): # <-- ADDED
        '''Initialize the forward LLM surrogate model (Adapter version).'''
        
        self.task_context = task_context
        self.n_gens = n_gens
        self.lower_is_better = lower_is_better
        self.bootstrapping = bootstrapping
        self.n_templates = n_templates
        assert not (bootstrapping and use_recalibration), 'Cannot do recalibration and boostrapping at the same time' 
        self.use_recalibration = use_recalibration
        if rate_limiter is None:
            self.rate_limiter = RateLimiter(max_tokens=100000, time_frame=60)
        else:
            self.rate_limiter = rate_limiter

        if warping_transformer is not None:
            self.warping_transformer = warping_transformer
            self.apply_warping = True
        else:
            self.warping_transformer = None
            self.apply_warping = False

        self.chat_engine = chat_engine
        self.verbose = verbose


        # ... (Keep initial assignments for task_context, n_gens, etc.) ...
        self.llm_adapter = llm_adapter # <-- STORED
        if self.llm_adapter is None:
             raise ValueError("LLM Adapter instance must be provided to LLM_DIS_SM.")
        self.chat_engine = chat_engine # Keep for compatibility maybe
        self.prompt_setting = prompt_setting
        self.shuffle_features = shuffle_features
        # ... (Keep warping_transformer, rate_limiter setup) ...
        # --- Acquisition Strategy for Surrogate Model ---
        self.acquisition_strategy = acquisition_strategy
        self.ucb_kappa = ucb_kappa
        # ---------------------------------------------
        self.apply_warping = warping_transformer is not None
        self.use_recalibration = use_recalibration
        self.recalibrator = None # Recalibration logic might need sync adaptation if used

        assert type(self.shuffle_features) == bool, 'shuffle_features must be a boolean'

        # Add public attribute to store SM-specific attention results
        self.sm_attention_results = []

        self.sm_avg_entropy_per_trial_log = [] # Initialize list to store per-trial SM entropy


    # Replace _async_generate with sync version using adapter
    def _sync_generate_with_adapter(self, few_shot_template, query_example, query_idx, n_history): # Add n_history
        """Synchronous generation using the local adapter for SM."""
        if not self.llm_adapter:
            raise ValueError("LLM Adapter not provided to LLM_DIS_SM")

        # Format the prompt
        if hasattr(few_shot_template, 'format'): # Check if it's a LangChain template
             user_message = few_shot_template.format(Q=query_example['Q'])
        else: # Assume it's already a string
             user_message = few_shot_template

        MAX_RETRIES = 3
        resp_data = None
        entropy = np.nan # We might not log SM entropy, but adapter returns it
        seq_len = 0
        # Determine generations per call based on bootstrapping
        n_preds = max(1, int(self.n_gens / self.n_templates)) if self.bootstrapping and self.n_templates > 0 else max(1, int(self.n_gens))

        for retry in range(MAX_RETRIES):
            try:
                start_time = time.time()
                # Rate limiting might need adjustment/removal
                # self.rate_limiter.add_request(request_text=user_message, current_time=start_time)

                # --- CALL ADAPTER ---
                # Decide whether to log attention data from SM calls
                LOG_SM_ATTENTION = True # <<< CONFIGURATION FLAG - Should SM store its entropy?

                # print(user_message)
                # print(f"Surrogate function prompt: {user_message}")

                mock_openai_response, current_entropy, current_seq_len, avg_head_attention_tensor = self.llm_adapter.generate(
                    prompt_text=user_message,
                    n_gens=max(n_preds, 1), # Request at least 1, ensure adapter handles num_return_sequences
                    temperature=1,
                    # max_new_tokens=8, # Low for score prediction
                    top_p=0.95,
                    log_attention_data=True, 
                    # n_history_for_log not relevant here
                    source_tag='SM' # Tag the source
                )
                # ------------------

                # --- Store SM entropy internally if requested ---
                if LOG_SM_ATTENTION and not np.isnan(current_entropy):
                    self.sm_attention_results.append({
                        'entropy': current_entropy,
                        'seq_len': current_seq_len,
                        'n_history': n_history # <-- ADDED n_history
                    })

                resp_data = mock_openai_response
                entropy = current_entropy
                seq_len = current_seq_len

                # Dummy cost/token calculation
                tot_tokens = resp_data['usage']['total_tokens']
                tot_cost = 0.0 # Local model cost is 0

                # Rate limiting update might need adjustment/removal
                # self.rate_limiter.add_request(request_token_count=tot_tokens, current_time=start_time)
                if self.verbose:
                    print(f"    SM Adapter generated {len(resp_data['choices'])} response(s). Entropy: {entropy:.4f}. SeqLen: {seq_len}")
                break # Success

            except Exception as e:
                print(f'[SM Adapter] RETRYING LLM REQUEST {retry+1}/{MAX_RETRIES}...')
                print(f"Error details: {e}")
                time.sleep(1) # Simple backoff

        # Return structure similar to original async version's output tuple items
        if resp_data is None:
            return None

        # Expected: query_idx, resp_data, tot_cost, tot_tokens
        return query_idx, resp_data, tot_cost, tot_tokens

    # Replace _generate_concurrently with sync version
    def _generate_sequentially(self, few_shot_templates, query_examples, n_history): # Add n_history
        """Perform sequential generation using the synchronous adapter call."""
        # Structure: list of lists, outer indexed by query_idx, inner has results from each template
        all_results = [[] for _ in range(len(query_examples))]
        total_calls = 0
        if self.verbose: print(f"  Generating SM predictions sequentially for {len(query_examples)} points, {len(few_shot_templates)} templates...")

        # Loop structure mirroring original async coroutine creation
        for template in few_shot_templates:
            for query_idx, query_example in enumerate(query_examples):
                total_calls += 1
                # Call the synchronous generation function, passing n_history
                response_package = self._sync_generate_with_adapter(template, query_example, query_idx, n_history)
                # response_package should be (query_idx, resp_data, tot_cost, tot_tokens) or None

                if response_package is not None:
                    q_idx, resp_data, cost, tokens = response_package
                    if q_idx == query_idx:
                         # Store [resp, cost, tokens] structure for compatibility with _predict parsing
                         all_results[query_idx].append([resp_data, cost, tokens])
                    else:
                         print(f"Warning: Mismatched query index during SM generation! Expected {query_idx}, got {q_idx}")
                # else: Keep inner list empty for this query_idx if call failed

        successful_queries = sum(1 for res_list in all_results if res_list)
        if self.verbose: print(f"  Finished {total_calls} SM generation calls for {successful_queries}/{len(query_examples)} query points.")
        return all_results

    def _predict(self, all_prompt_templates, query_examples, n_history): # Add n_history
        """Synchronous prediction using the adapter with robust parsing."""
        start = time.time()
        all_preds = [] # Stores lists of numerical predictions for each query_example
        tot_tokens = 0
        tot_cost = 0
        successful_queries = 0

        # Call the sequential generation function
        # chunk_results format: list (size=num_queries) of lists (size=num_templates) of [resp_data, cost, tokens]
        chunk_results = self._generate_sequentially(all_prompt_templates, query_examples, n_history) # Pass n_history
        successful_queries = sum(1 for res_list in chunk_results if res_list)

        for i, sample_response_list in enumerate(chunk_results):
            # Each item in sample_response_list corresponds to one template for this query_idx
            # Determine expected number of generations for padding/truncation later
            num_expected_gens = max(1, int(self.n_gens)) # Total expected across all templates

            if not sample_response_list: # No successful responses for this query_idx
                if self.verbose: print(f"  Warning: No valid LLM response received for query index {i}")
                # Pad with NaNs based on expected number of generations
                sample_preds_for_query = [np.nan] * num_expected_gens
            else:
                sample_preds_for_query = []
                # Aggregate generations across all templates for this query_idx
                all_gens_text = []
                for resp_package in sample_response_list:
                     # resp_package is [resp_data, cost, tokens]
                     resp_data, cost, tokens = resp_package
                     # Extend with content from all choices in this response
                     all_gens_text.extend([choice['message']['content'] for choice in resp_data.get('choices', [])])
                     tot_cost += cost
                     tot_tokens += tokens

                # ===>>> Integrate Robust Parsing Logic Here <<<===
                for gen_text in all_gens_text:
                    extracted_value = np.nan # Default to NaN
                    # 1. Try the strict '## float ##' regex first
                    # Looks for ##, optional space, optional sign, digits/dot, optional space, ##
                    strict_match = re.findall(r"##\s*(-?[\d.]+)\s*##", gen_text)
                    if len(strict_match) == 1:
                        try:
                            extracted_value = float(strict_match[0])
                            # if self.verbose: print(f"  DEBUG: Strict regex matched: {extracted_value} in '{gen_text}'")
                        except ValueError:
                            if self.verbose: print(f"  Warning: Strict regex matched '{strict_match[0]}' but failed float conversion in '{gen_text}'.")
                            # Keep extracted_value as np.nan
                    else:
                        # 2. Fallback: Find the first likely float/int number in the string
                        # Looks for patterns like -0.123, .456, 100, 1e-5 etc.
                        # Takes the first match found in the string.
                        fallback_match = re.search(r"([-+]?\s*\d*\.?\d+([eE][-+]?\d+)?)", gen_text)
                        if fallback_match:
                            try:
                                potential_num_str = fallback_match.group(1).strip() # Get matched number string
                                extracted_value = float(potential_num_str)
                                if self.verbose: print(f"  Warning: Used fallback regex for '{gen_text}'. Extracted: {extracted_value}")
                            except ValueError:
                                # This shouldn't happen often if regex is correct, but handle anyway
                                if self.verbose: print(f"  Warning: Fallback regex matched '{fallback_match.group(1)}' but failed float conversion in '{gen_text}'.")
                        else:
                             # Only log warning if strict match also failed
                             if not strict_match and self.verbose:
                                  print(f"  Warning: No number found via strict or fallback regex in '{gen_text}'")

                    sample_preds_for_query.append(extracted_value)
                # ===>>> End Robust Parsing Logic <<<===

                # Pad or truncate to expected number of generations (self.n_gens)
                while len(sample_preds_for_query) < num_expected_gens:
                    sample_preds_for_query.append(np.nan)
                sample_preds_for_query = sample_preds_for_query[:num_expected_gens]

            all_preds.append(sample_preds_for_query)

        end = time.time()
        time_taken = end - start

        success_rate = successful_queries / len(query_examples) if query_examples else 0
        if self.verbose: print(f"  SM Prediction processing finished in {time_taken:.2f}s. Query success rate: {success_rate:.2%}")

        # --- Calculate Mean/Std Dev ---
        all_preds = np.array(all_preds).astype(float) # Shape: (num_queries, n_gens)

        # Add check for empty or invalid shape (as developed before)
        num_queries = len(query_examples)
        if all_preds.size == 0:
            if self.verbose: print("  Warning: No predictions generated in _predict (all_preds is empty). Returning NaNs.")
            y_mean = np.full(num_queries, np.nan) if num_queries > 0 else np.array([])
            y_std = np.full(num_queries, np.nan) if num_queries > 0 else np.array([])

        elif all_preds.ndim == 2 and all_preds.shape[1] > 0:
            # Expected case: 2D array
            with np.warnings.catch_warnings():
                np.warnings.filterwarnings('ignore', r'Mean of empty slice')
                y_mean = np.nanmean(all_preds, axis=1)
                y_std = np.nanstd(all_preds, axis=1)
            # Impute NaNs from failed generations within a query
            nan_means_mask = np.isnan(y_mean)
            if np.any(nan_means_mask):
                 valid_preds_flat = all_preds[~np.isnan(all_preds)]
                 global_mean_impute = np.mean(valid_preds_flat) if valid_preds_flat.size > 0 else 0.0
                 y_mean[nan_means_mask] = global_mean_impute
                 valid_stds = y_std[~np.isnan(y_std)]
                 global_std_impute = np.mean(valid_stds) if valid_stds.size > 0 else 1e-5
                 y_std[nan_means_mask] = global_std_impute
        else: # Handle unexpected shapes
             print(f"  Warning: 'all_preds' has unexpected shape {all_preds.shape}. Returning global mean/std.")
             valid_preds_flat = all_preds[~np.isnan(all_preds)]
             global_mean_impute = np.mean(valid_preds_flat) if valid_preds_flat.size > 0 else 0.0
             global_std_impute = np.std(valid_preds_flat) if valid_preds_flat.size > 0 else 1e-5
             y_mean = np.full(num_queries, global_mean_impute)
             y_std = np.full(num_queries, global_std_impute)

        # Floor standard deviation
        y_std[np.isnan(y_std)] = 1e-5 # Catch remaining NaNs
        y_std[y_std < 1e-5] = 1e-5

        # Final shape check
        if num_queries > 0:
            if y_mean.shape != (num_queries,): y_mean = np.full(num_queries, np.nanmean(y_mean) if y_mean.size > 0 else 0.0)
            if y_std.shape != (num_queries,): y_std = np.full(num_queries, np.nanmean(y_std) if y_std.size > 0 else 1e-5)
        elif num_queries == 0: # Ensure empty arrays if no queries
             y_mean = np.array([])
             y_std = np.array([])


        return y_mean, y_std, success_rate, tot_cost, tot_tokens, time_taken


    # Modify _evaluate_candidate_points to be sync
    def _evaluate_candidate_points(self, observed_configs, observed_fvals, candidate_configs,
                                     use_context='full_context', use_feature_semantics=True, return_ei=False):
        '''Synchronously evaluate candidate points using the LLM adapter.'''


        if self.prompt_setting is not None:
            use_context = self.prompt_setting

        all_run_cost = 0
        all_run_time = 0

        # --- Determine n_history ---
        n_history = len(observed_configs) if observed_configs is not None else 0

        # --- Recalibration (Needs review if used) ---
        if self.use_recalibration:
             print("Warning: Recalibration logic requires review for synchronous execution.")
             # recalibrator, tot_cost, time_taken = self._get_recalibrator(observed_configs, observed_fvals) # Assume sync for now
             # ... (handle potential issues) ...
             pass # Skipping adaptation for now

        # --- Prompt Generation ---
        # Ensure gen_prompt_tempates (from utils) doesn't use async
        try:
             all_prompt_templates, query_examples = gen_prompt_tempates(
                 self.task_context, observed_configs, observed_fvals, candidate_configs,
                 n_prompts=self.n_templates, bootstrapping=self.bootstrapping,
                 use_context=use_context, use_feature_semantics=use_feature_semantics,
                 shuffle_features=self.shuffle_features, apply_warping=self.apply_warping
             )
        except Exception as e:
             print(f"Error during SM gen_prompt_tempates: {e}")
             n_candidates = len(candidate_configs)
             return np.zeros(n_candidates), np.ones(n_candidates)*1e-5, 0.0, 0.0 # Dummy return on error

        if self.verbose:
             print('*'*100)
             print(f'SM: Number of prompt_templates: {len(all_prompt_templates)}')
             print(f'SM: Number of query_examples: {len(query_examples)}')
             if all_prompt_templates and query_examples:
                  try:
                     print("SM Example Prompt:")
                     print(all_prompt_templates[0].format(Q=query_examples[0]['Q']))
                  except Exception as fmt_e:
                     print(f"Error formatting SM prompt example: {fmt_e}")
             else:
                 print("SM: Prompt/Query templates empty or invalid.")
             print('*'*100)

        # --- Call Synchronous Predict, passing n_history ---
        response = self._predict(all_prompt_templates, query_examples, n_history)
        y_mean, y_std, success_rate, tot_cost, tot_tokens, time_taken = response

        all_run_cost += tot_cost
        all_run_time += time_taken

        # --- Recalibration Application (Needs review if used) ---
        if self.recalibrator is not None:
             # Assuming recalibrator call is synchronous
             recalibrated_res = self.recalibrator(y_mean, y_std, 0.68)
             y_std_recal = np.abs(recalibrated_res.upper - recalibrated_res.lower) / 2
             y_std = np.maximum(y_std_recal, 1e-5) # Use recalibrated, ensure floor

        # --- Return or Calculate EI ---
        if not return_ei:
            return y_mean, y_std, all_run_cost, all_run_time
        else:

            print(f"We calculate EI")

            # Calculate EI (standard numpy code)
            if self.lower_is_better:
                best_fval = np.nanmin(observed_fvals.to_numpy()) if not observed_fvals.empty else 0
                delta = -1 * (y_mean - best_fval)
            else:
                best_fval = np.nanmax(observed_fvals.to_numpy()) if not observed_fvals.empty else 0
                delta = y_mean - best_fval

            with np.errstate(divide='ignore', invalid='ignore'):
                Z = delta / y_std
                ei = np.where(y_std > 1e-9, delta * norm.cdf(Z) + y_std * norm.pdf(Z), 0.0)
                ei[np.isnan(ei)] = 0.0 # Handle NaNs


                # a way to maximise exploitation 
                # not caring about other parameters that haven't being explored  
                # upper confidence bound 

            return ei, y_mean, y_std, all_run_cost, all_run_time

    # Modify select_query_point to remove asyncio.run
    def select_query_point(self, observed_configs, observed_fvals, candidate_configs):
        '''Select the next query point using expected improvement (synchronous).'''
        print("\n--- Starting select_query_point (SM) ---")
        start_time_select = time.time()

        observed_configs_orig = observed_configs # Keep originals if warping
        candidate_configs_orig = candidate_configs

        # --- Warping ---
        if self.apply_warping:
            if self.warping_transformer is None:
                 print("Warning: apply_warping is True but warping_transformer is None in SM.")
            else:
                 try:
                      if not observed_configs.empty:
                           observed_configs = self.warping_transformer.warp(observed_configs)
                      if not candidate_configs.empty:
                           candidate_configs = self.warping_transformer.warp(candidate_configs)
                 except Exception as e:
                      print(f"Error during SM warping: {e}. Proceeding with original configs.")
                      observed_configs = observed_configs_orig
                      candidate_configs = candidate_configs_orig
        
        

        # --- Call synchronous evaluate function ---
        # We need EI to select the best point
        eval_result = self._evaluate_candidate_points(
            observed_configs, observed_fvals, candidate_configs, return_ei=True
        )
        # -------------------------------------------
        ei, y_mean, y_std, cost, time_taken_eval = eval_result # time_taken includes prediction time
        
        # --- Select best point based on chosen acquisition strategy ---
        best_point_index = -1
        acquisition_scores = None

        if self.acquisition_strategy == 'EI':
            acquisition_scores = ei
            if acquisition_scores is not None and acquisition_scores.size > 0:
                # Handle case where all EI values are 0 or NaN
                if np.all(acquisition_scores <= 0) or np.all(np.isnan(acquisition_scores)):
                    print("  Warning: All EI values non-positive or NaN in SM. Selecting based on predicted mean.")
                    # Fallback logic is handled below common to EI and UCB if scores are all NaN
                else:
                    best_point_index = np.nanargmax(acquisition_scores) # Use nanargmax for safety
            else:
                print("  Warning: EI calculation failed or produced empty result.")
        
        elif self.acquisition_strategy == 'UCB':
            if y_mean is not None and y_std is not None and y_mean.size > 0 and y_std.size > 0:
                if self.lower_is_better:
                    # For minimization, we use LCB: mean - kappa * std, and pick the minimum
                    acquisition_scores = y_mean - self.ucb_kappa * y_std
                    if not np.all(np.isnan(acquisition_scores)):
                        best_point_index = np.nanargmin(acquisition_scores)
                    else:
                        print("  Warning: All UCB (LCB) values are NaN.")
                else:
                    # For maximization, we use UCB: mean + kappa * std, and pick the maximum
                    acquisition_scores = y_mean + self.ucb_kappa * y_std
                    if not np.all(np.isnan(acquisition_scores)):
                        best_point_index = np.nanargmax(acquisition_scores)
                    else:
                        print("  Warning: All UCB values are NaN.")
            else:
                print("  Warning: Cannot calculate UCB due to missing mean or std predictions.")
        else:
            raise ValueError(f"Unknown acquisition strategy: {self.acquisition_strategy}")

        # Fallback if no best_point_index could be determined from acquisition scores (e.g., all NaNs)
        if best_point_index == -1:
            print(f"  Warning: Could not determine best point using {self.acquisition_strategy}. Selecting based on predicted mean.")
            if y_mean is not None and y_mean.size > 0 and not np.all(np.isnan(y_mean)):
                best_point_index = np.argmin(y_mean) if self.lower_is_better else np.argmax(y_mean)
            else:
                print("  Error: All predicted means are NaN. Cannot select point. Defaulting to index 0.")
                best_point_index = 0 # Select first candidate as last resort
        
        if best_point_index < 0 or best_point_index >= len(candidate_configs_orig): # Ensure index is valid
            print(f"  Warning: Invalid best_point_index ({best_point_index}) after selection. Defaulting to index 0.")
            best_point_index = 0

        # --- Unwarp selected point ---
        # Use original candidate_configs DF for indexing
        if self.apply_warping and self.warping_transformer is not None:
             # Select the row using the index determined from warped EI/mean
             # Then unwarp just that selected row/point
             selected_warped_point_df = candidate_configs.iloc[[best_point_index], :]
             try:
                  best_point_df = self.warping_transformer.unwarp(selected_warped_point_df)
             except Exception as e:
                  print(f"Error unwarping selected point: {e}. Returning original unwarped point.")
                  best_point_df = candidate_configs_orig.iloc[[best_point_index], :]
        else:
             best_point_df = candidate_configs_orig.iloc[[best_point_index], :]

        end_time_select = time.time()
        time_taken_select = end_time_select - start_time_select
        print(f"--- select_query_point finished in {time_taken_select:.2f}s (includes evaluation time: {time_taken_eval:.2f}s) ---")

        # Return the selected point (as DF), total cost from evaluation, total time from evaluation
        return best_point_df, cost, time_taken_eval

    def reset_sm_attention_results(self):
        """Clears the stored SM-specific attention results."""
        self.sm_attention_results = []