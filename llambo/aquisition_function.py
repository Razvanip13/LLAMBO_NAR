# llambo/acquisition_function.py
import os
import random
import math
import time
import re
import json 

# import openai # No longer needed
# import asyncio # No longer needed
import numpy as np
import pandas as pd
# from aiohttp import ClientSession # No longer needed
from langchain import FewShotPromptTemplate # Keep if _gen_prompt_templates uses it
from langchain import PromptTemplate      # Keep if _gen_prompt_templates uses it
from llambo.rate_limiter import RateLimiter
# from .gemma_adapter import GemmaLocalAdapter # Import adapter if in separate file

# Remove OpenAI config
# openai.api_type = os.environ["OPENAI_API_TYPE"]
# ... etc ...


class LLM_ACQ:
     # Add llm_adapter argument
     def __init__(self, task_context, n_candidates, n_templates, lower_is_better,
                    jitter=False, rate_limiter=None, warping_transformer=None, chat_engine=None,
                    prompt_setting=None, shuffle_features=False,
                    llm_adapter=None, acq_attention_filepath=None): # <-- ADDED
          '''Initialize the LLM Acquisition function.'''
          self.task_context = task_context
          self.n_candidates = n_candidates
          self.n_templates = n_templates
          # Ensure n_gens is at least 1, handle potential division by zero
          self.n_gens = max(1, int(n_candidates / n_templates)) if n_templates > 0 else n_candidates
          self.lower_is_better = lower_is_better
          self.apply_jitter = jitter
          if rate_limiter is None:
               # Adjust rate limiter if needed for local, or remove its usage later
               self.rate_limiter = RateLimiter(max_tokens=40000, time_frame=60)
          else:
               self.rate_limiter = rate_limiter
          # --- Warping Transformer ---
          self.warping_transformer = warping_transformer
          self.apply_warping = warping_transformer is not None
          # --- Adapter ---
          self.llm_adapter = llm_adapter # <-- STORED
          if self.llm_adapter is None:
               raise ValueError("LLM Adapter instance must be provided to LLM_ACQ.")
          # ----------------
          self.chat_engine = chat_engine # Kept for potential compatibility elsewhere
          self.prompt_setting = prompt_setting
          self.shuffle_features = shuffle_features
          self.current_n_history = 0 # Track history length for logging

          # Add public attribute to store ACQ-specific attention results
          self.acq_attention_results = []
          self.acq_attention_filepath = acq_attention_filepath 
          # collecting one attention vector for each trial 
          self.got_attention_vector = False 

          assert type(self.shuffle_features) == bool, 'shuffle_features must be a boolean'

     def _jitter(self, desired_fval):
          # ... (keep this function as is) ...
          if not self.apply_jitter:
               return desired_fval
          # Ensure attributes exist before access
          if not all(hasattr(self, attr) for attr in ['observed_best', 'observed_worst', 'alpha']):
               print("Warning: Skipping jitter, required attributes not set.")
               return desired_fval

          # Ensure bounds are valid
          low_bound = min(desired_fval, self.observed_best)
          high_bound = max(desired_fval, self.observed_best)
          if low_bound >= high_bound: # Avoid error if bounds are equal or inverted
               return desired_fval

          jittered = np.random.uniform(low=low_bound, high=high_bound, size=1).item()
          return jittered

     def _count_decimal_places(self, n):
          # ... (keep this function as is) ...
          try:
               s = format(n, '.10f')
               if '.' not in s:
                    return 0
               # Handle potential trailing zeros if needed, though rstrip('0') is good
               decimal_part = s.split('.')[1].rstrip('0')
               return len(decimal_part)
          except Exception as e:
               print(f"Warning: Error counting decimal places for {n}: {e}")
               return 2 # Default or fallback precision

     def _prepare_configurations_acquisition(
          self,
          observed_configs=None,
          observed_fvals=None,
          seed=None,
          use_feature_semantics=True,
          shuffle_features=False # Already part of self, maybe remove arg?
     ):
          # ... (keep most of this function as is, ensure it handles warping flag) ...
          # This function generates the list of dictionaries for FewShotPromptTemplate
          examples = []
          if observed_configs is None and observed_fvals is not None:
               # Handle case where only desired_fval is passed (for query example)
               examples = [{'A': f'{observed_fvals:.6f}'}] # Use the value directly
               return examples
          elif observed_configs is None:
               return [] # Return empty if no configs provided

          # --- Ensure sorting/shuffling happens correctly ---
          current_configs = observed_configs.copy()
          current_fvals = observed_fvals.copy() if observed_fvals is not None else None

          if seed is not None:
               np.random.seed(seed)
               shuffled_indices = np.random.permutation(current_configs.index)
               current_configs = current_configs.loc[shuffled_indices]
               if current_fvals is not None:
                    current_fvals = current_fvals.loc[shuffled_indices]
          elif current_fvals is not None and not current_fvals.empty:
               # Sort by fvals only if fvals are provided and not empty
               sort_col = current_fvals.columns[0]
               ascending_sort = not self.lower_is_better # Sort ascending if higher is better
               sorted_indices = current_fvals.sort_values(by=sort_col, ascending=ascending_sort).index
               current_configs = current_configs.loc[sorted_indices]
               current_fvals = current_fvals.loc[sorted_indices]
          # else: keep original order if no seed and no fvals for sorting

          if self.shuffle_features: # Use self.shuffle_features
               # This should likely happen *before* serialization
               # Make sure random seed is consistent if needed across calls
               shuffled_columns = np.random.permutation(current_configs.columns)
               current_configs = current_configs[shuffled_columns]
          # -------------------------------------------------

          # serialization logic (ensure self.apply_warping is used correctly)
          hyperparameter_names = current_configs.columns # Use potentially shuffled names
          for index, row in current_configs.iterrows():
               row_string = '## '
               for i, col_name in enumerate(hyperparameter_names): # Iterate using potentially shuffled names
                    # Get constraints for the original hyperparameter name
                    orig_hyp_name = col_name # In this case, they are the same unless shuffling map is stored
                    if orig_hyp_name not in self.task_context['hyperparameter_constraints']:
                         print(f"Warning: Hyperparameter '{orig_hyp_name}' not found in constraints. Skipping.")
                         continue # Or handle error appropriately
                    hyp_constraint = self.task_context['hyperparameter_constraints'][orig_hyp_name]
                    hyp_type = hyp_constraint[0]
                    hyp_transform = hyp_constraint[1]

                    if use_feature_semantics:
                         row_string += f'{orig_hyp_name}: '
                    else:
                         # Using X{i+1} might be confusing if columns are shuffled
                         row_string += f'{orig_hyp_name}: ' # Safer to just use the name

                    value = row[col_name] # Access value using the current column name

                    # Determine precision
                    n_dp = 0
                    if hyp_type in ['int', 'float', 'ordinal']:
                         # Get bounds from constraint for precision calculation
                         bounds_or_values = hyp_constraint[2]
                         if hyp_type == 'float':
                              n_dp = self._count_decimal_places(bounds_or_values[0]) # Use lower bound for precision reference
                         elif hyp_type == 'ordinal':
                              # Use precision of the first value in the list
                              n_dp = self._count_decimal_places(bounds_or_values[0]) if bounds_or_values else 0
                         # For int, n_dp remains 0 unless log-transformed

                    # --- Formatting based on warping ---
                    if self.apply_warping: # Check the flag stored in self
                         if hyp_type == 'int' and hyp_transform != 'log':
                              row_string += str(int(round(value))) # Round before int casting if warped value is float
                         elif hyp_type == 'float' or (hyp_type == 'int' and hyp_transform == 'log'):
                              # If log warped int, treat as float for printing
                              n_dp = max(n_dp, 2) # Ensure reasonable precision for warped floats/log-ints
                              row_string += f'{value:.{n_dp}f}'
                         elif hyp_type == 'ordinal':
                              # Ordinal usually not warped, but print with determined precision
                              row_string += f'{value:.{n_dp}f}'
                         else: # Categorical? Ensure constraints cover this
                              row_string += str(value)
                    else: # No warping applied
                         if hyp_type == 'int':
                              row_string += str(int(value))
                         elif hyp_type in ['float', 'ordinal']:
                              row_string += f'{value:.{n_dp}f}'
                         else: # Categorical?
                              row_string += str(value)

                    if i != len(hyperparameter_names)-1:
                         row_string += ', '
               row_string += ' ##'
               example = {'Q': row_string}

               if current_fvals is not None:
                    # Ensure index exists after potential sorting/shuffling
                    if index in current_fvals.index:
                         fval_row_index = current_fvals.index.get_loc(index)
                         perf = f'{current_fvals.iloc[fval_row_index, 0]:.6f}' # Use iloc with index position
                         example['A'] = perf
                    else:
                         # Should not happen if indices are aligned, but handle defensively
                         example['A'] = 'NaN'

               examples.append(example)

          return examples

     # Modify to store n_history
     def _gen_prompt_tempates_acquisitions(
          self,
          observed_configs,
          observed_fvals,
          desired_fval,
          n_prompts=1,
          use_context='full_context',
          use_feature_semantics=True,
          # shuffle_features is now self.shuffle_features
     ):
          '''Generate prompt templates for acquisition function.'''
          # --- STORE n_history ---
          self.current_n_history = len(observed_configs) if observed_configs is not None else 0
          # -----------------------
          all_prompt_templates = []
          all_query_templates = []

          # Use self.shuffle_features inside _prepare_configurations_acquisition
          # Need to ensure consistent shuffling if n_prompts > 1 and seed is used?
          # Original code used seed=i, implying different shuffles per template. Let's keep that.

          for i in range(n_prompts):
               # Pass shuffle_features flag from self
               few_shot_examples = self._prepare_configurations_acquisition(
                    observed_configs, observed_fvals, seed=i,
                    use_feature_semantics=use_feature_semantics,
                    shuffle_features=self.shuffle_features # Pass instance variable
               )
               jittered_desired_fval = self._jitter(desired_fval)

               # contextual information about the task (keep as is)
               task_context = self.task_context
               # ... (extract model, task, features, etc. as before) ...
               model = task_context.get('model', 'Unknown Model')
               task = task_context.get('task', 'Unknown Task')
               tot_feats = task_context.get('tot_feats', '?')
               cat_feats = task_context.get('cat_feats', '?')
               num_feats = task_context.get('num_feats', '?')
               n_classes = task_context.get('n_classes', '?')
               metric_name = task_context.get('metric', 'unknown_metric')
               metric = 'mean squared error' if metric_name == 'neg_mean_squared_error' else metric_name
               num_samples = task_context.get('num_samples', '?')
               hyperparameter_constraints = task_context.get('hyperparameter_constraints', {})

               example_template = """Performance: {A}\nHyperparameter configuration: {Q}"""

               example_prompt = PromptTemplate(
                    input_variables=["Q", "A"],
                    template=example_template
               )

               # --- Build Prefix String ---
               prefix = f"The following are examples of performance of a {model} measured in {metric} and the corresponding model hyperparameter configurations."
               if use_context == 'full_context':
                    # Safely access task context keys
                    task_info = f" The model is evaluated on a tabular {task} task"
                    if task == 'classification':
                         task_info += f" containing {n_classes} classes."
                    prefix += task_info
                    prefix += f" The tabular dataset contains {num_samples} samples and {tot_feats} features ({cat_feats} categorical, {num_feats} numerical)."
               prefix += f" The allowable ranges for the hyperparameters are:\n"

               # Iterate through constraints safely
               for hyp_idx, (hyperparameter, constraint) in enumerate(hyperparameter_constraints.items()):
                    if not isinstance(constraint, (list, tuple)) or len(constraint) < 3:
                         print(f"Warning: Invalid constraint format for {hyperparameter}. Skipping.")
                         continue

                    hyp_type = constraint[0]
                    hyp_transform = constraint[1]
                    bounds_or_values = constraint[2]
                    range_str_parts = []

                    if hyp_type in ['float', 'int']:
                         if not isinstance(bounds_or_values, (list, tuple)) or len(bounds_or_values) != 2:
                              print(f"Warning: Invalid bounds format for {hyperparameter}. Skipping.")
                              continue
                         lower_bound_orig, upper_bound_orig = bounds_or_values
                         n_dp = 0
                         if hyp_type == 'float':
                              n_dp = self._count_decimal_places(lower_bound_orig)

                         # Handle warping for display bounds if needed
                         if self.apply_warping and hyp_transform == 'log':
                              lower_bound_disp = np.log10(lower_bound_orig) if lower_bound_orig > 0 else -np.inf
                              upper_bound_disp = np.log10(upper_bound_orig) if upper_bound_orig > 0 else np.inf
                              # Determine precision for log scale
                              if hyp_type == 'int': # Log-int needs float precision for display
                                   n_dp = max(2, self._count_decimal_places(lower_bound_disp)) # Use at least 2 decimals for log
                              else: # Log-float
                                   n_dp = max(n_dp, self._count_decimal_places(lower_bound_disp)) # Use original or log precision
                         else:
                              lower_bound_disp, upper_bound_disp = lower_bound_orig, upper_bound_orig
                              if hyp_type == 'int': n_dp = 0

                         range_str_parts.append(f"[{lower_bound_disp:.{n_dp}f}, {upper_bound_disp:.{n_dp}f}]")

                         if self.apply_warping and hyp_transform == 'log':
                              range_str_parts.append(f"(log scale, precise to {n_dp} decimals)")
                         elif hyp_type == 'int':
                              range_str_parts.append("(int)")
                         else: # float, not log warped
                              range_str_parts.append(f"(float, precise to {n_dp} decimals)")

                    elif hyp_type == 'ordinal':
                         if not isinstance(bounds_or_values, (list, tuple)):
                              print(f"Warning: Invalid values format for ordinal {hyperparameter}. Skipping.")
                              continue
                         range_str_parts.append(f"(ordinal, must take value in {bounds_or_values})")
                    else: # Add handling for 'categorical' if needed
                         print(f"Warning: Unknown or unhandled hyperparameter type '{hyp_type}' for {hyperparameter}. Skipping range display.")
                         continue

                    feature_name = hyperparameter if use_feature_semantics else f"X{hyp_idx+1}"
                    prefix += f"- {feature_name}: {' '.join(range_str_parts)}\n"
               # --- End Prefix Build ---

               prefix += f"Recommend a configuration that can achieve the target performance of {jittered_desired_fval:.6f}. "
               if use_context in ['partial_context', 'full_context']:
                    # Keep constraints as in original code
                    prefix += "Do not recommend values at the minimum or maximum of allowable range, do not recommend rounded values. Recommend values with highest possible precision, as requested by the allowed ranges. "
               prefix += f"Your response must only contain the predicted configuration, in the format ## configuration ##. You must not explain your choice. Answer the request precisely and as short as possible. Respect the answer structure\n" # Ensure newline at end

               suffix = """\nPerformance: {A}\nHyperparameter configuration:""" # Added newline for clarity

               few_shot_prompt = FewShotPromptTemplate(
                    examples=few_shot_examples,
                    example_prompt=example_prompt,
                    prefix=prefix,
                    suffix=suffix,
                    input_variables=["A"],
                    example_separator="\n" # Use newline as separator
               )
               all_prompt_templates.append(few_shot_prompt)

               # Create query example structure expected by adapter/template
               # Note: _prepare_configurations_acquisition expects fvals, not just a single value
               # Let's create a dummy structure or pass the value directly
               # query_examples_prep = self._prepare_configurations_acquisition(observed_fvals=jittered_desired_fval)
               # all_query_templates.append(query_examples_prep)
               # Simpler: Pass the target value directly, template uses {A}
               all_query_templates.append({'A': f'{jittered_desired_fval:.6f}'})


          return all_prompt_templates, all_query_templates


     # Replace _async_generate with sync version using adapter
     def _sync_generate_with_adapter(self, user_message):
          """Synchronous generation using the local adapter for ACQ."""
          if not self.llm_adapter:
               raise ValueError("LLM Adapter not provided to LLM_ACQ")

          MAX_RETRIES = 5
          resp_data = None
          entropy = np.nan # Initialize entropy
          seq_len = 0      # Initialize seq_len

          for retry in range(MAX_RETRIES):
               try:
                    start_time = time.time()
                    # Rate limiting might need adjustment/removal for local model
                    # self.rate_limiter.add_request(request_text=user_message, current_time=start_time)

                    # --- CALL ADAPTER ---
                    # Always log attention for ACQ, pass n_history
                    # print(f"Aquisition function prompt: {user_message}")



                    mock_openai_response, current_entropy, current_seq_len, avg_head_attention_tensor = self.llm_adapter.generate(
                    prompt_text=user_message,
                    n_gens=self.n_gens, # Use n_gens from ACQ config
                    temperature=1, # Match original params
                    max_new_tokens=500, # Sufficient for config generation
                    top_p=0.95, # Match original params
                    log_attention_data=True,
                    n_history_for_log=self.current_n_history, # <<< Pass n_history
                    source_tag='ACQ' # Tag the source
                    )
                    # ------------------

                    # --- Store ACQ entropy internally ---
                    entropy = current_entropy # Store returned entropy
                    seq_len = current_seq_len # Store returned seq_len


                    # ---- MODIFICATION: Write attention data to JSONL file ----
                    if avg_head_attention_tensor is not None and self.acq_attention_filepath is not None:
                         try:
                              # Convert tensor to list of lists for JSON serialization
                              # Ensure it's on CPU before converting to list
                              attention_matrix_list = avg_head_attention_tensor.cpu().tolist() 

                              log_data_for_jsonl = {
                                   'n_history': self.current_n_history,
                                   'seq_len': current_seq_len,
                                   'source_tag': 'ACQ', # Good to keep track
                                   'entropy': current_entropy if not np.isnan(current_entropy) else None, # Optional: also log entropy
                                   'attention_matrix': attention_matrix_list
                              }

                              if not self.got_attention_vector: 
                                   with open(self.acq_attention_filepath, 'a') as f_jsonl: # 'a' for append mode
                                        json_record = json.dumps(log_data_for_jsonl)
                                        f_jsonl.write(json_record + '\n')

                                   self.got_attention_vector = True 
                                   
                         except Exception as e:
                              print(f"  Warning: Failed to write ACQ attention data to JSONL: {e}")
                    # ---- END MODIFICATION ----


                    if not np.isnan(entropy):
                         self.acq_attention_results.append({
                              'n_history': self.current_n_history,
                              'entropy': entropy,
                              'seq_len': seq_len
                         })

                    resp_data = mock_openai_response

                    # Dummy cost/token calculation
                    tot_tokens = resp_data['usage']['total_tokens']
                    tot_cost = 0.0 # Local model cost is 0

                    # Rate limiting update might need adjustment/removal
                    # self.rate_limiter.add_request(request_token_count=tot_tokens, current_time=start_time)

                    print(f"    ACQ Adapter generated {len(resp_data['choices'])} response(s). Entropy: {entropy:.4f}, SeqLen: {seq_len}")
                    print(resp_data['choices'])
                    # time.sleep(10)
                    break # Success

               except Exception as e:
                    print(f'[ACQ Adapter] RETRYING LLM REQUEST {retry+1}/{MAX_RETRIES}...')
                    print(f"Error details: {e}")
                    time.sleep(1) # Simple backoff

          # Return structure similar to original, but without async/await results
          if resp_data is None:
               return None

          # Expected return: (resp, tot_cost, tot_tokens)
          return resp_data, tot_cost, tot_tokens


     # Replace _async_generate_concurrently with sync version
     def _generate_all_candidates(self, prompt_templates, query_templates):
          """Generates candidates using the adapter sequentially for ACQ."""
          all_results = []

          print(f"Generating ACQ candidates sequentially for {len(prompt_templates)} templates...")
          for i, (prompt_template, query_template) in enumerate(zip(prompt_templates, query_templates)):
               print(f"  Template {i+1}/{len(prompt_templates)}")
               # Ensure query_template is the simple dict {'A': 'value'}
               if isinstance(prompt_template, FewShotPromptTemplate) and isinstance(query_template, dict) and 'A' in query_template:
                    final_prompt = prompt_template.format(A=query_template['A'])
               else:
                    print(f"  Warning: Invalid prompt/query template structure for template {i+1}. Skipping.")
                    all_results.append(None) # Append None to maintain structure
                    continue


               # Call the synchronous generation function
               response_package = self._sync_generate_with_adapter(final_prompt)

               print(response_package)
          #   print("Sleep time")
          #   time.sleep(10)

               # response_package should be (resp_data, tot_cost, tot_tokens) or None
               all_results.append(response_package)

          # Check if any calls succeeded
          successful_calls = sum(1 for res in all_results if res is not None)
          print(f"  Finished ACQ generation. {successful_calls}/{len(prompt_templates)} calls successful.")

          return all_results # List of (resp, cost, tokens) tuples or Nones


     def _convert_to_json(self, response_str):
          # ... (keep this function as is, but add error handling) ...
          response_json = {}
          try:
               # Attempt to handle various potential formats, e.g., key: value, key=value
               # This regex is more robust: finds 'key': value or 'key': "value" etc.
               pattern = r"['\"]?([\w.-]+)['\"]?\s*[:=]\s*['\"]?([\d.eE+-]+)['\"]?"
               pairs = re.findall(pattern, response_str)

               if not pairs: # Fallback for simple comma/colon separation if regex fails
                    pairs_simple = response_str.split(',')
                    for pair_str in pairs_simple:
                         parts = pair_str.split(':')
                         if len(parts) == 2:
                              key = parts[0].strip().strip("'\"{} ")
                              value_str = parts[1].strip().strip("'\"{} ")
                              try:
                                   response_json[key] = float(value_str)
                              except ValueError:
                                   print(f"  Warning: Could not convert value '{value_str}' to float for key '{key}' in '{response_str}'")
                                   # Optionally store as string or skip
                         else:
                              print(f"  Warning: Could not parse pair '{pair_str}' in '{response_str}'")
                    return response_json # Return potentially partially parsed dict

               # Process regex results
               for key, value_str in pairs:
                    key = key.strip("'\" ")
                    value_str = value_str.strip("'\" ")
                    try:
                         response_json[key] = float(value_str)
                    except ValueError:
                         print(f"  Warning: Could not convert value '{value_str}' to float for key '{key}' in '{response_str}'")
                    # Optionally store as string or skip: response_json[key] = value_str

          except Exception as e:
               print(f"Error converting LLM response to JSON: '{response_str}'. Error: {e}")
               return {} # Return empty dict on error
          return response_json


     def _filter_candidate_points(self, observed_points_dicts, candidate_points_dicts, precision=8):
          # ... (keep this function as is, ensure it uses self.apply_warping correctly) ...
          # Minor improvements for robustness
          if not candidate_points_dicts:
               return pd.DataFrame()

          # Ensure observed points are dicts
          if isinstance(observed_points_dicts, pd.DataFrame):
               observed_points_dicts = observed_points_dicts.to_dict(orient='records')

          # --- Filtering existing points ---
          try:
               # Use frozenset for comparison to handle dict hashing issues and order independence
               observed_rounded_sets = {frozenset((key, round(value, precision)) for key, value in d.items()) for d in observed_points_dicts}
               
               filtered_candidates = []
               candidate_rounded_sets = set()
               for cand_dict in candidate_points_dicts:
                    try:
                         rounded_set = frozenset((key, round(value, precision)) for key, value in cand_dict.items())
                         if rounded_set not in observed_rounded_sets and rounded_set not in candidate_rounded_sets:
                              filtered_candidates.append(cand_dict)
                              candidate_rounded_sets.add(rounded_set)
                    except (TypeError, ValueError) as e:
                         print(f"  Warning: Could not round/process candidate point {cand_dict}. Error: {e}. Skipping.")
                         continue
          except Exception as e:
               print(f"  Error during duplicate filtering: {e}. Proceeding without duplicate removal.")
               filtered_candidates = candidate_points_dicts # Fallback

          if not filtered_candidates:
               print("  Warning: All candidate points were duplicates or invalid.")
               return pd.DataFrame()

          # --- Filtering based on constraints/ranges ---
          def is_within_range(value, constraint):
               value_type, transform, search_range = constraint
               # ... (logic from original, ensure self.apply_warping is used) ...
               # Adding safety checks
               if value is None or not isinstance(value, (int, float)): return False

               if value_type == 'int':
                    if not isinstance(search_range, (list, tuple)) or len(search_range) != 2: return False
                    min_val, max_val = search_range
                    # If warped, value might be float - check range first, then if it's close to an int
                    if self.apply_warping and transform == 'log':
                         min_val_log = np.log10(min_val) if min_val > 0 else -np.inf
                         max_val_log = np.log10(max_val) if max_val > 0 else np.inf
                         return min_val_log <= value <= max_val_log # Warped int is treated as float range
                    else:
                         # Check range and if it's essentially an integer
                         return min_val <= value <= max_val and abs(value - round(value)) < 1e-9

               elif value_type == 'float':
                    if not isinstance(search_range, (list, tuple)) or len(search_range) != 2: return False
                    min_val, max_val = search_range
                    if self.apply_warping and transform == 'log':
                         min_val_log = np.log10(min_val) if min_val > 0 else -np.inf
                         max_val_log = np.log10(max_val) if max_val > 0 else np.inf
                         return min_val_log <= value <= max_val_log
                    else:
                         return min_val <= value <= max_val

               elif value_type == 'ordinal':
                    if not isinstance(search_range, (list, tuple)): return False
                    # Check if value is close to any allowed ordinal value
                    return any(math.isclose(value, x, abs_tol=1e-6) for x in search_range)

               else: # Categorical or Unknown
                    print(f"Warning: Unsupported type '{value_type}' in constraint check.")
                    return False # Or handle categorical if needed

          def is_dict_within_ranges(d, ranges_dict):
               return all(key in ranges_dict and is_within_range(value, ranges_dict[key]) for key, value in d.items())

          hyperparameter_constraints = self.task_context.get('hyperparameter_constraints', {})
          try:
               filtered_candidates_in_range = [d for d in filtered_candidates if is_dict_within_ranges(d, hyperparameter_constraints)]
          except Exception as e:
               print(f" Error filtering by ranges: {e}. Returning candidates without range check.")
               filtered_candidates_in_range = filtered_candidates # Fallback

          if not filtered_candidates_in_range:
               print("  Warning: No candidates remaining after range filtering.")
               return pd.DataFrame()

          # Final conversion and duplicate removal
          final_df = pd.DataFrame(filtered_candidates_in_range)
          # Drop exact duplicates again after filtering (though set logic should handle most)
          final_df = final_df.drop_duplicates().reset_index(drop=True)
          return final_df



# Inside LLM_ACQ class

    # (Keep _jitter, _count_decimal_places, _prepare_configurations_acquisition,
    #  _gen_prompt_tempates_acquisitions, _sync_generate_with_adapter,
    #  _generate_all_candidates, _convert_to_json, _filter_candidate_points methods)

     def get_candidate_points(self, observed_configs, observed_fvals,
                              use_feature_semantics=True, use_context='full_context', alpha=-0.2):
          '''Generate candidate points for acquisition function (synchronous) with validation.'''
          print(f"\n--- Starting get_candidate_points ---")
          start_time = time.time()
          assert alpha >= -1 and alpha <= 1, 'alpha must be between -1 and 1'
          self.alpha = alpha

          if self.prompt_setting is not None:
               use_context = self.prompt_setting

          # --- Calculate desired f_val ---
          # ... (Keep your existing logic for calculating desired_fval) ...
          # ... (Ensure self.observed_best, self.observed_worst are set) ...
          # --- Example placeholder ---
          desired_fval = 0.5 # Replace with your actual calculation
          if observed_fvals is not None and not observed_fvals.empty:
               # ... calculate based on actual fvals ...
               fvals_np = observed_fvals.values.flatten()[~np.isnan(observed_fvals.values.flatten())]
               if len(fvals_np) > 0:
                    # ... calculate current_best, current_worst, range_val, desired_fval ...
                    pass # Keep your detailed logic here
          # ---------------------------
          self.desired_fval = desired_fval

          # --- Warping ---
          observed_configs_for_prompt = observed_configs
          if self.apply_warping and self.warping_transformer is not None:
               if observed_configs is not None and not observed_configs.empty:
                    try:
                         observed_configs_for_prompt = self.warping_transformer.warp(observed_configs)
                    except Exception as e:
                         print(f"Warning: Warping observed_configs failed: {e}")
                         observed_configs_for_prompt = observed_configs # Use original on error
          # ---------------------------

          # --- Generate Prompts ---
          try:
               prompt_templates, query_templates = self._gen_prompt_tempates_acquisitions(
                    observed_configs_for_prompt, observed_fvals, desired_fval,
                    n_prompts=self.n_templates, use_context=use_context,
                    use_feature_semantics=use_feature_semantics
               )
          except Exception as e:
               print(f"Error generating prompt templates: {e}")
               return pd.DataFrame(), 0.0, 0.0

          # --- Debug Print Prompt ---
          # ... (keep your prompt printing logic) ...

          # --- Generate Candidates with Validation Retry ---
          MAX_GENERATION_RETRIES = 5 # Limit overall retries
          # Aim to generate a decent pool of *valid* candidates before filtering
          # Maybe target n_candidates needed by SM, or slightly more?
          MIN_VALID_CANDIDATES_TARGET = max(10, self.n_candidates // 2) # Aim for at least 10 valid points, or half of n_cand

          retry = 0
          accumulated_valid_parsed_dicts = [] # Store validated dicts across retries
          expected_keys = set(self.task_context['hyperparameter_constraints'].keys()) # Get expected keys

          print(f"Targeting at least {MIN_VALID_CANDIDATES_TARGET} structurally valid candidates...")


          while len(accumulated_valid_parsed_dicts) < MIN_VALID_CANDIDATES_TARGET and retry < MAX_GENERATION_RETRIES:
               print(f"ACQ Generation attempt {retry+1}/{MAX_GENERATION_RETRIES} (Currently have {len(accumulated_valid_parsed_dicts)} valid)")
               # Call the synchronous generation function
               generation_responses = self._generate_all_candidates(prompt_templates, query_templates)

               # Process and *Validate* Responses for this attempt
               validated_this_attempt = []
               raw_parsed_count = 0
               for response_package in generation_responses:
                    if response_package is None: continue
                    resp_data, _, _ = response_package
                    for choice in resp_data.get('choices', []):
                         response_content = choice.get('message', {}).get('content', '')
                    try:
                         parsed_dict = self._convert_to_json(response_content) # Use your parsing function
                         if parsed_dict:
                              raw_parsed_count += 1

                              if set(parsed_dict.keys()) == expected_keys:
                                   validated_this_attempt.append(parsed_dict)
                              else:
                                   if self.verbose or True: # Always log validation failures for now
                                        print(f"  Warning: Discarding invalid candidate. Keys found: {set(parsed_dict.keys())}. Expected: {expected_keys}")

                    except Exception as parse_error:
                         print(f"  Error parsing/validating response content: '{response_content}'. Error: {parse_error}")
                         continue

               print(f"  Attempt {retry+1}: Parsed {raw_parsed_count} candidates, {len(validated_this_attempt)} were structurally valid.")
               accumulated_valid_parsed_dicts.extend(validated_this_attempt)
               # Remove exact duplicates accumulated across retries
               if accumulated_valid_parsed_dicts:
                    try:
                         accumulated_valid_parsed_dicts = [dict(t) for t in {tuple(sorted(d.items())) for d in accumulated_valid_parsed_dicts}]
                    except TypeError:
                         print("Warning: Cannot remove duplicates due to unhashable types.")
               print(f"  Total valid & unique candidates accumulated so far: {len(accumulated_valid_parsed_dicts)}")

               retry += 1
               if len(accumulated_valid_parsed_dicts) >= MIN_VALID_CANDIDATES_TARGET:
                    print("  Sufficient valid candidates accumulated.")
                    break
          # --- End Generation Retry Loop ---

          if not accumulated_valid_parsed_dicts:
               print("ERROR: Failed to generate ANY valid candidate points after all retries.")
               return pd.DataFrame(), 0.0, time.time() - start_time

          # --- Filter Accumulated *Valid* Points ---
          print(f"Filtering {len(accumulated_valid_parsed_dicts)} accumulated valid candidates...")
          # Ensure observed_configs is a list of dicts for filtering
          observed_dicts = observed_configs.to_dict(orient='records') if observed_configs is not None and not observed_configs.empty else []
          filtered_df = self._filter_candidate_points(
               observed_dicts,
               accumulated_valid_parsed_dicts # Filter the validated list
          )
          print(f"  {len(filtered_df)} candidates remaining after observation/range filtering.")
          # ----------------------------------

          if filtered_df.empty:
               print("Warning: All valid generated candidates were filtered out (duplicates or out of range). Returning empty DataFrame.")
               # You might want more sophisticated handling here, like returning the best *unfiltered* valid point

          # --- Unwarp Final Candidates ---
          final_candidates_unwarped = filtered_df # Start with filtered
          if self.apply_warping and self.warping_transformer is not None:
               if not filtered_df.empty:
                    try: final_candidates_unwarped = self.warping_transformer.unwarp(filtered_df)
                    except Exception as e: print(f"Error unwarping: {e}"); # Keep filtered_df on error
               # else: Keep empty df if filtered_df was empty
          # -----------------------------

          end_time = time.time()
          time_taken = end_time - start_time
          print(f"--- get_candidate_points finished in {time_taken:.2f}s ---")

          # Return the final set
          return final_candidates_unwarped, 0.0, time_taken

    



     def reset_acq_attention_results(self):
          """Clears the stored ACQ-specific attention results."""
          self.acq_attention_results = []