import time
import numpy as np
import pandas as pd
# Removed asyncio, aiohttp, openai imports
from llambo.rate_limiter import RateLimiter # Keep if needed, but adapter handles its own logic
from llambo.generative_sm_utils import gen_prompt_tempates
# Removed OpenAI specific setup

class LLM_GEN_SM:
    # Modify __init__ to accept the adapter
    def __init__(self, task_context, n_gens, lower_is_better, top_pct,
                 n_templates=1, rate_limiter=None,
                 verbose=False, # chat_engine removed
                 llm_adapter=None): # <-- ADD ADAPTER ARGUMENT
        '''Initialize the generative LLM surrogate model using a local adapter.'''
        self.task_context = task_context
        self.n_gens = n_gens
        self.lower_is_better = lower_is_better
        self.top_pct = top_pct
        self.n_templates = n_templates
        # Rate limiter might be less critical or need different logic for local models
        if rate_limiter is None:
            self.rate_limiter = RateLimiter(max_tokens=1000000, time_frame=60) # Adjust limits if needed
        else:
            self.rate_limiter = rate_limiter
        self.verbose = verbose

        self.llm_adapter = llm_adapter # <-- STORE ADAPTER INSTANCE
        if self.llm_adapter is None:
            raise ValueError("LLM Adapter must be provided to LLM_GEN_SM")

        self.current_n_history = 0 # Track history size for potential logging/analysis

    # Removed _async_generate
    # Removed _generate_concurrently

    # Add synchronous generation using the adapter
    def _sync_generate_with_adapter(self, user_message, n_preds):
        """Synchronous generation using the local adapter for GEN SM."""
        MAX_RETRIES = 3
        resp_data = None
        entropy = np.nan # SM doesn't directly use entropy, but adapter calculates it
        seq_len = 0

        for retry in range(MAX_RETRIES):
            try:
                start_time = time.time()
                # Rate limiting might need adjustment or removal for local model
                # self.rate_limiter.add_request(request_text=user_message, current_time=start_time)

                # --- CALL ADAPTER ---
                # Pass generation parameters
                # Note: Original used gpt-3.5-turbo-instruct (Completion). Adapter uses ChatCompletion format.
                # This might affect prompt effectiveness slightly.
                mock_openai_response, current_entropy, current_seq_len = self.llm_adapter.generate(
                    prompt_text=user_message,
                    n_gens=max(n_preds, 3), # Match original logic: max(n_preds, 3)
                    temperature=0.7, # Match original params
                    max_new_tokens=8, # Match original params (predicting "0" or "1")
                    top_p=0.95 # Match original params
                    # logprobs are not returned by the current adapter
                )
                # ------------------

                # Store attention results (associating with n_history) if adapter has storage
                if hasattr(self.llm_adapter, 'attention_results') and not np.isnan(current_entropy):
                     self.llm_adapter.attention_results.append({
                         'n_history': self.current_n_history, # Needs to be set by caller
                         'entropy': current_entropy,
                         'seq_len': current_seq_len,
                         'component': 'GEN_SM' # Add context
                     })

                resp_data = mock_openai_response
                entropy = current_entropy
                seq_len = current_seq_len

                # Dummy cost/token calculation for compatibility
                tot_tokens = resp_data['usage']['total_tokens']
                tot_cost = 0.0 # Zero cost for local model

                # Rate limiting update might need adjustment/removal
                # self.rate_limiter.add_request(request_token_count=tot_tokens, current_time=start_time)

                if self.verbose:
                    print(f"    Adapter generated {len(resp_data['choices'])} response(s) for GEN SM. Entropy: {entropy:.4f}, SeqLen: {seq_len}")
                break # Success

            except Exception as e:
                print(f'[GEN SM Adapter] RETRYING LLM REQUEST {retry+1}/{MAX_RETRIES}...')
                print(e)
                time.sleep(1) # Simple backoff

        if resp_data is None:
            return None

        # Return structure similar to original async, but without async session stuff
        return resp_data, tot_cost, tot_tokens

    # Modify process_response to work without logprobs
    def process_response_text(self, all_raw_responses):
        """Processes text responses ("0" or "1") to estimate probability."""
        pred_classes = [] # Store interpreted class (0 or 1) for each generation
        for response in all_raw_responses:
            # Extract text content from the mocked structure
            text = response.get('message', {}).get('content', '').strip()
            # Simple check for "1" or "0" - might need refinement based on actual model output format
            if '1' in text and '0' not in text: # Prioritize "1" if both appear? Or check start?
                pred_classes.append(1)
            elif '0' in text and '1' not in text:
                pred_classes.append(0)
            else:
                pred_classes.append(np.nan) # Unable to classify
        return pred_classes

    # Modify _predict to use the adapter and new processing
    def _predict_with_adapter(self, all_prompt_templates, query_examples):
        """Predicts classification probability using the adapter."""
        start = time.time()
        all_pred_classes = [] # Store lists of 0/1 predictions per query point
        tot_tokens = 0
        tot_cost = 0
        successful_predictions = 0

        n_preds_per_template = int(self.n_gens / self.n_templates)

        for query_idx, query_example in enumerate(query_examples):
            sample_responses_all_templates = []
            query_cost = 0
            query_tokens = 0
            query_successful = False

            for template in all_prompt_templates:
                final_prompt = template.format(Q=query_example['Q'])
                response_package = self._sync_generate_with_adapter(final_prompt, n_preds_per_template)

                if response_package:
                    resp_data, cost, tokens = response_package
                    query_cost += cost
                    query_tokens += tokens
                    query_successful = True
                    # Add all choices from this template's response
                    sample_responses_all_templates.extend(resp_data.get('choices', []))

            # Process collected responses for this query point
            if not sample_responses_all_templates:
                all_pred_classes.append([np.nan] * self.n_gens) # Pad with NaNs if all templates failed
            else:
                # Use the text processing function
                processed_classes = self.process_response_text(sample_responses_all_templates)
                # Pad/truncate to ensure n_gens results
                while len(processed_classes) < self.n_gens:
                    processed_classes.append(np.nan)
                all_pred_classes.append(processed_classes[:self.n_gens])
                tot_cost += query_cost
                tot_tokens += query_tokens
                successful_predictions += 1

        end = time.time()
        time_taken = end - start

        success_rate = successful_predictions / len(query_examples) if query_examples else 0

        # Calculate mean probability from the observed classes (fraction of 1s)
        pred_probs = [np.nanmean(classes) if not np.all(np.isnan(classes)) else np.nan for classes in all_pred_classes]
        mean_probs = np.array(pred_probs)
        # Handle cases where all generations failed (resulting in NaN mean)
        mean_probs[np.isnan(mean_probs)] = 0.5 # Impute with 0.5 probability? Or 0? Needs consideration.

        return mean_probs, success_rate, tot_cost, tot_tokens, time_taken

    # Modify _evaluate_candidate_points
    def _evaluate_candidate_points(self, observed_configs, observed_fvals, candidate_configs):
        '''Evaluate candidate points using the LLM adapter (GEN SM).'''
        # --- Store n_history for adapter logging ---
        self.current_n_history = len(observed_configs) if observed_configs is not None else 0
        # ------------------------------------------

        all_run_cost = 0
        all_run_time = 0

        # Generate prompts (remains the same)
        all_prompt_templates, query_examples = gen_prompt_tempates(self.task_context, observed_configs, observed_fvals, candidate_configs,
                                                                   self.lower_is_better, self.top_pct, n_prompts=self.n_templates)

        if self.verbose:
            print('*'*100)
            print(f'[GEN SM] Number of prompt templates: {len(all_prompt_templates)}')
            print(f'[GEN SM] Number of query examples: {len(query_examples)}')
            if all_prompt_templates and query_examples:
                 print(all_prompt_templates[0].format(Q=query_examples[0]['Q']))
            print('*'*100)

        # Call the adapted prediction function
        response = self._predict_with_adapter(all_prompt_templates, query_examples)
        pred_probs, success_rate, tot_cost, tot_tokens, time_taken = response

        all_run_cost += tot_cost
        all_run_time += time_taken

        return pred_probs, all_run_cost, all_run_time

    # select_query_point remains largely the same, just calls the adapted evaluation
    def select_query_point(self, observed_configs, observed_fvals, candidate_configs, return_raw_preds=False):
        '''Select the next query point using predicted probability from the adapter.'''
        # Warping/Unwarping logic might be needed if GEN SM uses warped inputs
        # The original GEN SM didn't seem to use the warping transformer explicitly,
        # but the prompt generation utils might need checking if warping is enabled globally.
        # Assuming no warping needed specifically for GEN SM based on original code.

        pred_probs, cost, time_taken = self._evaluate_candidate_points(observed_configs, observed_fvals, candidate_configs)

        best_point_index = np.argmax(pred_probs)

        best_point = candidate_configs.iloc[[best_point_index], :]

        if return_raw_preds:
            return best_point, pred_probs, cost, time_taken
        else:
            return best_point, cost, time_taken