# gemma_adapter.py
import torch
import numpy as np
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

class GemmaLocalAdapter:
    """
    Adapter to replace OpenAI API calls with a local Gemma model
    using the Hugging Face transformers library. Also extracts
    attention entropy for analysis.
    """

    def __init__(self, model_name="google/gemma-2b", device=None,
                 max_length=1024, layer_to_analyze=-1, verbose=False):
        """
        Initializes the adapter and loads the model.

        Args:
            model_name (str): Name of the Gemma model on Hugging Face Hub.
            device (str, optional): Device to load model onto ('cuda', 'cpu'). Auto-detects if None.
            max_length (int): Maximum sequence length for tokenizer truncation.
            layer_to_analyze (int): Index of the decoder layer from which to extract attention.
                                     -1 usually means the last layer.
            verbose (bool): If True, print detailed logs during generation.
        """
        print(f"Initializing GemmaLocalAdapter with model: {model_name}")
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.layer_to_analyze = layer_to_analyze
        self.verbose = verbose

        print(f"Loading model {self.model_name} onto {self.device}...")
        # Using device_map="auto" handles multi-GPU or CPU offloading if needed
        # Consider torch_dtype=torch.bfloat16 for memory saving if supported

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for quantization
            device_map="auto", 
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Set padding token if not present (common for decoder models)
        if self.tokenizer.pad_token_id is None:
            print("Setting pad_token_id to eos_token_id")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Use model's actual device if device_map="auto" was used
        self.device = self.model.device
        print(f"Gemma model '{self.model_name}' loaded successfully on device: {self.device}")

        # Storage for attention analysis results
        # List of dictionaries: {'n_history': int, 'entropy': float, 'seq_len': int, 'source': str}
        self.attention_results = []

    def _calculate_entropy(self, attention_vector):
        """Calculates Shannon entropy for a probability distribution (PyTorch tensor)."""
        if attention_vector is None or attention_vector.numel() == 0:
            return np.nan
        try:
            attention_vector = attention_vector.float() # Ensure float
            # Add epsilon for numerical stability before log
            attention_vector = attention_vector + 1e-9
            # Normalize just in case (should be close to 1 from softmax)
            attention_vector = attention_vector / attention_vector.sum()
            log_probs = torch.log2(attention_vector)
            entropy = -torch.sum(attention_vector * log_probs)
            return entropy.item()
        except Exception as e:
            print(f"  Error during entropy calculation: {e}")
            return np.nan


    #initially the temperature was set to 0.7
    @torch.no_grad()
    def generate(self, prompt_text, n_gens=1, temperature=1, top_p=0.95,
                 max_new_tokens=50, log_attention_data=False, n_history_for_log=None, source_tag=None):
        """
        Generates text using the local Gemma model, mimicking OpenAI API structure,
        and optionally extracts attention entropy from the prompt processing phase.

        Args:
            prompt_text (str): The input prompt.
            n_gens (int): Number of independent completions to generate.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling probability.
            max_new_tokens (int): Max tokens to generate for the completion.
            log_attention_data (bool): If True, calculate and log attention entropy.
            n_history_for_log (int, optional): The 'n_history' value associated with this call (for logging).
            source_tag (str, optional): Tag ('ACQ', 'SM', etc.) to identify the source of the call in logs.

        Returns:
            tuple: (mock_openai_response, entropy, sequence_length)
                   mock_openai_response (dict): Dict mimicking OpenAI structure.
                   entropy (float): Calculated Shannon entropy (or NaN).
                   sequence_length (int): Length of the tokenized prompt.
        """

        # We need to manage the cache the longer the sequence becomes 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.verbose: print(f"--- Adapter Generate Call (Source: {source_tag}, n_hist={n_history_for_log}) ---")
        start_time = time.time()

        # --- Prepare Inputs ---
        # Reserve space for generated tokens within max_length
        max_prompt_len = self.max_length - max_new_tokens
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            # truncation=True,
            # max_length=max_prompt_len
        ).to(self.device) # Move inputs to the same device as the model

        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]
        if self.verbose: print(f"  Tokenized prompt length: {prompt_len}")
        if prompt_len == 0:
             print("  Warning: Tokenized prompt is empty.")
             # Return empty response structure?
             return {'choices': [], 'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}, np.nan, 0


        # --- Extract Attention from Prompt Processing ---
        entropy = np.nan
        seq_len = prompt_len
        if log_attention_data:
            try:
                # Get hidden states and attentions for the prompt
                if self.verbose: print("  Running forward pass for attention...")
                outputs_prompt = self.model(**inputs, output_attentions=True)
                attentions_prompt = outputs_prompt.attentions
                if self.verbose: print(f"  Extracted {len(attentions_prompt)} layers of attentions.")

                if attentions_prompt and self.layer_to_analyze < len(attentions_prompt):
                    # (batch_size, num_heads, sequence_length, sequence_length)
                    layer_attention = attentions_prompt[self.layer_to_analyze]
                    # Average over heads: (batch_size, sequence_length, sequence_length)
                    avg_head_attention = layer_attention.mean(dim=1).squeeze(0) # Remove batch

                    # Get attention FROM last token of prompt TO all previous tokens
                    last_token_attention_dist = avg_head_attention[-1, :]
                    entropy = self._calculate_entropy(last_token_attention_dist)
                    if self.verbose: print(f"  Calculated entropy: {entropy:.4f}")

                    # Store result for analysis
                    log_entry = {
                        'n_history': n_history_for_log if n_history_for_log is not None else -1,
                        'entropy': entropy,
                        'seq_len': seq_len,
                        'source': source_tag if source_tag else 'Unknown'
                    }
                    # self.attention_results.append(log_entry)
                    if self.verbose: print(f"  Logged attention data: {log_entry}")

                else:
                     if self.verbose: print(f"  Could not get attentions or invalid layer index ({self.layer_to_analyze})")

            except Exception as e:
                print(f"  Error calculating/logging attention entropy: {e}")
                # Continue to text generation even if attention logging fails

        # --- Generate Text Completion(s) ---
        generated_texts = []
        total_completion_tokens = 0
        if self.verbose: print(f"  Generating {n_gens} completion(s)...")
        try:
            # Ensure generation parameters are valid
            gen_temp = temperature if temperature > 0 else 1.0
            gen_top_p = top_p if temperature > 0 else 1.0
            do_sample_flag = temperature > 0

            # Note: transformers generate() handles multiple sequences via num_return_sequences
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=gen_temp,
                top_p=gen_top_p,
                do_sample=do_sample_flag,
                num_return_sequences=n_gens,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Decode generated part only
            # generated_ids shape: (num_return_sequences, full_length)
            generated_texts = self.tokenizer.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)
            total_completion_tokens = sum(len(ids) for ids in generated_ids[:, prompt_len:]) # Approx token count
            if self.verbose: 
                print(f"  Generated texts: {generated_texts}")


        except Exception as e:
            print(f"  Error during text generation: {e}")
            # Fallback: return empty choices
            generated_texts = [""] * n_gens # Provide empty strings if generation failed

        # --- Mimic OpenAI Response Structure ---
        choices = [{'message': {'content': text}} for text in generated_texts]
        total_tokens = prompt_len * n_gens + total_completion_tokens # Rough estimate

        mock_openai_response = {
            'choices': choices,
            'usage': {
                'prompt_tokens': prompt_len * n_gens, # Rough estimate
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens
            }
        }

        end_time = time.time()
        if self.verbose: print(f"--- Adapter Generate Call finished in {end_time - start_time:.2f}s ---")

        # Return the mocked response AND the calculated entropy/seq_len for the prompt
        # returning the avg_head_attention to observe the dispersion 



        return mock_openai_response, entropy, seq_len,  avg_head_attention

    # def get_attention_results(self):
    #     """Returns the collected attention analysis results."""
    #     return self.attention_results

    # def reset_attention_results(self):
    #     """Clears the stored attention results."""
    #     self.attention_results = []