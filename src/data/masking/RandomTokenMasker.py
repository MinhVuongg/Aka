import re
import random
from src.utils.mylogger import logger

class RandomTokenMasker:
    """
    handle random token masking in code snippet.
    Randomly masks 10%-15% of tokens with special mask tokens.
    """
    def __init__(self, mask_rate_min=0.10, mask_rate_max=0.15):
        """
        Initialize the RandomTokenMasker.

        Args:
            mask_rate_min (float): Minimum percentage of token to mask (between 0 and 1)
            mask_rate_max (float): Maximum percentage of tokens to mask (between 0 and 1)
        """
        self.mask_rate_min = mask_rate_min
        self.mask_rate_max = mask_rate_max
        self.masked_map = {}
        self.reverse_map = {}
        self.mask_counter = 0

    def reset(self):
        """Reset all masking data."""
        self.masked_map = {}
        self.reverse_map = {}
        self.mask_counter = 0

    def _generate_mask_token(self):
        """Generate a unique mask token."""
        mask = f"<MASK_{self.mask_counter}>"
        self.mask_counter += 1
        return mask

    def mask_tokens(self, code):
        """
        Randomly mask tokens in the given code at a rate of 10-15%.
        """
        if not code or not code.strip():
            return ""

        try:
            # Reset for a fresh masking session
            self.reset()

            # Tokenize the code
            # We'll use a simple regex to split code into tokens including operators,
            # punctuation, identifiers, and literals
            token_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*|\d+|\d+\.\d+|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}()\[\];,.<>:?!+\-*/%&|^~=]+|\s+)'
            tokens = re.findall(token_pattern, code)

            # Determine how many tokens to mask
            mask_rate = random.uniform(self.mask_rate_min, self.mask_rate_max)
            num_tokens_to_mask = max(1, int(len(tokens) * mask_rate))

            # Select random indices to mask
            # Don't mask whitespace tokens
            non_whitespace_indices = [i for i, token in enumerate(tokens) if not token.isspace()]
            if not non_whitespace_indices:
                return code

            indices_to_mask = random.sample(non_whitespace_indices,
                                            min(num_tokens_to_mask, len(non_whitespace_indices)))

            # Mask selected tokens
            masked_tokens = tokens.copy()
            for idx in indices_to_mask:
                token = tokens[idx]
                # Skip whitespace
                if token.isspace():
                    continue

                # Generate a mask token
                mask = self._generate_mask_token()

                # Store the mapping
                self.masked_map[token] = mask
                self.reverse_map[mask] = token

                # Replace the token with its mask
                masked_tokens[idx] = mask

            # Join tokens back into code
            masked_code = ''.join(masked_tokens)

            return masked_code

        except Exception as e:
            logger.error(f"[ERROR] mask_tokens: {e}")
            return code

    def unmask_tokens(self, masked_code):
        """
        Restore original tokens from masked code
        """
        if not masked_code or not self.reverse_map:
            return masked_code

        try:
            result = masked_code
            for mask, original in self.reverse_map.items():
                result = result.replace(mask, original)
            return result
        except Exception as e:
            logger.error(f"[ERROR] unmask_tokens: {e}")
            return masked_code

    def get_masking_statistics(self):
        """
        Get statistics about the masking operation
        """
        return {
            'total_tokens_masked': len(self.masked_map),
            'unique_tokens': len(set(self.masked_map.keys()))
        }