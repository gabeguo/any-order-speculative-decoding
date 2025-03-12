import numpy as np

class ContextNGram():
    def __init__(self, seqlen, mask_token=6):
        self.unigrams = dict()
        self.bigrams = dict()

        self.total_tokens = 0

        self.seqlen = seqlen
        self.seen_unigram_at_idx = [False] * seqlen
        self.seen_bigram_starting_at_idx = [False] * seqlen

        self.mask_token = mask_token

        return
    
    def update(self, tokens, starting_idx):
        assert isinstance(tokens, tuple)
        assert len(tokens) in (1, 2)
        for token in tokens:
            assert isinstance(token, int)
        # log bigram IF it's a valid bigram at an index that hasn't been logged yet
        if len(tokens) == 2 \
                and self.mask_token not in tokens \
                and not self.seen_bigram_starting_at_idx[starting_idx]:
            # log the bigram in tree style
            if tokens[0] not in self.bigrams:
                self.bigrams[tokens[0]] = dict() # maps first token to possible completions
            if tokens[1] not in self.bigrams[tokens[0]]:
                self.bigrams[tokens[0]][tokens[1]] = 0
            self.bigrams[tokens[0]][tokens[1]] += 1
            # mark that we've logged this bigram at this index, so we don't re-log it
            self.seen_bigram_starting_at_idx[starting_idx] = True
        # log the unigrams
        for offset, token in enumerate(tokens):
            # track the index of the unigram
            idx = starting_idx + offset
            # only log it IF it's visible and at an index that hasn't been logged yet
            if token != self.mask_token \
                    and not self.seen_unigram_at_idx[idx]:
                # log the unigram
                if token not in self.unigrams:
                    self.unigrams[token] = 0
                self.unigrams[token] += 1
                # mark that we've logged this unigram at this index, so we don't re-log it
                self.seen_unigram_at_idx[idx] = True
                # add to the total number of tokens
                self.total_tokens += 1 

        return
    
    def sample(self, token):
        assert isinstance(token, int)

        if token in self.bigrams:
            keys = list(self.bigrams[token].keys())
            probs = [self.bigrams[token][key] for key in keys]
            probs = [p / sum(probs) for p in probs]
            assert np.isclose(sum(probs), 1.0)
        else:
            keys = list(self.unigrams.keys())
            probs = [self.unigrams[key] for key in keys]
            probs = [p / sum(probs) for p in probs]
            assert np.isclose(sum(probs), 1.0)
        
        pre_idx = np.random.choice(len(keys), p=probs)
        the_key = keys[pre_idx]
        the_prob = probs[pre_idx]

        return {
            "sampled_token": the_key,
            "sampled_token_prob": the_prob,
            "possible_tokens": keys,
            "possible_token_probs": probs
        }



        
