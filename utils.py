import torch
import math
from collections import Counter

def translate_sentence(model, sentence, german_vocab, english_vocab, device, max_length=50):
    from main import tokenize_ger
    
    model.eval()
    
    tokens = [german_vocab["<sos>"]] + [german_vocab[token] for token in tokenize_ger(sentence)] + [german_vocab["<eos>"]]
    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)
    
    with torch.no_grad():
        encoder_states, hidden, cell = model.encoder(src_tensor)
    
    trg_idx = [english_vocab["<sos>"]]
    
    for _ in range(max_length):
        prev_word = torch.LongTensor([trg_idx[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(prev_word, encoder_states, hidden, cell)
            best_guess = output.argmax(1).item()
        
        trg_idx.append(best_guess)
        
        if best_guess == english_vocab["<eos>"]:
            break
    
    translated_sentence = [english_vocab.lookup_tokens([idx])[0] for idx in trg_idx]
    return translated_sentence[1:-1]

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def bleu(data, model, german_vocab, english_vocab, device):
    from main import tokenize_eng
    
    targets = []
    outputs = []
    
    for example in data:
        src_text, trg_text = example
        
        target = [token for token in tokenize_eng(trg_text)]
        targets.append([target])
        
        output = translate_sentence(model, src_text, german_vocab, english_vocab, device)
        outputs.append(output)
    
    return calculate_bleu(outputs, targets)

def calculate_bleu(outputs, references, max_n=4, weights=None):
    """
    Calculate BLEU score without using torchtext.
    
    Args:
        outputs: List of output/candidate sentences
        references: List of references for each sentence (can be multiple references)
        max_n: Maximum n-gram size to consider
        weights: Weights for each n-gram precision (defaults to uniform weights)
        
    Returns:
        BLEU score as float
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n
        
    bp = brevity_penalty(outputs, references)
    precisions = [modified_precision(outputs, references, i) for i in range(1, max_n + 1)]
    
    weighted_prec = sum(w * math.log(p) if p > 0 else float('-inf') for w, p in zip(weights, precisions))
    
    if min(precisions) > 0:
        bleu_score = bp * math.exp(weighted_prec)
    else:
        bleu_score = 0
        
    return bleu_score

def brevity_penalty(outputs, references):
    """
    Calculate brevity penalty for BLEU.
    
    Args:
        outputs: List of output sentences
        references: List of lists of reference sentences
        
    Returns:
        Brevity penalty factor
    """
    output_length = sum(len(output) for output in outputs)
    reference_length = 0
    
    for i, output in enumerate(outputs):
        ref_lens = [len(ref) for ref in references[i]]
        closest_ref_len = min(ref_lens, key=lambda x: abs(len(output) - x))
        reference_length += closest_ref_len
    
    if output_length > reference_length:
        return 1
    return math.exp(1 - reference_length / output_length if output_length > 0 else float('-inf'))

def modified_precision(outputs, references, n):
    """
    Calculate modified n-gram precision for BLEU.
    
    Args:
        outputs: List of output sentences
        references: List of lists of reference sentences
        n: n-gram size
        
    Returns:
        Modified precision for n-grams
    """
    total_ngrams = 0
    matched_ngrams = 0
    
    for i, output in enumerate(outputs):
        output_ngrams = Counter(get_ngrams(output, n))
        total_ngrams += sum(output_ngrams.values())
        
        max_ref_counts = Counter()
        for ref in references[i]:
            ref_ngrams = Counter(get_ngrams(ref, n))
            for ngram, count in ref_ngrams.items():
                max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)
        
        for ngram, count in output_ngrams.items():
            matched_ngrams += min(count, max_ref_counts.get(ngram, 0))
    
    return matched_ngrams / total_ngrams if total_ngrams > 0 else 0

def get_ngrams(tokens, n):
    """
    Get n-grams from a list of tokens.
    
    Args:
        tokens: List of tokens
        n: n-gram size
        
    Returns:
        List of n-grams (as tuples)
    """
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]