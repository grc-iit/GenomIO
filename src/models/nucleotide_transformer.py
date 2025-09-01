import sys
sys.path.append("./LLMs/nucleotide-transformer")

import os
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from nucleotide_transformer.pretrained import get_pretrained_model
from sklearn.metrics import accuracy_score

def load_sequences(file_path):
    """Load context and target sequences from file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    context_index = lines.index("CONTEXT:\n") + 1
    target_index = lines.index("TARGET:\n") + 1
    
    context = "".join(lines[context_index:target_index-1]).replace("\n", "")
    target = "".join(lines[target_index:]).replace("\n", "")
    
    return context, target

def generate_predictions(context, tokenizer, model_fn, params, n_percentage=0):
    """Generate prediction of model to complete sequence."""
    # Add 'N' after context
    num_n_to_add = int(len(context) * n_percentage)  # Calculate the n_percentage of len(context) = len(target)
    context = context + 'N' * num_n_to_add  
    #print("context: ", context)

    # Tokenize context 
    tokens_text = tokenizer.tokenize(context)[0]
    #print("tokens:", tokens_text)
    tokens_ids = tokenizer.tokenize(context)[1]
    #print("tokens ids: ", tokens_ids)

    # Compute tokens as an array of token ids
    tokens = jnp.asarray([tokens_ids], dtype=jnp.int32)

    predicted_tokens = []
    random_key = jax.random.PRNGKey(0)

    output = model_fn.apply(params, random_key, tokens)
    #print(output.keys())

    embeddings = output["embeddings_20"][:, 1:, :]  # removing CLS token
    padding_mask = jnp.expand_dims(tokens[:, 1:] != tokenizer.pad_token_id, axis=-1)
    masked_embeddings = embeddings * padding_mask  # multiply by 0 pad tokens embeddings
    sequences_lengths = jnp.sum(padding_mask, axis=1)
    mean_embeddings = jnp.sum(masked_embeddings, axis=1) / sequences_lengths

    logits = output['logits']

    probabilities = []

    # get probabilities separately for each sequence. In our case, context has only one sequence
    for seq_id in range(logits.shape[0]):

        logits_seq = logits[seq_id]
        seq_length = int(sequences_lengths[seq_id][0])

        logits_seq = logits_seq[1 : (seq_length + 1)]  # remove CLS token and pads
        logits_seq = logits_seq - jnp.max(logits_seq, axis=-1, keepdims=True) # Center logits to avoid big values

        probas = jax.nn.softmax(
            logits_seq, axis=-1
        )  # use softmax to transform logits into probabilities


        probabilities.append(probas)
    for pos in  range(num_n_to_add):
        sequence_id = 0
        position_id = int(sequences_lengths[seq_id][0]) - num_n_to_add - pos # Last position in the sequence

        probs = probabilities[sequence_id][position_id]  # Probabilities at the last token's position

        sorted_positions = jnp.argsort(-probs)  # Sort the probabilities in descending order
        sorted_probs = probs[sorted_positions]  # Get sorted probabilities

        # For visualization
        '''
        top_k = 3
        top_probs = []
        top_tokens = []
        
        for k in range(top_k):
            predicted_token = tokenizer.id_to_token(int(sorted_positions[k]))  # Get token from id
            prob = sorted_probs[k]  # Probability of the token
            top_probs.append(prob)
            top_tokens.append(predicted_token)
            print(f"token: {predicted_token}, probability: {prob * 100:.2f}%")
        '''
        max_predicted_token = tokenizer.id_to_token(int(sorted_positions[0]))  # Get max token from id
        #print("max predicted token: ", max_predicted_token)
        predicted_tokens.append(max_predicted_token)

    #print("Predicted tokens length: ", len(predicted_tokens))
    #print("Predicted tokens: ", predicted_tokens)

    return predicted_tokens

def evaluate_predictions(predicted_seq, target_seq, n):
    """Calculate accuracy comparing predicted sequence with real sequence."""
    pred_string = ''.join(predicted_seq)[:n]
    target_string = target_seq[:n]
    
    y_pred = np.array(list(pred_string))
    y_true = np.array(list(target_string))
    
    accuracy = accuracy_score(y_true, y_pred)
    
    return accuracy

def main(test_data_folder):
    """Load model and evaluate with test files."""
    
    params, forward_fn, tokenizer, config = get_pretrained_model(
    model_name="250M_multi_species_v2",
    embeddings_layers_to_save=(20,)
    )
    forward_fn = hk.transform(forward_fn)

    results = []
    processed_files = 0
    for file_name in os.listdir(test_data_folder):
        if file_name.endswith(".txt"):
            if processed_files >= 10:
                break
            file_path = os.path.join(test_data_folder, file_name)
            context, target = load_sequences(file_path)
            perc = [0.1, 0.25, 0.5, 0.75, 1]
            prediction = generate_predictions(context, tokenizer, forward_fn, params, 0.2)
            for p in perc:
                n = int(len(context) * p)
                accuracy = evaluate_predictions(prediction, target, n)
                results.append((file_name, p, accuracy))

        processed_files +=1
    
    for file_name, p, accuracy in results:
        print(f"Results for {file_name} at {p*100:.0f}%: \n accuracy: {accuracy}")
# Dataset folder, change as needed
test_folder = "./data/test_sequences/1000bp"
test_results = main(test_folder)
