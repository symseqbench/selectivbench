import torch
import numpy as np

def get_num_heads(hidden_size, max_head_size=256):
    #if hidden_size <= 1024:
    if (hidden_size / max_head_size) <= 4:
        return 4
    # Check powers of two starting from 4 up to hidden_size
    num_heads = 4
    while num_heads <= hidden_size:
        if hidden_size % num_heads == 0:
            head_dim = hidden_size // num_heads
            if head_dim <= max_head_size:
                return num_heads
        num_heads *= 2
    raise ValueError(f"No valid number of heads found for hidden size {hidden_size}")


def plot_cumulative_attention_bands(att_matrix, ax=None):
    """
    For a square L×L attention matrix, compute and plot the cumulative fraction
    of all below-diagonal attention (row >= col), arranged by distance from the
    diagonal (largest distance → 0).
    
    Args:
        att_matrix:  L×L numpy array or torch.Tensor
        ax:          optional matplotlib Axes to draw into
    
    Returns:
        ax: the Axes with the plot
    """
    # to numpy
    A = att_matrix.to(torch.float32).detach().cpu().numpy() if hasattr(att_matrix, "detach") else np.array(att_matrix)
    L = A.shape[0]
    if A.shape[1] != L:
        raise ValueError(f"Expected square matrix, got {A.shape}")
    
    # build row/col index arrays
    rows, cols = np.indices((L, L))
    mask = rows >= cols               # below-diagonal (incl. diag)
    dist = rows - cols                # distance from diagonal
    
    # extract masked values
    vals = A[mask]
    dists = dist[mask]
    
    # unique distances sorted descending (farthest first)
    unique_d = np.sort(np.unique(dists))[::-1]

    # 5) cumulative sum over bands
    total = vals.sum(dtype=np.float32)
    run = 0.0
    cum_frac = []
    for d in unique_d:
        run += vals[dists == d].sum(dtype=np.float32)
        cum_frac.append(run / total)

    return np.array(cum_frac)

def seq_classification_acc(logits, target_seqs, mask, num_classes, regression=False):

    if regression:
        mask_bool = mask.type(torch.bool)
        smax_logits = torch.softmax(logits, dim=-1)
        masked_logits = smax_logits[mask_bool]
        masked_targets = target_seqs[mask_bool]

        #masked_acc_per = torch.argmax(masked_logits, dim=-1, keepdim=True)==torch.argmax(masked_targets, dim=-1, keepdim=True)
        masked_acc_per = torch.argmax(masked_logits, dim=-1, keepdim=True)==torch.argmax(masked_targets, dim=-1, keepdim=True)

        acc = torch.mean(masked_acc_per.float())

    else:

        smax_logits = torch.softmax(logits, dim=-1)
        logit_sum_per_element = torch.einsum('blc,bel->bec', smax_logits, mask)
        pred_sequence = torch.argmax(logit_sum_per_element, dim=-1)

        trgs_seqs = torch.nn.functional.one_hot(target_seqs, num_classes=num_classes)

        trgs_seqs = torch.einsum('blc, bel->bec', trgs_seqs.float(), mask)
        trgs_seqs = torch.argmax(trgs_seqs, dim=-1)

        correct_preds = pred_sequence==trgs_seqs

        acc = torch.mean(correct_preds.float())

    return acc


def seq_classification_loss(logits, target_seqs, mask, loss_fn):
    # Assuming logits is of shape (batch_size, seq_len, num_classes)

    # Have to convert the mask to bool
    mask_bool = mask.type(torch.bool)
   
    # Calculate the loss per element in the sequence
    loss = loss_fn(logits[mask_bool], 
                   target_seqs[mask_bool])

    return loss


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
