import torch 
import torch.nn.functional as F
import numpy as np

def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )

    evidence = relu_evidence(output)
    alpha = evidence + 1
    #uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)

    revisedY = target.clone()
    revisedY[revisedY > 0] = 1
    revisedY = revisedY * prob + target
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1), 1).transpose(0, 1)

    new_target = revisedY
    return loss, new_target


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss_A(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )

    evidence = relu_evidence(output)
    alpha = evidence + 1
    uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
    _, preds = torch.max(output, 1)

    revisedY = target.clone()
    revisedY[revisedY > 0] = 1

    conflict_scores = compute_conflict_scores(output, evidence, revisedY, 3, device)

    #super_loss = -torch.mean(torch.sum(torch.log(1.0000001 - F.softmax(output, dim=1)) * (1 - revisedY), dim=1))
    S = torch.sum(alpha, dim=1, keepdim=True)  # (B,1)
    S1 = S + 1
    non_cand_mask = 1 - revisedY

    # First order: E[p_j] = alpha_j / S
    term1 = alpha / S

    # Second order: E[p_j^2]
    term2 = 0.5 * (alpha * (alpha + 1)) / (S * S1)

    # Only penalize non-candidate
    super_loss = (term1 + term2) * non_cand_mask
    #super_loss.sum(dim=1).mean()
    super_loss = torch.mean(super_loss)

    ########################
    revisedY = revisedY * uncertainty
    if epoch_num > 100:
        indices = torch.where(uncertainty > 0.5)[0]
        for i in range(len(indices)):
            index = indices[i]
            revisedY[index][preds[index]] = 1 - uncertainty[index]

    # indices = torch.where(uncertainty > 0.5)[0]
    #
    # for i in range(len(indices)):
    #     index = indices[i]
    #     revisedY[index][preds[index]] = 1 - uncertainty[index]


    #########################

    revisedY = 0.05*revisedY * prob + target
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1), 1).transpose(0, 1)

    new_target = revisedY
    return loss + conflict_scores + super_loss,new_target




def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step,
    device=None, lambda_super=1.0, lambda_conflict=1.0
):
    if not device:
        device = output.device

    if epoch_num > 100:
        lambda_super = 0.8
        lamdda_dis = 0.2
    else:
        lambda_super = 0.8
        lamdda_dis = 0.2


    # Compute evidence and Dirichlet parameters
    evidence = relu_evidence(output)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)  # Dirichlet strength
    S1 = S + 1

    # Compute EDL loss (e.g., digamma based negative log likelihood)
    loss = lamdda_dis * torch.mean(
        edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    )

    # Compute uncertainty and predicted class (from alpha)
    uncertainty = num_classes / S


    prob = alpha / S
    _, preds = torch.max(prob, dim=1)  # Use alpha instead of raw output

    # Construct revisedY as binary mask of candidates
    revisedY = target.clone()
    revisedY[revisedY > 0] = 1  # multi-hot candidates
    non_cand_mask = 1 - revisedY  # used for super_loss

    # Compute conflict regularization
    conflict_scores = lambda_conflict * compute_conflict_scores(output, evidence, revisedY, 3, device)



    # Super loss: penalize non-candidate classes
    term1 = alpha / S
    term2 = 0.5 * (alpha * (alpha + 1)) / (S * S1)
    super_penalty = (term1 + term2) * non_cand_mask
    super_loss = lambda_super * torch.mean(super_penalty)



    loss_all = loss + 0.5*conflict_scores + 0.8*super_loss


    # Combine revisedY and prediction prob
    revisedY = 0.1*revisedY * F.softmax(prob, dim=1) + target
    revisedY = revisedY / (revisedY.sum(dim=1, keepdim=True) + 1e-8)  # normalize per sample


    # Min-Max normalization to scale revisedY to [0, 1]
    min_val = revisedY.min(dim=1, keepdim=True)[0]
    max_val = revisedY.max(dim=1, keepdim=True)[0]

    revisedY = (revisedY - min_val) / (max_val - min_val + 1e-8)  # avoid division by zero


    return loss_all, revisedY



def compute_conflict_scores(probs, evidence, partial_label, top_k=3, device=None):
    if device is None:
        device = probs.device

    B, K = probs.shape
    _, preds = torch.max(probs, 1)
    alpha = evidence + 1
    conflict_scores = torch.zeros(1, device=device)

    for v in range(K):
        indices = torch.where(preds == v)[0]
        n = len(indices)
        if n <= 1:
            continue

        alpha_v = alpha[indices]
        S = torch.sum(alpha_v, dim=1, keepdim=True)
        p = alpha_v / S

        # 1. 距离矩阵（预测分布）
        dist_matrix = torch.cdist(p, p, p=2).pow(2)  # [n, n]

        # 2. 不确定性矩阵
        confidence = (K / S).squeeze()
        conf_outer = (1 - confidence).unsqueeze(1) * (1 - confidence).unsqueeze(0)  # [n, n]

        # 3. Jacard 相似矩阵
        A = partial_label[indices].float()
        n11 = torch.matmul(A, A.t())
        n00 = torch.matmul(1 - A, (1 - A).t())
        Jac_matrix = 1 - (n11 / (K - n00 + 1e-8))

        # 4. 冲突矩阵
        conflict_matrix = dist_matrix * conf_outer * Jac_matrix

        # 5. Top-k 平均冲突得分
        k_val = min(top_k + 1, n)
        _, topk_indices = dist_matrix.topk(k=k_val, dim=1, largest=False)
        selected_conflict = conflict_matrix.gather(1, topk_indices)[:, 1:]
        group_conflict = selected_conflict.mean()

        conflict_scores += group_conflict

    return conflict_scores / K





















def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


# def rotate_img(x, deg):
#     return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()


def compute_conflict_scores_a(probs, evidence, partial_label, top_k=3, device=None):
    if device is None:
        device = probs.device

    B, K = probs.shape
    _, preds = torch.max(probs, 1)
    alpha = evidence + 1
    conflict_scores = torch.tensor(0.0, device=device)

    for v in range(K):
        indices = torch.where(preds == v)[0]
        n = len(indices)
        if n == 0 or n == 1:
            continue

        alpha_v = alpha[indices]
        S = torch.sum(alpha_v, dim=1, keepdim=True)
        p = alpha_v / S
        # --- Step 1: Pairwise squared L2 distances ---
        dist_matrix = torch.cdist(p, p, p=2).pow(2)  # [B, B]
        # log_p = torch.log(probs + 1e-8)
        # dist_matrix = F.kl_div(log_p.expand(B, B, K), probs.unsqueeze(1), reduction='none').sum(dim=2)

        # --- Step 2: Pairwise confidence outer product ---
        confidence = torch.squeeze(K / S)
        conf_outer = torch.ger(1-confidence, 1-confidence)  # [B, B]

        # --- Step 3: similarity matrix ---

        # Jac_matrix = torch.zeros((len(indices), len(indices))).to(device)
        #
        # for i in range(len(indices)):
        #     for j in range(len(indices)):
        #         Jac_matrix[i, j] = 1
        #         n11 = sum((partial_label[indices[i]].int() == 1) & (partial_label[indices[j]].int() == 1))
        #         n00 = sum((partial_label[indices[i]].int() == 0) & (partial_label[indices[i]].int() == 0))
        #         jac = n11 / (K - n00)
        #         if jac != 0:
        #             Jac_matrix[i, j] = 1

        A = partial_label[indices].float()
        n11 = torch.matmul(A, A.t())
        n00 = torch.matmul(1 - A, (1 - A).t())
        Jac_matrix = n11 / (K - n00 + 1e-8)

        # --- Step 4: Conflict matrix ---
        conflict_matrix = dist_matrix * conf_outer * Jac_matrix # [B, B]

        # --- Step 5: Average top-k conflict scores ---
        # k = min(top_k + 1, B)  # 动态调整k值
        #
        # _, indices = dist_matrix.topk(k=k + 1, dim=1, largest=False)
        # topk_conflict = conflict_matrix.gather(1, indices)[:, 1:]
        #
        # conflict_scores += topk_conflict.mean()  # [B]

        if n == 1:
            group_conflict = torch.tensor(0.0, device=device)
        else:
            k_val = min(top_k + 1, n)
            _, topk_indices = dist_matrix.topk(k=k_val, dim=1, largest=False)
            selected_conflict = conflict_matrix.gather(1, topk_indices)[:, 1:]
            group_conflict = selected_conflict.mean()

        conflict_scores += group_conflict

    return conflict_scores/K


def knn(query, data, k=10):
    # 确保输入是浮点类型
    query = query.float()
    data = data.float()

    assert data.shape[1] == query.shape[1]

    # 分批计算距离矩阵
    batch_size = 1024 # 根据显存调整
    n = query.size(0)
    indices = []

    for i in range(0, n, batch_size):
        batch_query = query[i:i + batch_size]
        dist = torch.cdist(batch_query, data)
        k_val = min(k, data.size(0))
        _, batch_ind = dist.topk(k_val, dim=1, largest=False)
        indices.append(batch_ind)

    return torch.cat(indices, dim=0)


def knn_to_partial_labels(query_embd, y_onehot, k=10, n_class=10):
    """显存优化的部分标签生成"""
    device = query_embd.device
    y_onehot = y_onehot.to(device)



    n_sample = query_embd.size(0)

    # 获取k近邻索引 (n_sample, k)
    neighbour_ind = knn(query_embd, query_embd, k=k)


    # 分批聚合邻居标签
    batch_size = 2048  # 根据显存调整
    partial_labels = torch.zeros(n_sample, n_class, device=device)

    for i in range(0, n_sample, batch_size):
        end_idx = min(i + batch_size, n_sample)
        batch_size_actual = end_idx - i

        # 获取当前批次的邻居索引 (batch_size, k)
        batch_neighbours = neighbour_ind[i:end_idx]

        # 收集邻居标签 (batch_size, k, n_class)
        neighbour_labels = y_onehot[batch_neighbours]

        # 聚合邻居标签 (batch_size, n_class)
        aggregated = torch.sum(neighbour_labels, dim=1)

        # 与自身标签相乘 (batch_size, n_class)
        batch_labels = y_onehot[i:end_idx]
        batch_partial = aggregated * batch_labels

        partial_labels[i:end_idx] = batch_partial

    partial_labels = partial_labels / (partial_labels.sum(dim=1, keepdim=True) + 1e-8)  # normalize per sample

    return partial_labels



def knn_to_partial_labels_A(query_embd, y_query, k=10, n_class=10):
    # 确保标签是整数张量
    if not isinstance(y_query, torch.Tensor):
        y_query = torch.tensor(y_query, dtype=torch.long)

    n_sample = query_embd.size(0)

    # 获取k近邻索引 (n_sample, k)
    neighbour_ind = knn(query_embd, query_embd, k=k)

    # 创建邻接矩阵的稀疏表示
    row_idx = torch.arange(n_sample, device=query_embd.device).unsqueeze(1).expand(-1, k)
    adj_matrix = torch.zeros((n_sample, n_sample),
                             dtype=torch.float,
                             device=query_embd.device)
    adj_matrix[row_idx.flatten(), neighbour_ind.flatten()] = 1.0

    # 转换标签为one-hot编码 (n_sample, n_class)
    y_onehot = torch.nn.functional.one_hot(y_query, num_classes=n_class).float()

    # 计算部分标签: P = (A · Y) ⊙ Y_onehot
    neighbor_labels = torch.mm(adj_matrix, y_onehot)
    partial_labels = neighbor_labels * y_onehot

    return partial_labels

