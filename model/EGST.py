import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.STutil import sample_and_group_multi, square_distance, angle_difference
import numpy as np
from model.StructureTransformer import Transformer

_EPS = 1e-5


def get_prepool(in_dim, out_dim):
    net = nn.Sequential(
        nn.Conv2d(in_dim, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),
    )
    return net


def get_postpool(in_dim, out_dim):
    net = nn.Sequential(
        nn.Conv1d(in_dim, in_dim, 1),
        nn.GroupNorm(8, in_dim),
        nn.ReLU(),
        nn.Conv1d(in_dim, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),
        nn.Conv1d(out_dim, out_dim, 1),
    )
    return net


def match_features(feat_src, feat_ref, metric='l2'):
    """ Compute pairwise distance between features

    Args:
        feat_src: (B, J, C)
        feat_ref: (B, K, C)
        metric: either 'angle' or 'l2' (squared euclidean)

    Returns:
        Matching matrix (B, J, K). i'th row describes how well the i'th point
         in the src agrees with every point in the ref.
    """
    assert feat_src.shape[-1] == feat_ref.shape[-1]

    if metric == 'l2':
        dist_matrix = square_distance(feat_src, feat_ref)
    elif metric == 'angle':
        feat_src_norm = feat_src / (torch.norm(feat_src, dim=-1, keepdim=True) + _EPS)
        feat_ref_norm = feat_ref / (torch.norm(feat_ref, dim=-1, keepdim=True) + _EPS)

        dist_matrix = angle_difference(feat_src_norm, feat_ref_norm)
    else:
        raise NotImplementedError

    return dist_matrix


def sinkhorn(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha


def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): (B, M, 3) points
        b (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)

    Returns:
        Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
    """

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
    centroid_a = torch.sum(a * weights_normalized, dim=1)
    centroid_b = torch.sum(b * weights_normalized, dim=1)
    a_centered = a - centroid_a[:, None, :]
    b_centered = b - centroid_b[:, None, :]


    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    H = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)
    R = []
    for i in range(a.size(0)):
        u, s, v = torch.svd(H[i])
        r = torch.matmul(v, u.transpose(1, 0)).contiguous()
        r_det = torch.det(r).item()
        diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                          [0, 1.0, 0],
                                          [0, 0, r_det]]).astype('float32')).to(v.device)
        r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
        R.append(r)

    R = torch.stack(R, dim=0).cuda()
    translation = -R @ centroid_a[:, :, None] + centroid_b[:, :, None]
    transform = torch.cat((R, translation), dim=2)
    return transform


def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise NotImplementedError


def se3_transform(g, a, normals=None):
    """ Applies the SE3 transform

    Args:
        g: SE3 transformation matrix of size ([1,] 3/4, 4) or (B, 3/4, 4)
        a: Points to be transformed (N, 3) or (B, N, 3)
        normals: (Optional). If provided, normals will be transformed

    Returns:
        transformed points of size (N, 3) or (B, N, 3)

    """
    R = g[..., :3, :3]  # (B, 3, 3)
    p = g[..., :3, 3]  # (B, 3)

    if len(g.size()) == len(a.size()):
        b = torch.matmul(a, R.transpose(-1, -2)) + p[..., None, :]
    else:
        raise NotImplementedError
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p  # No batch. Not checked

    if normals is not None:
        rotated_normals = normals @ R.transpose(-1, -2)
        return b, rotated_normals

    else:
        return b


def convert2transformation(rotation_matrix: torch.Tensor, translation_vector: torch.Tensor):
    one_ = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(rotation_matrix.shape[0], 1, 1).to(rotation_matrix)  # (Bx1x4)
    transformation_matrix = torch.cat([rotation_matrix, translation_vector[:, 0, :].unsqueeze(-1)], dim=2)  # (Bx3x4)
    transformation_matrix = torch.cat([transformation_matrix, one_], dim=1)  # (Bx4x4)
    return transformation_matrix


class Feature_Embeeding(nn.Module):
    def __init__(self, emb_dims=256, radius=0.3, num_neighbors=64):
        super(Feature_Embeeding, self).__init__()
        self.radius = radius
        self.n_sample = num_neighbors
        self.features = ['xyz', 'dxyz', 'ppfg']
        raw_dim = 11 
        self.prepool = get_prepool(raw_dim, emb_dims * 2)
        self.postpool = get_postpool(emb_dims * 2, emb_dims)  

    def forward(self, xyz, normals):
        features = sample_and_group_multi(-1, self.radius, self.n_sample, xyz, normals)
        features['xyz'] = features['xyz'][:, :, None, :]  # (B,N,1,3)
        concat = []
        for i in range(len(self.features)):
            f = self.features[i]
            expanded = (features[f]).expand(-1, -1, self.n_sample, -1)
            concat.append(expanded)
        fused_input_feat = torch.cat(concat, -1)

        new_feat = fused_input_feat.permute(0, 3, 2, 1)  # [B, 11, n_sample, N]
        new_feat = self.prepool(new_feat)
        pooled_feat = torch.max(new_feat, 2)[0]  # Max pooling (B, C, N)
        post_feat = self.postpool(pooled_feat)  # Post pooling dense layers
        cluster_feat = post_feat.permute(0, 2, 1)
        
        return cluster_feat  # (B, N, C)


class Feature_Extraction(nn.Module):
    def __init__(self):
        super(Feature_Extraction, self).__init__()
        self.embedding = Feature_Embeeding(emb_dims=128, radius=0.3, num_neighbors=64)
        self.feature = Transformer(d_model=128, h=4)

    def forward(self, template, source):
        template_emb = self.embedding(template[..., :3], template[..., 3:])
        source_emb = self.embedding(source[..., :3], source[..., 3:])
        template_trasformer,template_attention_score = self.feature(template_emb, source_emb)
        source_transfromer ,source_attention_score= self.feature(source_emb, template_emb)
        return template_trasformer, source_transfromer,template_attention_score,source_attention_score


class ParameterPredictionNet(nn.Module):
    def __init__(self, weights_dim):
        super().__init__()
        self.weights_dim = weights_dim

        self.prepool = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.Linear(256, 2 + np.prod(weights_dim)),
        )

    def forward(self, x):
        src_padded = F.pad(x[0], (0, 1), mode='constant', value=0)
        ref_padded = F.pad(x[1], (0, 1), mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        beta = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])

        return beta, alpha


class EGST(nn.Module):
    def __init__(self):
        super(EGST, self).__init__()
        self.weights_net = ParameterPredictionNet(weights_dim=[0])
        self.feat_extractor = Feature_Extraction()

        self.add_slack = True
        self.num_sk_iter = 5

    def compute_affinity(self, beta, feat_distance, alpha=0.5):
        """Compute alogarithm of Initial match matrix values, i.e. log(m_jk)"""
        if isinstance(alpha, float):
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha)
        else:
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha[:, None, None])
        return hybrid_affinity

    def spam(self, template, source):
        xyz_template, norm_template = template[..., :3], template[..., 3:]
        xyz_source, norm_source = source[..., :3], source[..., 3:]
        self.beta, self.alpha = self.weights_net([xyz_source, xyz_template])
        self.feat_template, self.feat_source = self.feat_extractor(template, source)
        feat_distance = match_features(self.feat_source, self.feat_template)
        self.affinity = self.compute_affinity(self.beta, feat_distance, alpha=self.alpha)

        # Compute weighted coordinates
        log_perm_matrix = sinkhorn(self.affinity, n_iters=self.num_sk_iter, slack=self.add_slack)
        self.perm_matrix = torch.exp(log_perm_matrix)
        weighted_template = self.perm_matrix @ xyz_template / (torch.sum(self.perm_matrix, dim=2, keepdim=True) + _EPS)

        return weighted_template

    def forward(self, template, source, max_iterations=2):
        transforms = []
        transforms_inverse = []
        all_gamma, all_perm_matrices, all_weighted_template = [], [], []
        all_beta, all_alpha = [], []
        xyz_source, norm_source = source[..., :3], source[..., 3:]
        xyz_template, norm_template = template[..., :3], template[..., 3:]
        xyz_source_t, norm_source_t, source_t = xyz_source, norm_source, source
        for i in range(max_iterations):

            self.beta, self.alpha = self.weights_net([xyz_source_t, xyz_template])

            self.feat_template, self.feat_source,self.template_attention_score,self.source_attention_score = self.feat_extractor(template, source_t)
            feat_distance = match_features(self.feat_source, self.feat_template)
            self.affinity = self.compute_affinity(self.beta, feat_distance, alpha=self.alpha)

            # Compute weighted coordinates
            log_perm_matrix = sinkhorn(self.affinity, n_iters=self.num_sk_iter, slack=self.add_slack)

            self.perm_matrix = torch.exp(log_perm_matrix)
            weighted_template = self.perm_matrix @ xyz_template / (
                        torch.sum(self.perm_matrix, dim=2, keepdim=True) + _EPS)

            # Compute transform and transform points
            transform = compute_rigid_transform(xyz_source, weighted_template,
                                                weights=torch.sum(self.perm_matrix, dim=2))
            xyz_source_t, norm_source_t = se3_transform(transform.detach(), xyz_source,
                                                        norm_source)  
            source_t = torch.cat([xyz_source_t, norm_source_t], dim=-1)
            est_R_inverse = transform[:, :3, :3].permute(0, 2, 1).contiguous()  
            est_t_inverse = torch.squeeze(-est_R_inverse @ transform[:, :3, 3][..., None], -1)
            est_T_inverse = convert2transformation(est_R_inverse, est_t_inverse.unsqueeze(1))
            transforms_inverse.append(est_T_inverse)

            transforms.append(transform)
            all_gamma.append(torch.exp(self.affinity))
            all_perm_matrices.append(self.perm_matrix)
            all_weighted_template.append(weighted_template)
            all_beta.append(to_numpy(self.beta))
            all_alpha.append(to_numpy(self.alpha))
        est_T = convert2transformation(transforms[max_iterations - 1][:, :3, :3],
                                       transforms[max_iterations - 1][:, :3, 3].unsqueeze(1))
        transformed_source = torch.bmm(est_T[:, :3, :3], source[:, :, :3].permute(0, 2, 1)).permute(0, 2, 1) + est_T[:,
                                                                                                               :3,
                                                                                                               3].unsqueeze(
            1)

        result = {'est_R': est_T[:, :3, :3],  
                  'est_t': est_T[:, :3, 3],  
                  'est_T': est_T, 
                  'r': self.feat_template - self.feat_source,
                  'transformed_source': transformed_source}

        result['perm_matrices_init'] = all_gamma
        result['perm_matrices'] = all_perm_matrices
        result['transforms_inverse'] = transforms_inverse
        result['weighted_template'] = all_weighted_template
        result['transforms'] = transforms
        result['temp_score']=self.template_attention_score
        result['src_score']=self.source_attention_score
        return result


