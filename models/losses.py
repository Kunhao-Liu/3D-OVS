import torch
import torch.nn.functional as F

class CorrelationLoss(torch.nn.Module):

    def __init__(self, args):
        super(CorrelationLoss,self).__init__()

        self.dino_shift = args.dino_shift
    
    def tensor_correlation(self, a, b):
        return torch.einsum("nsc,nlc->nsl", F.normalize(a, dim=-1), F.normalize(b, dim=-1))

    def l1_correlation(self, a, b):
        '''
        not in log space!!!
        a, b: [N,HW,C]
        '''
        a = a.unsqueeze(2) # [N, HW, 1, C]
        b = b.unsqueeze(1) # [N, 1, HW, C]
        l1_corr = torch.abs(a-b).sum(dim=-1) # [N,HW,HW]

        return l1_corr

    def js_correlation(self, a, b):
        '''
        note that the input probabilities are already in log space
        a, b: [N,HW,C]
        '''
        m = torch.log((a.exp() + b.exp()) / 2.) # [N,HW,C]
        m = m.unsqueeze(1) # [N,1,HW,C]
        # KL(a||m), a is the true distribution
        a = a.unsqueeze(2) # [N,HW,1,C]
        kl_pointwise_am = a.exp() * (a-m) # [N,HW,HW,C]
        kl_am = kl_pointwise_am.sum(dim=-1) # [N,HW,HW]
        # KL(b||m), b is the true distribution
        b = b.unsqueeze(2)
        kl_pointwise_bm = b.exp() * (b-m)
        kl_bm = kl_pointwise_bm.sum(dim=-1)
        
        return (kl_am + kl_bm) / 2.

    def forward(self,
                feats: torch.Tensor,
                p_class: torch.Tensor,
                ):
        '''
        feats: [N,H,W,C]
        p_class: [N,H,W,N_class]
        '''
        feats = feats.reshape(feats.size(0), -1, feats.size(-1))
        p_class = p_class.reshape(p_class.size(0), -1, p_class.size(-1))

        with torch.no_grad():
            # get dino feature correlation
            f_corr = self.tensor_correlation(feats, feats) - self.dino_shift
            f_corr_pos = f_corr.clamp(min=0)
            f_corr_neg = f_corr.clamp(max=0)

        p_corr = self.js_correlation(p_class, p_class)

        return  (f_corr_pos * p_corr).sum() / torch.count_nonzero(f_corr_pos), \
                (f_corr_neg * p_corr).sum() / torch.count_nonzero(f_corr_neg)


class OriginalCorrelationLoss(torch.nn.Module):

    def __init__(self, args=None):
        super(OriginalCorrelationLoss, self).__init__()
        self.zero_clamp = True
        self.stabalize = False
        self.pointwise = True
        self.rand_neg = False
        
        self.feature_samples = 11
        # self.self_shift = 0.18
        # self.self_weight = 0.67
        # self.neg_shift = 0.46
        # self.neg_weight = 0.63
        self.self_shift = args.self_shift
        self.self_weight = args.self_weight
        self.neg_shift = args.neg_shift
        self.neg_weight = args.neg_weight



    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2
    
    def tensor_correlation(self, a, b):
        return torch.einsum("nchw,ncij->nhwij", a, b)
    
    def norm(self, t):
        return F.normalize(t, dim=1, eps=1e-10)

    def sample(self, t: torch.Tensor, coords: torch.Tensor):
        return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)
    
    def super_perm(self, size: int, device: torch.device):
        perm = torch.randperm(size, device=device, dtype=torch.long)
        perm[perm == torch.arange(size, device=device)] += 1
        return perm % size

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = self.tensor_correlation(self.norm(f1), self.norm(f2))

            if self.pointwise: # True #
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = self.tensor_correlation(self.norm(c1), self.norm(c2))

        if self.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.stabalize:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(self,
                orig_feats: torch.Tensor,
                orig_code: torch.Tensor,
                sim_matrix: torch.Tensor
                ):

        coord_shape = [orig_feats.shape[0], self.feature_samples, self.feature_samples, 2]

        coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1 # coord_shape: [16, 11, 11, 2]  * 2 - 1  let range [-1, 1]
        coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1 # coord_shape: [16, 11, 11, 2]

        feats = self.sample(orig_feats, coords1)
        code = self.sample(orig_code, coords1)

        # find negtive pair
        if sim_matrix is None:
            neg_indx = self.super_perm(orig_feats.shape[0], orig_feats.device)
        else:
            assert len(sim_matrix.shape) == 2
            neg_indx = torch.min(sim_matrix, dim=0)[1]
        
        if self.rand_neg:
            neg_indx = torch.randperm(sim_matrix.shape[0], device=orig_feats.device, dtype=torch.long)

        neg_feats = orig_feats[neg_indx]
        neg_code = orig_code[neg_indx]
        neg_feats = self.sample(neg_feats, coords2)
        neg_code = self.sample(neg_code, coords2)

        neg_corr_loss, neg_corr_cd = self.helper(
            feats, neg_feats, code, neg_code, self.neg_shift)
        
        self_corr_loss, self_corr_cd = self.helper(
            feats, feats, code, code, self.self_shift)

        return self.neg_weight * neg_corr_loss.mean() + self.self_weight * self_corr_loss.mean()