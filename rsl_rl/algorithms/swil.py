from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl.modules import Discriminator
from rsl_rl.storage import DemoBuffer
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.utils import ortho_layer_init
from rsl_rl.algorithms.swil_rewards_isaaclab import (
    build_swil_reward_head_from_cfg,
)

from collections import deque


class SWIL(PPO):
    discriminator: Discriminator

    def __init__(self, actor_critic, discriminator, il_opt, device, **kwargs):
        # init swil
        super().__init__(
            actor_critic,
            device=device,
            **kwargs,
        )

        # keep a reference to full IL options
        self.il_opt = il_opt

        self.il_lr = il_opt.learning_rate
        self.use_actions = il_opt.use_actions
        self.use_dones = il_opt.use_dones
        self.use_next_obs = il_opt.use_next_obs
        self.use_weight_norm = il_opt.use_weight_norm
        self.use_spectral_norm = il_opt.use_spectral_norm
        self.l2_coeff = il_opt.l2_coeff
        self.num_irl_epochs = il_opt.num_irl_epochs
        self.irl_batch_size = il_opt.irl_batch_size
        self.shuffle_atom_batches = il_opt.shuffle_atom_batches
        self.max_q_len = il_opt.max_q_len
        self.repl_loss_type = il_opt.repl_loss_type

        # SWIL specific arguments
        self.n_proj = il_opt.n_proj
        self.use_linear_proj = il_opt.use_linear_proj

        # Reward selection: LEGACY (default) or one of {SR, DUAL, RPL}
        self.swil_mode = getattr(il_opt, "swil_mode", None)
        self.reward_head = None

        # SWIL components
        self.discriminator = discriminator
        self.discriminator.to(self.device)

        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=self.il_lr, weight_decay=self.l2_coeff
        )
        self.pi_atoms_sorted = deque(maxlen=self.max_q_len)
        self.exp_atoms_sorted = deque(maxlen=self.max_q_len)

        self.rnd = torch.randn(self.n_proj, self.discriminator.input_dim).to(
            self.device
        )

    def init_storage_from_demos(
        self,
        demos,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        action_shape,
    ):
        self.demos_storage = DemoBuffer(
            num_envs,
            num_transitions_per_env,
            obs_shape[-1],
            action_shape[-1],
            self.device,
        )
        self.demos_storage.load_demos(demos)

        # Initialize optional reward head (SR/DUAL/RPL) and build expert buffers
        if self.swil_mode is not None and str(self.swil_mode).upper() in {"SR", "DUAL", "RPL"}:
            # Build reward head from config and load expert projections
            self.reward_head = build_swil_reward_head_from_cfg(
                obs_shape=(obs_shape[-1],),
                imitation_cfg=self.il_opt,
                device=self.device,
            )

            # Prepare demos dict expected by reward head: flatten across time and envs
            with torch.no_grad():
                obs = demos["obs"].to(self.device)
                acs = demos["acs"].to(self.device)
                # next observations (T, E, D)
                if "next_obs" in demos:
                    nobs = demos["next_obs"].to(self.device)
                else:
                    nobs = torch.cat((obs[1:], obs[-1].unsqueeze(0)), dim=0)
                # dones from term OR trunc
                term = demos["term"].to(self.device)
                trunc = demos["trunc"].to(self.device)
                dones = torch.logical_or(term, trunc)

                demos_flat = {
                    "obs": obs.flatten(0, 1),
                    "actions": acs.flatten(0, 1),
                    "next_obs": nobs.flatten(0, 1),
                    "dones": dones.flatten(0, 1),
                }
            self.reward_head.load_expert_from_demos(demos_flat)

    def test_mode(self):
        self.actor_critic.test()
        self.discriminator.test()

    def train_mode(self):
        self.actor_critic.train()
        self.discriminator.train()

    def update_ac(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        for batch in generator:
            obs_batch = batch["observations"]
            critic_obs_batch = batch["critic_observations"]
            actions_batch = batch["actions"]
            old_actions_log_prob_batch = batch["old_actions_log_prob"]
            returns_batch = batch["returns"]
            advantages_batch = batch["advantages"]
            masks_batch = batch["masks"]
            hid_states_batch = batch["hidden_states"]
            target_values_batch = batch["values"]
            old_sigma_batch = batch["sigma"]
            old_mu_batch = batch["mu"]

            self.actor_critic.act(
                obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss

    def proj(self, ob, ac, nob=None, d=None, noise=False, keep_proj=False):
        # actually use random projections here
        if self.use_linear_proj:
            # sample a number of sphere directions
            if not keep_proj:
                self.rnd = torch.randn(self.n_proj, self.discriminator.input_dim).to(
                    self.device
                )
            rnd = self.rnd
            norm_rnd = rnd / torch.norm(rnd, dim=-1, keepdim=True)
            input_ = self.concatenate_inputs(ob, ac, nob, d)
            prj = torch.matmul(input_, norm_rnd.T)
            return prj

        # random NN projections
        elif self.n_proj > 1 and not self.use_linear_proj:
            # If using a non-linear projection network as a random feature map,
            # do NOT reinitialize weights on every call; optionally allow
            # caller to request noise-based refresh via `noise=True`.
            if noise:
                with torch.no_grad():
                    self.discriminator.apply(ortho_layer_init)

        d_out = self.discriminator(self.concatenate_inputs(ob, ac, nob, d))

        return d_out / torch.norm(d_out, dim=-1, keepdim=True)

    def gsw_dist_nn(
        self,
        obs_pi,
        acs_pi,
        nobs_pi,
        d_pi,
        obs_exp,
        acs_exp,
        nobs_exp,
        d_exp,
        random=False,
    ):
        """
        Calculates sliced W2^2 between two empirical distributions under the
        current projection. Ensures consistent projections between policy and
        expert by reusing the same directions/params.
        """
        if random:
            with torch.no_grad():
                self.discriminator.apply(ortho_layer_init)

        # project slices
        # First call may sample directions; second call reuses them
        pi_slices = self.proj(obs_pi, acs_pi, nobs_pi, d_pi)
        exp_slices = self.proj(obs_exp, acs_exp, nobs_exp, d_exp, keep_proj=True)

        # sort slices
        pi_slices_sorted, _ = torch.sort(pi_slices, dim=0, stable=True)
        exp_slices_sorted, _ = torch.sort(exp_slices, dim=0, stable=True)

        self.pi_atoms_sorted.append(pi_slices_sorted.detach())
        self.exp_atoms_sorted.append(exp_slices_sorted.detach())

        # sliced W2^2: mean squared difference across samples and slices
        return (pi_slices_sorted - exp_slices_sorted).pow(2).mean()

    def compute_loss(
        self,
        exp_obs,
        exp_acs,
        pi_obs,
        pi_acs,
        exp_nobs=None,
        exp_dones=None,
        pi_nobs=None,
        pi_dones=None,
    ):
        # compute sliced Wasserstein distance here
        self.policy_obs = copy.deepcopy(pi_obs)
        self.policy_acs = copy.deepcopy(pi_acs)

        self.buffer_empty_cnt = 0

        d_loss = -self.gsw_dist_nn(
            pi_obs,
            pi_acs,
            pi_nobs,
            pi_dones,
            exp_obs,
            exp_acs,
            exp_nobs,
            exp_dones,
        )

        return d_loss

    def concatenate_inputs(self, ob, ac, nob, d):
        input_ = [ob]
        if self.use_actions:
            input_.append(ac)
        if self.use_next_obs and nob is not None:
            input_.append(nob)
        if self.use_dones and d is not None:
            input_.append(d)

        return torch.cat(input_, axis=-1)

    def forward(self, ob, ac, nob=None, d=None):
        return self.discriminator(self.concatenate_inputs(ob, ac, nob, d))

    def get_reward(self, ob, ac, nob=None, d=None):
        # If a reward head has been configured, use it instead of the legacy path
        if self.reward_head is not None and self.swil_mode is not None:
            mode = str(self.swil_mode).upper()
            rew = self.reward_head.get_reward(ob, ac, nob, d, mode=mode)
            return torch.squeeze(rew, -1)

        with torch.no_grad():
            num_envs = ob.shape[0]
            obs_t_slice = self.discriminator(self.concatenate_inputs(ob, ac, nob, d))
            rew = torch.zeros(num_envs, device=self.device)

            if len(self.pi_atoms_sorted) > 0:
                # sample random sorted atom batch from queues
                idx_ = torch.randint(0, high=len(self.pi_atoms_sorted), size=())
                idx__ = torch.randint(0, high=len(self.exp_atoms_sorted), size=())

                sorted_proj = self.pi_atoms_sorted[idx_]
                sorted_proj_tgt = self.exp_atoms_sorted[idx__]

                # TODO: figure out rewards based on multiple batches
                # -> some sort of hierarchy? OT approaches?
                # for _, (sorted_proj, sorted_proj_tgt) in enumerate(
                #     zip(pi_atoms_sorted, exp_atoms_sorted)
                # ):
                n = len(sorted_proj)
                # determine slice index in previously sorted atoms used for SWD computation
                idx = torch.searchsorted(
                    torch.transpose(sorted_proj, 0, -1), obs_t_slice.unsqueeze(0)
                ).squeeze()  # , right=True)

                # shift extreme indices
                idx[idx == n] -= 1
                idx[torch.where(idx == -1)] += 1

                if self.repl_loss_type == "diff":
                    a_prev = (
                        sorted_proj_tgt[idx.long()] - sorted_proj[idx.long()]
                    ) ** 2
                    a_new = (sorted_proj_tgt[idx.long()] - obs_t_slice) ** 2
                    rew = a_new - a_prev
                elif self.repl_loss_type == "diffmax0":
                    a_prev = (
                        sorted_proj_tgt[idx.long()] - sorted_proj[idx.long()]
                    ) ** 2
                    a_new = (sorted_proj_tgt[idx.long()] - obs_t_slice) ** 2
                    a_new[a_new > 0] = 0
                    rew += a_new - a_prev
                    if rew > 0:
                        rew = 0
                elif self.repl_loss_type == "diff2":
                    idx_i = idx.long()
                    idx_h = (idx.long() - 1).clamp_(0, None)

                    sorted_proj_tgt_i = sorted_proj_tgt[idx_i].squeeze()
                    sorted_proj_tgt_h = sorted_proj_tgt[idx_h].squeeze()
                    sorted_proj_i = torch.gather(
                        sorted_proj, 0, idx_i.view(1, sorted_proj.shape[1], 1)
                    ).squeeze()
                    sorted_proj_h = torch.gather(
                        sorted_proj,
                        0,
                        idx_h.view(1, sorted_proj.shape[1], 1),
                    ).squeeze()

                    obs_t_slice = obs_t_slice.squeeze()

                    a_i = (sorted_proj_tgt_i - sorted_proj_i) ** 2
                    a_h = (sorted_proj_tgt_h - sorted_proj_h) ** 2
                    a_new_i = (sorted_proj_tgt_i - obs_t_slice) ** 2
                    a_new_h = (sorted_proj_tgt_h - obs_t_slice) ** 2
                    diff_h = a_new_h - a_h
                    diff_i = a_new_i - a_i
                    rew = torch.where(diff_i > diff_h, diff_h, diff_i)
                elif self.repl_loss_type == "expert_diff":
                    idx_i = idx.long()
                    idx_h = (idx.long() - 1).clamp_(0, None)

                    sorted_proj_tgt_i = sorted_proj_tgt[idx_i].squeeze()
                    sorted_proj_tgt_h = sorted_proj_tgt[idx_h].squeeze()
                    sorted_proj_i = torch.gather(
                        sorted_proj, 0, idx_i.view(1, sorted_proj.shape[1], 1)
                    ).squeeze()
                    sorted_proj_h = torch.gather(
                        sorted_proj,
                        0,
                        idx_h.view(1, sorted_proj.shape[1], 1),
                    ).squeeze()

                    obs_t_slice = obs_t_slice.squeeze()

                    a_i = (sorted_proj_tgt_i - sorted_proj_i) ** 2
                    a_h = (sorted_proj_tgt_h - sorted_proj_h) ** 2
                    a_new_i = (sorted_proj_tgt_i - obs_t_slice) ** 2
                    a_new_h = (sorted_proj_tgt_h - obs_t_slice) ** 2

                    rew = -(a_new_i + a_new_h) / 2

        return rew

    def update_discriminator(self):
        # num_mini_batches = self.demos_storage.get_num_minibatches(self.irl_batch_size)
        # generator = self.storage.mini_batch_generator(
        #     num_mini_batches, self.num_irl_epochs, flatten=False
        # )
        # TODO: make sure all demo transitions are used!!!
        d_loss_avg = 0
        update_cnt = 0

        for epoch in range(self.num_irl_epochs):
            demos_generator = self.demos_storage.mini_batch_generator(
                self.irl_batch_size, shuffle=True, flatten=False
            )
            for demo_buffer_batch in demos_generator:
                exp_obs_batch = demo_buffer_batch["observations"]
                exp_actions_batch = demo_buffer_batch["actions"]
                exp_next_obs_batch = demo_buffer_batch["next_observations"]
                exp_dones_batch = demo_buffer_batch["dones"]

                # since we have much more rollout data, sample a random batch
                # TODO: think of ways to be more efficient here -> e.g. some smart sampling strategy
                rollout_buffer_batch = self.storage.get_random_batch(
                    len(exp_obs_batch), flatten=False
                )

                obs_batch = rollout_buffer_batch["observations"]
                actions_batch = rollout_buffer_batch["actions"]
                next_obs_batch = rollout_buffer_batch["next_observations"]
                dones_batch = rollout_buffer_batch["dones"]

                d_loss = self.compute_loss(
                    exp_obs=exp_obs_batch,
                    exp_acs=exp_actions_batch,
                    pi_obs=obs_batch,
                    pi_acs=actions_batch,
                    exp_nobs=exp_next_obs_batch,
                    exp_dones=exp_dones_batch,
                    pi_nobs=next_obs_batch,
                    pi_dones=dones_batch,
                )
                d_loss_avg += d_loss.item()
                update_cnt += 1
                if not self.use_linear_proj:
                    self.optimizer_d.zero_grad()
                    d_loss.backward()
                    self.optimizer_d.step()

        update_dict = {"d_loss": d_loss_avg / update_cnt}

        return update_dict


class NaSWIL(SWIL):
    """Non-Adversarial Sliced Wasserstein Imitation Learning"""

    def __init__(self, actor_critic, discriminator, il_opt, device, **kwargs):
        super().__init__(actor_critic, discriminator, il_opt, device, **kwargs)

        self.pi_slices_sorted = None
        self.exp_slices_sorted = None

    def compute_loss(
        self,
        obs_pi,
        acs_pi,
        nobs_pi,
        d_pi,
        obs_pi_2,
        acs_pi_2,
        nobs_pi_2,
        d_pi_2,
        exp_obs,
        exp_acs,
        exp_nobs,
        exp_dones,
    ):
        # 1) Project with shared directions
        pi_slices = self.proj(obs_pi, acs_pi, nobs_pi, d_pi)  # [N, K]
        pi_slices_2 = self.proj(
            obs_pi_2, acs_pi_2, nobs_pi_2, d_pi_2, keep_proj=True
        )  # [N, K]
        exp_slices = self.proj(
            exp_obs, exp_acs, exp_nobs, exp_dones, keep_proj=True
        )  # [N, K]

        # 2) Predict scalar diffs with reward NN
        pred_diffs = self.forward(obs_pi_2, acs_pi_2, nobs_pi_2, d_pi_2)  # [N, 1]

        with torch.no_grad():
            # 3) Sort per slice and transpose to [K, N]
            pi_slices_sorted, _ = torch.sort(pi_slices, dim=0, stable=True)  # [N, K]
            exp_slices_sorted, _ = torch.sort(exp_slices, dim=0, stable=True)  # [N, K]

            self.pi_slices_sorted = pi_slices_sorted
            self.exp_slices_sorted = exp_slices_sorted

            A = pi_slices_sorted.transpose(0, 1).contiguous()  # [K, N], sorted per row
            E = exp_slices_sorted.transpose(0, 1).contiguous()  # [K, N], sorted per row
            V = pi_slices_2.transpose(0, 1).contiguous()  # [K, N]
            N = A.shape[1]

            # 4) Neighbor indices where each V would be inserted into A
            idx_j = torch.searchsorted(A, V, right=False)  # [K, N] in [0..N]
            idx_j = torch.clamp(idx_j, 1, N - 1)
            idx_i = idx_j - 1

            # 5) Gather neighbors in policy and expert at same ranks
            a_i = torch.gather(A, 1, idx_i)  # [K, N]
            a_j = torch.gather(A, 1, idx_j)  # [K, N]
            e_i = torch.gather(E, 1, idx_i)  # [K, N]
            e_j = torch.gather(E, 1, idx_j)  # [K, N]

            # 6) Per-slice replacement targets
            if self.repl_loss_type == "diff2":
                # Nearest-neighbor distance (UNSIGNED). If you want signed, see below.
                diffs_per_slice = torch.minimum(torch.abs(V - a_i), torch.abs(V - a_j))
            elif self.repl_loss_type == "diff2_signed":
                # Signed distance toward the closer neighbor (useful if your head predicts signed deltas)
                left = V - a_i  # >0 if V is to the right of left neighbor
                right = a_j - V  # >0 if V is to the left of right neighbor
                use_left = torch.abs(left) <= torch.abs(right)
                nn_dist = torch.where(use_left, torch.abs(left), torch.abs(right))
                nn_sign = torch.where(
                    use_left, torch.sign(left), -torch.sign(right)
                )  # toward nearest
                diffs_per_slice = nn_sign * nn_dist
            elif self.repl_loss_type == "diff2max0":
                # Distance to the nearest boundary only if V is BETWEEN neighbors, else 0
                diffs_per_slice = torch.minimum(
                    (V - a_i).clamp_min(0.0), (a_j - V).clamp_min(0.0)
                )
            elif self.repl_loss_type == "diff3":
                # "Replacement gain" relative to expert: how much closer V is to expert than neighbor
                d_i = torch.abs(V - e_i) - torch.abs(a_i - e_i)
                d_j = torch.abs(V - e_j) - torch.abs(a_j - e_j)
                diffs_per_slice = -torch.minimum(d_i, d_j)
            elif self.repl_loss_type == "expert_diff":
                # Expert difficulty around the insertion rank (independent of V except via ranks)
                diffs_per_slice = -torch.minimum(
                    torch.abs(a_i - e_i), torch.abs(a_j - e_j)
                )
            elif self.repl_loss_type == "expert_diff_smooth":
                # After searchsorted:
                t = (V - a_i) / (a_j - a_i + 1e-12)  # [K, N], in [0,1] inside interval
                r = (idx_i + t).to(torch.float32)  # fractional rank in [0, N-1]

                # Interpolate policy/expert values at the same fractional rank
                a_interp = a_i + t * (a_j - a_i)  # policy at r
                e_interp = e_i + t * (e_j - e_i)  # expert at r

                # Smooth expert_diff:
                diffs_per_slice = -(a_interp - e_interp).abs()  # [K, N]
            else:
                # Fallback: same as expert_diff
                diffs_per_slice = -torch.minimum(
                    torch.abs(a_i - e_i), torch.abs(a_j - e_j)
                )

            # 7) Aggregate across slices -> scalar per sample
            # target = diffs_per_slice.mean(dim=0, keepdim=True).transpose(0, 1)  # [N, 1]
            target = diffs_per_slice.mean(dim=0, keepdim=True).transpose(0, 1)  # [N,1]
            target = target / (
                target.abs().mean() + 1e-6
            )  # optional per-batch scale norm

            # with torch.no_grad():
            #     # edge counts: how often searchsorted clamped
            #     edge_mask = (idx_j == 1) | (idx_j == N - 1)
            #     edge_frac = edge_mask.float().mean().item()

            #     # per-slice stats
            #     slice_std = diffs_per_slice.std(dim=1).mean().item()
            #     slice_mean = diffs_per_slice.mean().item()

            #     # target stats
            #     tgt_mean = target.mean().item()
            #     tgt_std = target.std().item()
            #     tgt_min = target.min().item()
            #     tgt_max = target.max().item()

            #     # correlation with predictions
            #     if pred_diffs.numel() == target.numel():
            #         corr = torch.corrcoef(
            #             torch.stack([pred_diffs.flatten(), target.flatten()])
            #         )[0, 1].item()
            #     else:
            #         corr = float("nan")

            #     print(
            #         f"[Batch stats] edge_frac={edge_frac:.2f}, "
            #         f"slice_std={slice_std:.4f}, slice_mean={slice_mean:.4f}, "
            #         f"target_mean={tgt_mean:.4f}, target_std={tgt_std:.4f}, "
            #         f"target_range=[{tgt_min:.4f}, {tgt_max:.4f}], "
            #         f"corr(pred,target)={corr:.3f}"
            #     )

        # 8) Regress reward head to target
        # l2_pred_diff_loss = F.mse_loss(pred_diffs, target)
        l2_pred_diff_loss = F.smooth_l1_loss(pred_diffs, target)

        # 9) For logging: [N, K]
        return l2_pred_diff_loss, diffs_per_slice.transpose(0, 1)

    def get_reward(self, ob, ac, nob=None, d=None):
        # Use reward head if configured, otherwise fall back to NaSWIL regressor output
        if self.reward_head is not None and self.swil_mode is not None:
            mode = str(self.swil_mode).upper()
            rew = self.reward_head.get_reward(ob, ac, nob, d, mode=mode)
            return torch.squeeze(rew, -1)
        reward = torch.squeeze(self.forward(ob, ac, nob, d))
        return reward

    def get_reward_(self, ob, ac, nob=None, d=None):
        obs_slice = self.proj(ob, ac, nob, d, keep_proj=True)

        if self.pi_slices_sorted is None:
            return torch.zeros(ob.shape[0], device=self.device)
        else:
            idx = torch.searchsorted(self.pi_slices_sorted.t(), obs_slice.t()).t()
            clamped_indices = torch.clamp(idx, 0, self.pi_slices_sorted.size(0) - 1)

            b1_s_i = torch.gather(self.pi_slices_sorted, dim=0, index=clamped_indices)
            expb_s_i = torch.gather(
                self.exp_slices_sorted, dim=0, index=clamped_indices
            )

            reward = -torch.abs(torch.mean(b1_s_i - expb_s_i, -1))
            return reward

    def update_discriminator(self):
        d_loss_avg = 0
        update_cnt = 0
        for epoch in range(self.num_irl_epochs):
            rollout_buffer_batch = self.storage.get_random_batch(
                self.irl_batch_size, flatten=True
            )
            obs_batch = rollout_buffer_batch["observations"]
            actions_batch = rollout_buffer_batch["actions"]
            next_obs_batch = rollout_buffer_batch["next_observations"]
            dones_batch = rollout_buffer_batch["dones"]

            rollout_buffer_batch_2 = self.storage.get_random_batch(
                self.irl_batch_size, flatten=True
            )
            obs_batch_2 = rollout_buffer_batch_2["observations"]
            actions_batch_2 = rollout_buffer_batch_2["actions"]
            next_obs_batch_2 = rollout_buffer_batch_2["next_observations"]
            dones_batch_2 = rollout_buffer_batch_2["dones"]

            exp_buffer_batch = self.demos_storage.get_random_batch(
                self.irl_batch_size, flatten=True
            )
            exp_obs_batch = exp_buffer_batch["observations"]
            exp_actions_batch = exp_buffer_batch["actions"]
            exp_next_obs_batch = exp_buffer_batch["next_observations"]
            exp_dones_batch = exp_buffer_batch["dones"]

            d_loss, diffs = self.compute_loss(
                obs_batch,
                actions_batch,
                next_obs_batch,
                dones_batch,
                obs_batch_2,
                actions_batch_2,
                next_obs_batch_2,
                dones_batch_2,
                exp_obs_batch,
                exp_actions_batch,
                exp_next_obs_batch,
                exp_dones_batch,
            )

            d_loss_avg += d_loss.item()
            update_cnt += 1

            self.optimizer_d.zero_grad()
            d_loss.backward()
            self.optimizer_d.step()

        update_dict = {"d_loss": d_loss_avg / update_cnt, "diffs": diffs}

        return update_dict
