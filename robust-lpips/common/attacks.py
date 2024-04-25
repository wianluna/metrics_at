import torch
import random


class Attack:
    def __init__(self, proba=0.5):
        """
        Helper class for attacks
        @param proba: probability of actually attacking the batch
        """
        self.proba = proba

    def attack_impl(self, ref, dist, direction):
        """
        Attack image 
        @param ref: reference image
        @param dist: batch of images to attack
        @param direction: 1d tensor containing directions (-1 or 1) for each image in batch
        @returns: attacked
        """
        raise NotImplementedError()

    def get_name(self):
        """
        Attack name
        @returns: attack name
        """
        raise NotImplementedError()

    def attack_pairs(self, model, ref, left, right, scores):
        """
        Attack images from two batches
            if subjective quality of image is greater than correspoing image from the other batch,
            attack tries to decrease its quality. Otherwise, attack tries to increase it.
        @param ref: reference image
        @param left: first batch of images to attack
        @param right: second batch of images to attack
        @param scores: 1d tesnor containing subjective scores of corresponing images in batch.
            Greater value means corresponding image from second batch is better
        @param direction: 1d tensor containing directions (-1 or 1) for each image in batch
        @returns: optimized image
        """
        left_worse = (scores > 0.5).view(-1, 1, 1, 1).to(left.device)
        worse = left * left_worse + right * ~left_worse
        better = left * ~left_worse + right * left_worse

        worse_attacked = self.attack_impl(model, ref,  worse, True) if random.uniform(0, 1) < self.proba else worse
        better_attacked = self.attack_impl(model, ref, better, False) if random.uniform(0, 1) < self.proba else better
        left_attacked = worse_attacked * left_worse + better_attacked * ~left_worse
        right_attacked = worse_attacked * ~left_worse + better_attacked * left_worse
        return left_attacked, right_attacked


class NoAttack(Attack):
    """
    Dummy class that doesn't attack anything
    """
    def __init__(self):
        super().__init__(0)

    def attack_impl(self, model, ref, dist, direction):
        return dist

    def get_name(self):
        return "NoAttack"


class BaseFGSM(Attack):
    def __init__(self, eps, noisy, step, proba=0.5):
        super().__init__(proba)
        self.eps = eps
        self.noisy = noisy
        self.step = step or (eps * 1.25)

    def attack_impl(self, model, ref, dist, direction):
        delta = torch.zeros_like(dist)
        if self.noisy:
            delta.uniform_(-self.eps, self.eps)
        delta.requires_grad_(True)
        outs = model.forward(ref, dist + delta)
        grad = torch.autograd.grad(outs.sum(), [delta])[0].detach()
        if isinstance(direction, bool):
            dir = -1 if not direction else 1
            delta.data -= self.step * dir * torch.sign(grad)
        else:
            dir = direction * 2 - 1
            delta.data -= self.step * dir[:, None, None, None] * torch.sign(grad)
        delta.data.clamp_(-self.eps, self.eps)
        return delta + dist


class FGSM(BaseFGSM):
    def __init__(self, eps, step=None, proba=0.5):
        super().__init__(eps, False, step, proba)

    def get_name(self):
        return f"FGSM_{int(self.eps * 255)}_255"


class FreeFGSM(BaseFGSM):
    def __init__(self, eps, step=None, proba=0.5):
        super().__init__(eps, True, step, proba)

    def get_name(self):
        return f"FreeFGSM_{int(self.eps * 255)}_255"


class IFGSM(Attack):
    def __init__(self, eps, step=1/255, num_iters=10, proba=0.5):
        super().__init__(proba)
        self.eps = 10/255
        self.step = 1/255
        self.num_iters = 10

    def attack_impl(self, model, ref, dist, direction):
        if self.eps == 0:
            return dist
        delta = torch.zeros_like(dist)
        delta.requires_grad_(True)
        for i in range(self.num_iters):
            outs = model.forward(ref, dist + delta)
            grad = torch.autograd.grad(outs.sum(), [delta])[0].detach()
            if isinstance(direction, bool):
                dir = -1 if not direction else 1
                delta.data -= self.step * dir * torch.sign(grad)
            else:
                dir = direction * 2 - 1
                delta.data -= self.step * dir[:, None, None, None] * torch.sign(grad)
            delta.data.clamp_(-self.eps, self.eps)
        return delta + dist

    def get_name(self):
        return f"IFGSM_{int(self.eps * 255)}_255"


class PGD(Attack):
    def __init__(self, eps, step=1/255, num_iters=10, proba=0.5):
        super().__init__(proba)
        self.eps = eps
        self.step = step or self.eps
        self.num_iters = num_iters

    def attack_impl(self, model, ref, dist, direction):
        if self.eps == 0:
            return dist
        delta = torch.zeros_like(dist)
        delta.uniform_(-self.eps, self.eps)
        delta.requires_grad_(True)
        for i in range(self.num_iters):
            outs = model.forward(ref, dist + delta)
            grad = torch.autograd.grad(outs.sum(), [delta])[0].detach()
            if isinstance(direction, bool):
                dir = -1 if not direction else 1
                delta.data -= self.step * dir * torch.sign(grad)
            else:
                dir = direction * 2 - 1
                delta.data -= self.step * dir[:, None, None, None] * torch.sign(grad)
            delta.data.clamp_(-self.eps, self.eps)
        return delta + dist

    def get_name(self):
        return f"PGD_{int(self.eps * 255)}_255"


class AutoPGD(Attack):
    def __init__(
        self,
        n_iter: int = 2,
        eps: float = 4.0,
        alpha: float = 1.0,
        thr_decr: float = 0.75,
    ):
        self.n_iter = n_iter
        self.n_iter_min = max(int(0.06 * n_iter), 1)
        self.size_decr = max(int(0.03 * n_iter), 1)
        self.k = max(int(0.22 * n_iter), 1)

        self.eps = eps / 255
        self.alpha = alpha
        self.thr_decr = thr_decr


    def check_oscillation(self, loss_steps, cur_step):
        t = torch.zeros(loss_steps.shape[1], device=loss_steps.device, dtype=loss_steps.dtype)
        for i in range(self.k):
            t += (loss_steps[cur_step - i] > loss_steps[cur_step - i - 1]).float()

        return (t <= self.k * self.thr_decr * torch.ones_like(t)).float()

    def attack_impl(self, model, ref, dist, direction):
        self.k = max(int(0.22 * self.n_iter), 1)
        device = dist.device

        x_adv = dist.clone()
        x_best = x_adv.clone().detach()
        loss_steps = torch.zeros([self.n_iter, dist.shape[0]], device=device)
        loss_best_steps = torch.zeros([self.n_iter + 1, dist.shape[0]], device=device)

        step_size = (
            self.alpha
            * self.eps
            * torch.ones(
                [dist.shape[0], *[1] * (len(dist.shape) - 1)], device=device, dtype=dist.dtype
            )
        )

        counter3 = 0

        # 1 step of the classic PGD
        x_adv.requires_grad_()

        logits = model.forward(ref, x_adv)
        loss_indiv = (logits if direction else -logits)[:, 0, 0, 0].clone()
        loss = loss_indiv.sum()
        if not direction:
            loss *= -1

        grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        grad_best = grad.clone()
        x_adv.detach_()
        loss_indiv.detach_()
        loss.detach_()

        loss_best = loss_indiv.detach().clone()
        loss_best = loss_best.squeeze(-1)
        loss_best_last_check = loss_best.clone()
        loss_best_last_check = loss_best_last_check.squeeze(-1)
        reduced_last_check = torch.ones_like(loss_best)

        x_adv_old = x_adv.clone().detach()

        for i in range(self.n_iter):
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()

            a = 0.75 if i > 0 else 1.0

            x_adv_1 = x_adv + step_size * torch.sign(grad)
            x_adv_1 = torch.clamp(
                torch.min(torch.max(x_adv_1, dist - self.eps), dist + self.eps), 0.0, 1.0
            )
            x_adv_1 = torch.clamp(
                torch.min(
                    torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), dist - self.eps),
                    dist + self.eps,
                ),
                0.0,
                1.0,
            )

            x_adv = x_adv_1

            if i < self.n_iter - 1:
                x_adv.requires_grad_()

            logits = model.forward(ref, x_adv)
            loss_indiv = (logits if not direction else -logits)[:, 0, 0, 0].clone()

            loss = loss_indiv.sum()

            if i < self.n_iter - 1:
                grad = torch.autograd.grad(loss, [x_adv])[0].detach()

            x_adv.detach_()
            loss_indiv.detach_()
            loss.detach_()

            # check step size
            y1 = loss_indiv.detach().clone()
            y1 = y1.squeeze(-1)
            loss_steps[i] = y1
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind]
            loss_best_steps[i + 1] = loss_best

            counter3 += 1

            if counter3 == self.k:
                fl_oscillation = self.check_oscillation(loss_steps, i)
                fl_reduce_no_impr = (1.0 - reduced_last_check) * (
                    loss_best_last_check >= loss_best
                ).float()
                fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
                reduced_last_check = fl_oscillation.clone()
                loss_best_last_check = loss_best.clone()

                if fl_oscillation.sum() > 0:
                    ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                    step_size[ind_fl_osc] /= 2.0

                    x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                    grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                counter3 = 0
                self.k = max(self.k - self.size_decr, self.n_iter_min)

        return x_best
