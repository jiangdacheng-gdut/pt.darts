""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch


class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        参数：
            网络
            w_momentum: 对动量进行加权
        """
        self.net = net
        self.v_net = copy.deepcopy(net)  # 创建一个原始网络的副本
        self.w_momentum = w_momentum  # 权重动量
        self.w_weight_decay = w_weight_decay  # 权重衰减

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        计算展开权重 w' (虚拟步长)
        步骤:
        1) forward
        2) 计算损失函数
        3) 计算梯度 (by backprop)
        4) 更新梯度

        Args:
            xi: 虚拟梯度步长学习率(与权重lr相同)
            w_optim: 权重优化器
        """
        # 前向传播 & 计算损失
        loss = self.net.loss(trn_X, trn_y)  # L_trn(w)

        # 计算梯度
        gradients = torch.autograd.grad(loss, self.net.weights())

        # 执行虚拟步骤（更新梯度）
        # 下面的操作不需要跟踪梯度
        with torch.no_grad():
            # 字典的键不是值，而是指针。因此，原始网络权重也必须迭代。
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # 同步 alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)


    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y) # L_val(w`)

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h

    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
