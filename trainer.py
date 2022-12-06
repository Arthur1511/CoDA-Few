"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import jaccard_score

from networks import (FewShotGen, GPPatchMcResDis, UNet)
from utils import (dice_loss, get_model_list, get_scheduler, process_label, weights_init)


class FUNIT_Trainer(nn.Module):

    def __init__(self, hyperparameters, devices, tissue='lungs'):
        super(FUNIT_Trainer, self).__init__()

        self.cuda0, self.cuda1 = devices
        lr_gen = hyperparameters['lr_gen']
        lr_dis = hyperparameters['lr_dis']
        self.batch_size = hyperparameters['batch_size']
        self.tissue = tissue
        self.weights = None
        self.train_letters = hyperparameters['datasets_train']
        self.test_letters = hyperparameters['datasets_test']

        if hyperparameters['weights']:
            self.weights = hyperparameters['weights']

        # Initiate the networks
        self.gen = FewShotGen(hyperparameters['input_dim'] + hyperparameters['n_datasets'],
                              hyperparameters['fewshot_gen'], hyperparameters['n_datasets']).cuda(self.cuda0)

        assert hyperparameters['fewshot_dis']['num_classes'] == hyperparameters[
            'n_datasets'], 'num_classes must be equal to n_datasets'

        self.dis = GPPatchMcResDis(hyperparameters['input_dim'],
                                   hyperparameters['fewshot_dis']).cuda(self.cuda0)
        self.gen_test = copy.deepcopy(self.gen)

        self.sup = None
        self.sup = UNet(
            hyperparameters['input_dim'], hyperparameters['n_classes'], hyperparameters['sup']).cuda(self.cuda1)


        # Setup the optimizers
        dis_params = list(self.dis.parameters())
        gen_params = list(self.gen.parameters()) + list(self.sup.parameters())

        self.dis_opt = torch.optim.RMSprop(
            [p for p in dis_params if p.requires_grad],
            lr=lr_gen, weight_decay=hyperparameters['weight_decay'], )

        self.gen_opt = torch.optim.RMSprop(
            [p for p in gen_params if p.requires_grad],
            lr=lr_dis, weight_decay=hyperparameters['weight_decay'])

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))
        self.gen_test = copy.deepcopy(self.gen)

        # Presetting one hot encoding vectors.
        self.one_hot_img = torch.zeros(
            hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 256, 256).cuda(self.cuda0)
        self.one_hot_c = torch.zeros(
            hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 64, 64).cuda(self.cuda0)

        for i in range(hyperparameters['n_datasets']):
            self.one_hot_img[i, :, i, :, :].fill_(1)
            self.one_hot_c[i, :, i, :, :].fill_(1)

    def recon_criterion(self, predict, target):
        return torch.mean(torch.abs(predict - target))

    def sup_criterion(self, pred, target):

        if self.weights:
            weights = torch.tensor(self.weights).cuda(self.cuda1)
        else:
            weights = torch.tensor([1., 1.]).cuda(self.cuda1)
        
        cross_entropy = F.cross_entropy(pred, target, weight=weights, ignore_index=-1, reduction='mean')

        if -1 not in torch.unique(target):
            dice = dice_loss(pred, target, weight=weights, batch_size=self.batch_size,
                             device=self.cuda1, num_classes=2, ignore_index=-1)
        else:
            dice = 0.0

        return cross_entropy + dice

    def update_average(self, model_tgt, model_src, beta=0.999):
        with torch.no_grad():
            param_dict_src = dict(model_src.named_parameters())
            for p_name, p_tgt in model_tgt.named_parameters():
                p_src = param_dict_src[p_name]
                assert(p_src is not p_tgt)
                p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    def set_dis_trainable(self, train_bool):

        if train_bool:
            self.dis.train()
            for param in self.dis.parameters():
                param.requires_grad = True

        else:
            self.dis.eval()
            for param in self.dis.parameters():
                param.requires_grad = False

    def set_gen_trainable(self, train_bool):

        if train_bool:
            self.gen.train()
            for param in self.gen.parameters():
                param.requires_grad = True

        else:
            self.gen.eval()
            for param in self.gen.parameters():
                param.requires_grad = False

    def set_sup_trainable(self, train_bool):

        if train_bool:
            self.sup.train()
            for param in self.sup.parameters():
                param.requires_grad = True
        else:
            self.sup.eval()
            for param in self.sup.parameters():
                param.requires_grad = False

    def forward(self, x_a, x_b):

        self.eval()
        s_a = self.s_a
        s_b = self.s_b
        c_a, s_a_fake = self.gen.encode(x_a)
        c_b, s_b_fake = self.gen.encode(x_b)
        x_ba = self.gen.decode(c_b, s_a)
        x_ab = self.gen.decode(c_a, s_b)
        self.train()

        return x_ab, x_ba

    def sup_update(self, xa, xb, la, lb, ind_a, ind_b, hyperparameters):

        self.gen_opt.zero_grad()

        hot_xa = torch.cat([xa, self.one_hot_img[ind_a[0]]], 1)
        hot_xb = torch.cat([xb, self.one_hot_img[ind_b[0]]], 1)

        # encode a->b
        c_xa = self.gen.enc_content(hot_xa)
        s_xa = self.gen.enc_class_model(hot_xa)
        s_xb = self.gen.enc_class_model(hot_xb)

        # encode b->a
        c_xb = self.gen.enc_content(hot_xb)
        s_xb = self.gen.enc_class_model(hot_xb)
        s_xa = self.gen.enc_class_model(hot_xa)

        # decode
        hot_c_xa = torch.cat([c_xa, self.one_hot_c[ind_b[0]]], 1)
        hot_c_xb = torch.cat([c_xb, self.one_hot_c[ind_a[0]]], 1)

        # a->b
        xt_ab = self.gen.decode(hot_c_xa, s_xb)  # translation

        # b->a
        xt_ba = self.gen.decode(hot_c_xb, s_xa)  # translation

        # encode again
        hot_xab_t = torch.cat([xt_ab, self.one_hot_img[ind_b[0]]], 1)
        c_xab_t = self.gen.enc_content(hot_xab_t)

        hot_xba_t = torch.cat([xt_ba, self.one_hot_img[ind_a[0]]], 1)
        c_xba_t = self.gen.enc_content(hot_xba_t)

        # supervision
        p_a = self.sup(c_xa.cuda(self.cuda1))
        p_b = self.sup(c_xb.cuda(self.cuda1))
        p_a_recon = self.sup(c_xab_t.cuda(self.cuda1))
        p_b_recon = self.sup(c_xba_t.cuda(self.cuda1))

        # supervised loss
        self.loss_sup_a = self.sup_criterion(p_a, la)
        self.loss_sup_b = self.sup_criterion(p_b, lb)
        self.loss_sup_a_recon = self.sup_criterion(p_a_recon, la)
        self.loss_sup_b_recon = self.sup_criterion(p_b_recon, lb)

        # total loss
        self.loss_sup_total = hyperparameters['sup_w'] * self.loss_sup_a + \
            hyperparameters['sup_w'] * self.loss_sup_b + \
            hyperparameters['sup_w'] * self.loss_sup_a_recon + \
            hyperparameters['sup_w'] * self.loss_sup_b_recon

        self.loss_sup_total.backward()
        self.gen_opt.step()

        return hyperparameters['sup_w'] * self.loss_sup_a, \
            hyperparameters['sup_w'] * self.loss_sup_b, \
            hyperparameters['sup_w'] * self.loss_sup_a_recon, \
            hyperparameters['sup_w'] * self.loss_sup_b_recon, \
            self.loss_sup_total
    
    def gen_update(self, xa, xb, ind_a, ind_b, hyperparameters, multigpus=False):

        self.gen_opt.zero_grad()

        # encode
        hot_xa = torch.cat([xa, self.one_hot_img[ind_a[0]]], 1)
        hot_xb = torch.cat([xb, self.one_hot_img[ind_b[0]]], 1)

        c_xa = self.gen.enc_content(hot_xa)
        s_xa = self.gen.enc_class_model(hot_xa)
        s_xb = self.gen.enc_class_model(hot_xb)

        # decode
        hot_c_xa = torch.cat([c_xa, self.one_hot_c[ind_a[0]]], 1)

        xt = self.gen.decode(hot_c_xa, s_xb)  # translation
        xr = self.gen.decode(hot_c_xa, s_xa)  # reconstruction

        l_adv_t, gacc_t, xt_gan_feat = self.dis.calc_gen_loss(
            xt, ind_b)
        l_adv_r, gacc_r, xr_gan_feat = self.dis.calc_gen_loss(
            xr, ind_a)

        _, xb_gan_feat = self.dis(xb, ind_b)
        _, xa_gan_feat = self.dis(xa, ind_a)

        l_c_rec = self.recon_criterion(xr_gan_feat.mean(3).mean(2),
                                       xa_gan_feat.mean(3).mean(2))
        l_m_rec = self.recon_criterion(xt_gan_feat.mean(3).mean(2),
                                       xb_gan_feat.mean(3).mean(2))
        l_x_rec = self.recon_criterion(xr, xa)
        l_adv = 0.5 * (l_adv_t + l_adv_r)
        acc = 0.5 * (gacc_t + gacc_r)
        l_total = (hyperparameters['gan_w'] * l_adv + hyperparameters['r_w'] * l_x_rec + hyperparameters[
            'fm_w'] * (l_c_rec + l_m_rec))
        l_total.backward()

        self.loss_gen_total = torch.mean(l_total)
        self.loss_gen_recon_x = torch.mean(l_adv)
        self.loss_gen_recon_c = torch.mean(l_x_rec)
        self.loss_gen_recon_s = torch.mean(l_c_rec)
        self.loss_gen_adv = torch.mean(l_m_rec)
        self.accuracy_gen_adv = torch.mean(acc)

        self.gen_opt.step()
        self.update_average(self.gen_test, self.gen)

        return self.loss_gen_recon_x.item(), self.loss_gen_recon_c.item(), self.loss_gen_recon_s.item(), self.loss_gen_adv.item(), self.loss_gen_total.item(), self.accuracy_gen_adv.item()

    def dis_update(self, xa, xb, ind_a, ind_b, hyperparameters):
        self.dis_opt.zero_grad()

        xb.requires_grad_()
        l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(xb, ind_b)
        l_real = hyperparameters['gan_w'] * l_real_pre
        l_real.backward(retain_graph=True)
        l_reg_pre = self.dis.calc_grad2(resp_r, xb)
        l_reg = 10 * l_reg_pre
        l_reg.backward()

        with torch.no_grad():
            hot_xa = torch.cat([xa, self.one_hot_img[ind_a[0]]], 1)
            hot_xb = torch.cat([xb, self.one_hot_img[ind_b[0]]], 1)

            c_xa = self.gen.enc_content(hot_xa)
            s_xb = self.gen.enc_class_model(hot_xb)

            hot_c_xa = torch.cat([c_xa, self.one_hot_c[ind_a[0]]], 1)

            xt = self.gen.decode(hot_c_xa, s_xb)

        l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(
            xt.detach(), ind_b)
        l_fake = hyperparameters['gan_w'] * l_fake_p
        l_fake.backward()
        l_total = l_fake + l_real + l_reg
        acc = 0.5 * (acc_f + acc_r)

        self.loss_dis_total = torch.mean(l_total)
        self.loss_dis_fake_adv = torch.mean(l_fake_p)
        self.loss_dis_real_adv = torch.mean(l_real_pre)
        self.loss_dis_reg = torch.mean(l_reg_pre)
        self.accuracy_dis_adv = torch.mean(acc)
        self.dis_opt.step()


    def sample(self, img, msk, ind):
        self.sup.eval()
        self.gen_test.eval()

        img_recon_list = []
        pred_list = []
        msk_list = []

        if self.test_letters[ind] in self.train_letters:
            ind = self.train_letters.index(self.test_letters[ind])
        else:
            ind = 0

        for i in range(img.size(0)):

            hot_img = torch.cat(
                [img[i].unsqueeze(0), self.one_hot_img[ind, 0].unsqueeze(0)], 1)

            content_img = self.gen_test.enc_content(hot_img)
            style_img = self.gen_test.enc_class_model(hot_img)

            hot_content = torch.cat(
                [content_img, self.one_hot_c[ind, 0].unsqueeze(0)], 1)

            img_recon = self.gen_test.decode(hot_content, style_img).cpu()
            img_recon_list.append(img_recon)

            pred = self.sup(content_img.cuda(self.cuda1)).max(1)[1]
            pred_list.append(pred.unsqueeze(0).cpu())
            msk_list.append(msk[i].unsqueeze(0).unsqueeze(0).cpu())

        pred_list = torch.cat(pred_list)
        msk_list = torch.cat(msk_list)
        img_recon_list = torch.cat(img_recon_list)

        self.sup.train()
        self.gen_test.train()

        return img.cpu(), img_recon_list, msk_list, pred_list

    def sup_forward(self, x, y, d_index, hyperparameters):

        self.sup.eval()
        self.gen_test.eval()

        if self.test_letters[d_index] in self.train_letters:
            d_index = self.train_letters.index(self.test_letters[d_index])
        else:
            d_index = 0

        # Encoding content image.
        one_hot_x = torch.cat(
            [x, self.one_hot_img[d_index, 0].unsqueeze(0)], 1)

        content_img = self.gen_test.enc_content(one_hot_x)

        # Forwarding on supervised model.
        y_pred = self.sup(content_img.cuda(self.cuda1), only_prediction=True)

        # Computing metrics.
        pred = y_pred.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        if np.any(pred == 1):
            pred = process_label(pred, self.tissue)

        jacc = jaccard_score(y.cpu().squeeze(
            0).numpy().flatten(), pred.flatten())

        self.sup.train()
        self.gen_test.train()

        return jacc, pred
    
    def get_iso(self, x, y, d_index, hyperparameters):

        self.gen_test.eval()

        if self.test_letters[d_index] in self.train_letters:
            d_index = self.train_letters.index(self.test_letters[d_index])
        else:
            d_index = 0

        # Encoding content image.
        one_hot_x = torch.cat(
            [x, self.one_hot_img[d_index, 0].unsqueeze(0)], 1)

        content_img = self.gen_test.enc_content(one_hot_x)

        self.gen_test.train()

        return content_img.cpu().detach().numpy()

    def test(self, xa, xb, ind_a, ind_b):
        self.eval()
        self.gen.eval()
        self.gen_test.eval()

        hot_xa = torch.cat([xa, self.one_hot_img[ind_a]], 1)
        hot_xb = torch.cat([xb, self.one_hot_img[ind_b]], 1)

        c_xa_current = self.gen.enc_content(hot_xa)
        s_xa_current = self.gen.enc_class_model(hot_xa)
        s_xb_current = self.gen.enc_class_model(hot_xb)

        hot_c_xa_current = torch.cat([c_xa_current, self.one_hot_c[ind_a]], 1)

        xt_current = self.gen.decode(hot_c_xa_current, s_xb_current)
        xr_current = self.gen.decode(hot_c_xa_current, s_xa_current)

        c_xa = self.gen_test.enc_content(hot_xa)
        s_xa = self.gen_test.enc_class_model(hot_xa)
        s_xb = self.gen_test.enc_class_model(hot_xb)

        hot_c_xa = torch.cat([c_xa, self.one_hot_c[ind_a]], 1)

        xt = self.gen_test.decode(hot_c_xa, s_xb)
        xr = self.gen_test.decode(hot_c_xa, s_xa)
        self.train()

        return xa, xr_current, xt_current, xb, xr, xt

    def resume(self, checkpoint_dir, hyperparameters):

        # Load supervised
        last_model_name = get_model_list(checkpoint_dir, 'sup')
        state_dict = torch.load(last_model_name)
        self.sup.load_state_dict(state_dict['sup'])
        iterations = int(last_model_name[-11:-3])

        # Load generators
        last_model_name = get_model_list(checkpoint_dir, 'gen')
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict['gen'])
        self.gen_test.load_state_dict(state_dict['gen_test'])
        iterations = int(last_model_name[-11:-3])

        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, 'dis')
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict['dis'])

        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(
            self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(
            self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)

        return iterations

    def save(self, snapshot_dir, iterations):

        # Save generators, discriminators, and optimizers
        sup_name = os.path.join(snapshot_dir, 'sup_%08d.pt' % (iterations + 1))
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')

        torch.save({'sup': self.sup.state_dict()}, sup_name)
        torch.save({'gen': self.gen.state_dict(),
                    'gen_test': self.gen_test.state_dict()}, gen_name)
        torch.save({'dis': self.dis.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(),
                    'dis': self.dis_opt.state_dict()}, opt_name)

    def translate_k_shot(self, xa, xb, ind_a, ind_b, k):
        self.eval()

        hot_xa = torch.cat([xa, self.one_hot_img[ind_a]], 1)
        hot_xb = torch.cat([xb, self.one_hot_img[ind_b]], 1)

        c_xa_current = self.gen_test.enc_content(hot_xa)

        if k == 1:
            c_xa_current = self.gen_test.enc_content(hot_xa)
            s_xb_current = self.gen_test.enc_class_model(hot_xb)

            hot_c_xa_current = torch.cat(
                [c_xa_current, self.one_hot_c[ind_a]], 1)
            xt_current = self.gen_test.decode(hot_c_xa_current, s_xb_current)
        else:
            s_xb_current_before = self.gen_test.enc_class_model(hot_xb)
            s_xb_current_after = s_xb_current_before.squeeze(-1).permute(1,
                                                                         2,
                                                                         0)
            s_xb_current_pool = torch.nn.functional.avg_pool1d(
                s_xb_current_after, k)
            s_xb_current = s_xb_current_pool.permute(2, 0, 1).unsqueeze(-1)

            hot_c_xa_current = torch.cat(
                [c_xa_current, self.one_hot_c[ind_a]], 1)
            xt_current = self.gen_test.decode(hot_c_xa_current, s_xb_current)

        return xt_current
