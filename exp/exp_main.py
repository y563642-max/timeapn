from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, FEDformer, iTransformer, PatchTST, S_Mamba
# from models.DDN import DDN
from models.APN import APN
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
# from muon import MuonWithAuxAdam

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import time
from pytorch_msssim import ssim

import warnings
import matplotlib.pyplot as plt
from scipy.signal import hilbert


warnings.filterwarnings('ignore')

import torch.distributed as dist

def norm_phase(x):

    F_w = np.fft.fft(x, axis=1)
    phi_f = np.angle(F_w) 

    return phi_f

def split_parameters(model: nn.Module):
    muon_params = []
    adamw_params = []

    # 分离 mlp 中参数
    for p in model.mlp.parameters():
        (muon_params if p.ndim >= 2 else adamw_params).append(p)

    # 加入 dwt_ratio
    adamw_params.append(model.dwt_ratio)

    # 加入 DWT 参数（如果有）
    if any(p.requires_grad for p in model.dwt.parameters()):
        for p in model.dwt.parameters():
            (muon_params if p.ndim >= 2 else adamw_params).append(p)

    return muon_params, adamw_params


def hilbert_phase_scipy(x: torch.Tensor) -> torch.Tensor:
    """
    使用 scipy.signal.hilbert 计算 Hilbert 变换的瞬时相位
    
    参数:
        x: torch.Tensor, 形状为 [batch_size, channel, length]，实数输入
    
    返回:
        phase: torch.Tensor, 同样形状，瞬时相位，单位弧度
    """
    x_np = x.detach().cpu().numpy()  # 转为 numpy，先移到 CPU
    # 对最后一个维度做 hilbert
    analytic_signal = hilbert(x_np, axis=2)
    phase_np = np.angle(analytic_signal)
    phase = torch.from_numpy(phase_np).to(x.device)  # 转回 tensor 并放回原设备
    return phase

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.station_pretrain_epoch = args.pre_epoch if self.args.station_type == 'adaptive' else 0
        self.station_type = args.station_type
        self.kernel = kernel = args.kernel_len
        self.hkernel = hkernel = args.hkernel_len
        self.pred_len = args.pred_len
        self.kernel_size = args.kernel_size
        self.channels = args.enc_in if args.features == 'M' else 1
        self.pad = nn.ReplicationPad1d(padding=(kernel // 2, kernel // 2 - ((kernel + 1) % 2)))
        self.pad_phase = nn.ReplicationPad1d(padding=(2 // 2, 2 // 2 - ((2 + 1) % 2)))
        if hkernel is not None:
            self.hpad = nn.ReplicationPad1d(padding=(hkernel // 2, hkernel // 2 - ((hkernel + 1) % 2)))
        self.norm_f = self.norm_sliding
        self.norm_loss = self.norm_sliding_loss
        

    def _build_model(self):
        self.statistics_pred_P = APN(self.args).to(self.device)
        self.station_loss_P = self.sliding_loss_P
        self.channels = 7
        self.ratio = nn.Parameter(torch.clamp(torch.tensor(0.0), min=0., max=1.)).cuda()
        self.ratio1 = nn.Parameter(torch.clamp(torch.tensor(0.0), min=0., max=1.)).cuda()

        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'iTransformer': iTransformer,
            'PatchTST': PatchTST,
            'S_Mamba': S_Mamba,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        # self.muon_params = [p for p in model.body.parameters() if p.ndim >= 2]
        # self.adamw_params = [p for p in model.body.parameters() if p.ndim < 2]
        # self.adamw_params += list(model.head.parameters()) + list(model.embed.parameters())

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
        

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):

        # # 参数分组
        # muon_weights = [p for p in self.statistics_pred_P.parameters() if p.ndim >= 2]
        # adamw_weights = [p for p in self.statistics_pred_P.parameters() if p.ndim < 2]

        # # 分别设置两组参数
        # muon_group = {
        #     "params": muon_weights,
        #     "lr": 0.02,
        #     "weight_decay": 0.01,
        #     "use_muon": True  # 使用 muon 更新
        # }

        # adamw_group = {
        #     "params": adamw_weights,
        #     "lr": 3e-4,
        #     "betas": (0.9, 0.95),
        #     "weight_decay": 0.01,
        #     "use_muon": False  # 使用普通 AdamW 更新
        # }
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # self.station_optim_P = MuonWithAuxAdam([muon_group, adamw_group])
        self.station_optim_P = optim.Adam(self.statistics_pred_P.parameters(), lr=self.args.station_lr)
        # self.station_optim_P = MuAdam(self.statistics_pred_P.parameters(), lr=self.args.station_lr, betas=(0.9, 0.999), weight_decay=1e-4)
        return model_optim

    def _select_criterion(self):
        self.criterion = nn.MSELoss()
     

    def norm_sliding(self, x, kernel=None):
        if kernel is None:
            kernel, pad = self.kernel, self.pad
        else:
            pad = self.hpad
        # res_init, trend_init = self.decomp_module(x.permute(0,2,1))
        # x_window = trend_init.permute(0,2,1).unfold(-1, kernel, 1)

        x_window = x.unfold(-1, kernel, 1)  # sliding window
        m, s = x_window.mean(dim=-1), x_window.std(dim=-1)  # acquire sliding mean and sliding standard deviation
        m, s = pad(m), pad(s)  # nn.ReplicationPad1d(padding=(kernel // 2, kernel // 2 - ((kernel + 1) % 2)))
        # x = (x - m) / (s + self.epsilon)  # x is stationary series
        x = (x - m) 
        F_w = torch.fft.fft(x, dim=2)
        A_f = torch.abs(F_w)
        # A_f = torch.where(A_f < 10, torch.zeros_like(A_f), A_f)
        phi_f = torch.angle(F_w) 
        # phi_f = hilbert_phase_scipy(x)
        # eps = 10 + 1e-8
        # phi_f = torch_unwrap(torch.where(A_f > eps, phi_f, torch.zeros_like(phi_f)), dim=2)
        
        # phi_f = torch.where(A_f < 10, torch.zeros_like(phi_f), phi_f)
        return x, (m, phi_f)  # m, s are non-stationary factors
    
    def norm_sliding_loss(self, x, kernel=None):
        if kernel is None:
            kernel, pad = self.kernel, self.pad
        else:
            pad = self.hpad
        # res_init, trend_init = self.decomp_module(x.permute(0,2,1))
        # x_window = trend_init.permute(0,2,1).unfold(-1, kernel, 1)

        x_window = x.unfold(-1, kernel, 1)  # sliding window
        
        m, s = x_window.mean(dim=-1), x_window.std(dim=-1)  # acquire sliding mean and sliding standard deviation
        m, s = pad(m), pad(s)  # nn.ReplicationPad1d(padding=(kernel // 2, kernel // 2 - ((kernel + 1) % 2)))
        # x = (x - m) / (s + self.epsilon)  # x is stationary series
        x = (x - m) 
        F_w = torch.fft.fft(x, dim=2)
        A_f = torch.abs(F_w)
        # A_f = torch.where(A_f < 10, torch.zeros_like(A_f), A_f)
        phi_f = torch.angle(F_w)
        # phi_f = hilbert_phase_scipy(x)
        # eps = 10 + 1e-8
        # phi_f = torch_unwrap(torch.where(A_f > eps, phi_f, torch.zeros_like(phi_f)), dim=2)
        
        # phi_f = torch.where(A_f < 10, torch.zeros_like(phi_f), phi_f)
        return x, (m, phi_f)

    def station_loss(self, y, statistics_pred):
        bs, len, dim = y.shape
        y = y.reshape(bs, -1, self.args.period_len, dim)
        mean = torch.mean(y, dim=2)
        std = torch.std(y, dim=2)
        station_ture = torch.cat([mean, std], dim=-1)
        loss = self.criterion(statistics_pred, station_ture)
        return loss

    def san_loss(self, y, statistics_pred):
        bs, len, dim = y.shape
        y = y.reshape(bs, -1, self.args.period_len, dim)
        mean = torch.mean(y, dim=2)
        std = torch.std(y, dim=2)
        station_ture = torch.cat([mean, std], dim=-1)
        loss = self.criterion(statistics_pred, station_ture)
        return loss
    
    def sliding_loss_P(self, y, statistics_pred, y_pred, t):
        _, (mean, std) = self.norm_loss(y.transpose(-1, -2))
        _, (mean_pre, std_pre) = self.norm_loss(y_pred.transpose(-1, -2))

        station_ture = torch.cat([mean, std], dim=1).transpose(-1, -2)
        mean = statistics_pred[..., :statistics_pred.shape[-1] // 2]
        std = statistics_pred[..., statistics_pred.shape[-1] // 2:]

        loss_am = self.criterion(mean, station_ture[..., :station_ture.shape[-1] // 2])
        # loss_am1 = self.criterion(mean[:,0:25,:], station_ture[..., :station_ture.shape[-1] // 2][:,0:25,:])
        loss_phase = self.criterion(std, station_ture[..., station_ture.shape[-1] // 2:])
        # loss_phase = torch.mean(torch.square(torch.abs(torch.exp(1j * (std- station_ture[..., station_ture.shape[-1] // 2:])) )))
        # phase_diff = station_ture[..., station_ture.shape[-1] // 2:] - std - torch.pi
        # loss_phase = torch.mean(1 - torch.cos(phase_diff))

        # loss_phase = 1 - ssim(std.unsqueeze(1), station_ture[..., station_ture.shape[-1] // 2:].unsqueeze(1), data_range=1.0, size_average=True)
#        F_w_y = torch.fft.fft(y, dim=2)
#        amp = torch.abs(F_w_y)
#        F_w_y_pred = torch.fft.fft(y_pred, dim=2)
#        amp_pre = torch.abs(F_w_y_pred)
#        loss_amp = self.criterion(amp, amp_pre) 

        mov_r, dwt_r = self.ratio, 1 - self.ratio
        data_r, phase_r = self.ratio1, 1 - self.ratio1
        alpha = 1
        w0 =1
        k = 0.01
        t = torch.tensor(t, dtype=torch.float32)
        w = w0 * torch.exp(-k * t)
        loss = self.criterion(y, y_pred) + loss_phase + loss_am 
#        loss = self.criterion(y, y_pred) + loss_phase + loss_am* dwt_r + loss_amp* mov_r
        # print('mse_am:{}, mse_phase:{}'.format(loss_am, loss_phase))
        # loss = self.criterion(y, y_his) + smooth_loss_val
        return loss, loss_am 

    def vali(self, vali_data, vali_loader, criterion, epoch):
        total_loss = []
        self.model.eval()
        self.statistics_pred_P.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
       
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if  epoch + 1 <= ( self.station_pretrain_epoch) and self.args.use_norm == 'sliding':
                    batch_x, statistics_pred_P, statistics_seq = self.statistics_pred_P.normalize(batch_x, p_value=False)
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                y = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            y = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                if self.args.use_norm == 'sliding':
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                else:
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # y = self.model(batch_x)
                    norm_y, (seq_m_y, seq_s_y) = self.norm_f(y.transpose(-1, -2))#返回预测数据及预测数据的均值与相位谱
                    B1, T1, N1 = y.size()  # y: [B, T, N]
                    groups = B1 * N1       # 一共 672 个 group

                    # input: reshape to [1, groups, T]
                    input = y.permute(0, 2, 1).reshape(1, groups, T1)  # [1, 672, 336]

                    # kernel: [groups, 1, kernel_size]
                    kernel_raw = torch.fft.ifft(torch.exp(1j * statistics_pred_P[1]), dim=-1)       # shape: [B, T, N]
                    kernel = torch.abs(kernel_raw).permute(0, 2, 1).reshape(groups, 1, T1)  # [672, 1, 336]
                    kernel = torch.flip(kernel, dims=[2])  # 卷积核要翻转

                    # 卷积：grouped conv
                    out = F.conv1d(input, kernel, groups=groups, padding=T1 // 2)[:,:,:self.pred_len]  # [1, 672, 336]

                    # reshape 回原 shape
                    y = out.view(B1, N1, T1) + statistics_pred_P[0]
                    y = y.transpose(-1, -2)
                    statistics_pred_P1 = (statistics_pred_P[0], seq_s_y + statistics_pred_P[1])
                    statistics_pred_P1 = torch.cat(statistics_pred_P1, dim=1).transpose(-1, -2)

                    statistics_pred_P = (statistics_pred_P[0], statistics_pred_P[1])
                    statistics_pred_P = torch.cat(statistics_pred_P, dim=1).transpose(-1, -2)

                elif self.args.use_norm == 'sliding':
                    batch_x, statistics_pred_P, statistics_seq = self.statistics_pred_P.normalize(batch_x)
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float()
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                y = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            y = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                if self.args.use_norm == 'sliding':
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                else:
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    norm_y, (seq_m_y, seq_s_y) =self.norm_f(y.transpose(-1, -2))#返回预测数据及预测数据的均值与相位谱
                    
                    B1, T1, N1 = y.size()  # y: [B, T, N]
                    groups = B1 * N1       # 一共 672 个 group

                    # input: reshape to [1, groups, T]
                    input = y.permute(0, 2, 1).reshape(1, groups, T1)  # [1, 672, 336]

                    # kernel: [groups, 1, kernel_size]
                    kernel_raw = torch.fft.ifft(torch.exp(1j * statistics_pred_P[1]), dim=-1) 
                    kernel = torch.abs(kernel_raw).permute(0, 2, 1).reshape(groups, 1, T1)  # [672, 1, 336]
                    kernel = torch.flip(kernel, dims=[2])  # 卷积核要翻转

                    # 卷积：grouped conv
                    out = F.conv1d(input, kernel, groups=groups, padding=T1 // 2)[:,:,:self.pred_len]  # [1, 672, 336]

                    # reshape 回原 shape
                    y = out.view(B1, N1, T1) + statistics_pred_P[0]
                    y = y.transpose(-1, -2)
                    statistics_pred_P1 = (statistics_pred_P[0], seq_s_y + statistics_pred_P[1])
                    statistics_pred_P1 = torch.cat(statistics_pred_P1, dim=1).transpose(-1, -2)
                    statistics_pred_P = (statistics_pred_P[0], statistics_pred_P[1])
                    statistics_pred_P = torch.cat(statistics_pred_P, dim=1).transpose(-1, -2)

                else:
                    batch_x, statistics_pred_P = self.statistics_pred_P.normalize(batch_x)

                if epoch + 1 <= self.station_pretrain_epoch:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.features == 'MS':
                        statistics_pred_P1 = statistics_pred_P1[:, :, [self.args.enc_in - 1, -1]]
                    loss, loss_am = self.station_loss_P(batch_y, statistics_pred_P1, y, epoch)
                
                else:
                    # decoder x
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float()
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                if self.args.use_norm == 'sliding':
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    if self.args.features == 'MS':
                        statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                        statistics_pred_P = statistics_pred_P[:, :, [self.args.enc_in - 1, -1]]
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    outputs = self.statistics_pred_P.de_normalize(outputs, statistics_pred_P)
                    
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    loss = criterion(pred, true)

                total_loss.append(loss.cpu().item())
        total_loss = np.average(total_loss)
        self.model.train()
        self.statistics_pred_P.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        path_station_P = './station_P/' + '{}_s{}_p{}'.format(self.args.model_id, self.args.data,
                                                          self.args.seq_len, self.args.pred_len)
        if not os.path.exists(path_station_P):
            os.makedirs(path_station_P)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        early_stopping_station_model_P = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        train_loss2 = []
        for epoch in range(self.args.train_epochs + self.station_pretrain_epoch):
            iter_count = 0
            train_loss = []

            train_loss1 = []

            if epoch ==  self.station_pretrain_epoch  and self.args.station_type == 'adaptive':
                best_model_path_P = path_station_P + '/' + 'checkpoint.pth'
                self.statistics_pred_P.load_state_dict(torch.load(best_model_path_P))
                print('loading pretrained adaptive station model')
                if self.args.use_norm == 'sliding' and self.args.twice_epoch >= 0:
                    print('reset station model optim for finetune')
            ##
            if self.args.use_norm == 'sliding' and 0 <= self.args.twice_epoch == (epoch - self.station_pretrain_epoch):
                lr = model_optim.param_groups[0]['lr']
                model_optim.add_param_group({'params': self.statistics_pred_P.parameters(), 'lr': lr})

            self.model.train()
            self.statistics_pred_P.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if  epoch + 1 <=  self.station_pretrain_epoch and self.args.use_norm == 'sliding':
                    batch_x, statistics_pred_P, statistics_seq = self.statistics_pred_P.normalize(batch_x, p_value=False)
                    
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float()

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                y = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            y = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                if self.args.use_norm == 'sliding':
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                else:
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    norm_y, (seq_m_y, seq_s_y) =self.norm_f(y.transpose(-1, -2))#返回预测数据及预测数据的均值与相位谱
                    
                    B1, T1, N1 = y.size()  # y: [B, T, N]
                    groups = B1 * N1       # 一共 672 个 group

                    # input: reshape to [1, groups, T]
                    input = y.permute(0, 2, 1).reshape(1, groups, T1)  # [1, 672, 336]

                    # kernel: [groups, 1, kernel_size]
                    kernel_raw = torch.fft.ifft(torch.exp(1j * statistics_pred_P[1]), dim=-1)      # shape: [B, T, N]
                    kernel = torch.abs(kernel_raw).permute(0, 2, 1).reshape(groups, 1, T1)  # [672, 1, 336]
                    kernel = torch.flip(kernel, dims=[2])  # 卷积核要翻转

                    # 卷积：grouped conv
                    out = F.conv1d(input, kernel, groups=groups, padding=T1 // 2)[:,:,:self.pred_len]  # [1, 672, 336]

                    # reshape 回原 shape
                    y = out.view(B1, N1, T1) + statistics_pred_P[0]
                    # y = F.conv1d(y.transpose(-1, -2), torch.abs(torch.exp(1j*(statistics_pred_P[1]))), padding=T1//2) + statistics_pred_P[0] 
                    y = y.transpose(-1, -2)
                    statistics_pred_P1 = (statistics_pred_P[0], seq_s_y + statistics_pred_P[1])
                    statistics_pred_P1 = torch.cat(statistics_pred_P1, dim=1).transpose(-1, -2)
                    statistics_pred_P = (statistics_pred_P[0], statistics_pred_P[1])
                    statistics_pred_P = torch.cat(statistics_pred_P, dim=1).transpose(-1, -2)

                elif self.args.use_norm == 'sliding':
                    batch_x, statistics_pred_P, statistics_seq = self.statistics_pred_P.normalize(batch_x)
                   
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float()
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                y = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            y = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                if self.args.use_norm == 'sliding':
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                else:
                                    y = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    norm_y, (seq_m_y, seq_s_y) =self.norm_f(y.transpose(-1, -2))#返回预测数据及预测数据的均值与相位谱
                    
                    B1, T1, N1 = y.size()  # y: [B, T, N]
                    groups = B1 * N1       # 一共 672 个 group

                    # input: reshape to [1, groups, T]
                    input = y.permute(0, 2, 1).reshape(1, groups, T1)  # [1, 672, 336]

                    # kernel: [groups, 1, kernel_size]
                    kernel_raw = torch.fft.ifft(torch.exp(1j * (statistics_pred_P[1])), dim=-1)        # shape: [B, T, N]
                    kernel = torch.abs(kernel_raw).permute(0, 2, 1).reshape(groups, 1, T1)  # [672, 1, 336]
                    kernel = torch.flip(kernel, dims=[2])  # 卷积核要翻转

                    # 卷积：grouped conv
                    out = F.conv1d(input, kernel, groups=groups, padding=T1 // 2)[:,:,:self.pred_len]  # [1, 672, 336]

                    # reshape 回原 shape
                    y = out.view(B1, N1, T1) + statistics_pred_P[0]
                    y = y.transpose(-1, -2)
                    statistics_pred_P1 = (statistics_pred_P[0], seq_s_y + statistics_pred_P[1])
                    statistics_pred_P1 = torch.cat(statistics_pred_P1, dim=1).transpose(-1, -2)
                    statistics_pred_P = (statistics_pred_P[0], statistics_pred_P[1])
                    statistics_pred_P = torch.cat(statistics_pred_P, dim=1).transpose(-1, -2)
                    
                else:
                    batch_x, statistics_pred_P = self.statistics_pred_P.normalize(batch_x)
                    

                if  epoch + 1 <=  self.station_pretrain_epoch:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.features == 'MS':
                        statistics_pred_P1 = statistics_pred_P1[:, :, [self.args.enc_in - 1, -1]]
                    loss, loss_am = self.station_loss_P(batch_y, statistics_pred_P1, y, epoch)
                    train_loss.append(loss.item())
                    train_loss1.append(loss_am.item())
                    
                
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder x
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = self.criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                if self.args.use_norm == 'sliding':
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        if self.args.features == 'MS':
                            statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                            statistics_pred_P = statistics_pred_P[:, :, [self.args.enc_in - 1, -1]]
                        outputs = self.statistics_pred_P.de_normalize(outputs, statistics_pred_P)
                        
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = self.criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                            (self.args.train_epochs + self.station_pretrain_epoch - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    # two-stage training schema
                    if epoch + 1 <= self.station_pretrain_epoch:
                        self.station_optim_P.step()
                    else:
                        model_optim.step()
                    model_optim.zero_grad()
                    self.station_optim_P.zero_grad()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss2.append(np.average(train_loss1))
            

            vali_loss = self.vali(vali_data, vali_loader, self.criterion, epoch)
            test_loss = self.vali(test_data, test_loader, self.criterion, epoch)
                

            if  epoch + 1 <= self.station_pretrain_epoch:
                print(
                    "Station Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping_station_model_P(vali_loss, self.statistics_pred_P, path_station_P)
                adjust_learning_rate(self.station_optim_P, epoch + 1, self.args, self.args.station_lr)
                
            else:
                print(
                    "Backbone Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1 - self.station_pretrain_epoch, train_steps, train_loss, vali_loss, test_loss))
                # 若有更新之后stop,即保存
                if self.args.use_norm == 'sliding' and 0 <= self.args.twice_epoch <= epoch - self.station_pretrain_epoch:
                    early_stopping(vali_loss, self.model, path, self.statistics_pred_P, path_station_P)
                else:
                    early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                adjust_learning_rate(model_optim, epoch + 1 - self.station_pretrain_epoch, self.args,
                                     self.args.learning_rate)
                adjust_learning_rate(self.station_optim_P, epoch + 1 - self.station_pretrain_epoch, self.args,
                                     self.args.station_lr)
        np.save('/home/huyue/Torch/time_series_forecasting/phase_end/' + 'train_loss_mean.npy', train_loss2)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if self.args.use_norm == 'sliding' and self.args.twice_epoch >= 0:
            self.statistics_pred_P.load_state_dict(torch.load(path_station_P + '/' + 'checkpoint.pth'))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = '/home/huyue/Torch/time_series_forecasting/phase_end/test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.statistics_pred_P.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                input_x = batch_x

                if self.args.use_norm == 'sliding':
                    batch_x, statistics_pred_P, statistics_seq = self.statistics_pred_P.normalize(batch_x)
                    statistics_pred_P = torch.cat(statistics_pred_P, dim=1).transpose(-1, -2)
                    
                else:
                    batch_x, statistics_pred_P, statistics_seq = self.statistics_pred_P.normalize(batch_x)
                    statistics_pred_P = torch.cat(statistics_pred_P, dim=1).transpose(-1, -2) 

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder x
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_label = batch_x[:, -self.args.label_len:, :]
                dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                if self.args.use_norm == 'sliding':
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            if self.args.use_norm == 'sliding':
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if self.args.features == 'MS':
                    statistics_pred_P = statistics_pred_P[:, :, [self.args.enc_in - 1, -1]]
                outputs = self.statistics_pred_P.de_normalize(outputs, statistics_pred_P)
                
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 5 == 0:
                    x = input_x.detach().cpu().numpy()
                    gt = np.concatenate((x[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((x[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # if self.args.test_flop:
        #     test_params_flop((batch_x.shape[1], batch_x.shape[2]))
        #     exit()
        preds = np.array(preds, dtype=object)
        trues = np.array(trues, dtype=object)
        inputx = np.array(inputx, dtype=object)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        folder_path = '/home/huyue/Torch/time_series_forecasting/phase_end/test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds_phase = norm_phase(preds)
        trues_phase = norm_phase(trues)
        mae_phase, mse_phase, rmse, mape, mspe, rse, corr = metric(preds_phase, trues_phase)
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, mse_phase:{}, mae_phase:{}'.format(mse, mae, mae_phase, mse_phase))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x.npy', inputx)
        return mse, mae

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder x
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
