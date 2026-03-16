import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWT1DInverse, DWT1DForward
# from statsmodels.tsa.stattools import adfuller
from utils import *
from utils.ADF import ad_fuller as adf
from utils.learnable_wavelet import DWT1D
from utils.TCN import TemporalConvNet
from utils.S_Mamba import Mamba_Model
from layers.PatchTST_layers import series_decomp

from scipy.signal import hilbert

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

def gaussian_weight(size, sigma=1):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def exponential_smoothing(data, alpha):
    smoothed_data = data.clone()
    for t in range(1, data.shape[-1]):
        smoothed_data[..., t] = alpha * data[..., t] + (1 - alpha) * smoothed_data[..., t - 1]
    return smoothed_data


class Transformer1D(nn.Module):
    def __init__(self, seq_len, input_dim, d_model=512, nhead=16, num_layers=1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: (B, T, C)
        x = self.embedding(x.permute(0,2,1))
        x = self.transformer(x)
        return self.output_layer(x).permute(0,2,1)

def torch_unwrap(p, dim=-1):
    # 计算相邻差值
    diff = torch.diff(p, dim=dim)
        
    # 2π跳变位置
    correction = torch.remainder(diff + np.pi, 2 * np.pi) - np.pi

    # 特殊处理边界：不修改 ±π 跳变
    correction = correction - diff
    correction = torch.where(torch.abs(diff) < np.pi, torch.zeros_like(diff), correction)

    # 累加展开量
    phase_unwrapped = torch.cumsum(torch.cat((p.narrow(dim, 0, 1), correction), dim=dim), dim=dim)
    return phase_unwrapped

class SkipUNet1D(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 num_channels_down=[8, 16, 32, 64, 128],
                 num_channels_up=[8, 16, 32, 64, 128],
                 num_channels_skip=[0, 0, 0, 4, 4],
                 need_sigmoid=True,
                 need_bias=True,
                 pad='zero',
                 act_fun='LeakyReLU'):
        super(SkipUNet1D, self).__init__()

        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
        self.n_scales = len(num_channels_down)
        self.need_sigmoid = need_sigmoid

        self.encoders = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()

        input_depth = in_channels

        # Encoder path with skip connections
        for i in range(self.n_scales):
            self.encoders.append(self._block(input_depth, num_channels_down[i], act_fun, pad, bias=need_bias))

            # ✅ 修正：使用 encoder 输出通道构建 skip 分支
            if num_channels_skip[i] != 0:
                self.skips.append(self._skip_block(num_channels_down[i], num_channels_skip[i], act_fun, pad, bias=need_bias))
            else:
                self.skips.append(None)

            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            input_depth = num_channels_down[i]

        # Bottleneck
        self.bottleneck = self._block(num_channels_down[-1], num_channels_down[-1], act_fun, pad, bias=need_bias)

        # Decoder path
        for i in reversed(range(self.n_scales)):
            up_in_ch = num_channels_down[-1] if i == self.n_scales - 1 else num_channels_up[i + 1]
            up_out_ch = num_channels_up[i]
            self.upsamples.append(nn.ConvTranspose1d(up_in_ch, up_out_ch, kernel_size=2, stride=2))

            skip_ch = num_channels_skip[i]
            enc_ch = num_channels_down[i]
            decoder_in_ch = up_out_ch + enc_ch + skip_ch  # ✅ 合理拼接通道数
            self.decoders.append(self._block(decoder_in_ch, num_channels_up[i], act_fun, pad, bias=need_bias))

        self.final_conv = nn.Conv1d(num_channels_up[0], out_channels, kernel_size=1)
        if self.need_sigmoid:
            self.final = nn.Sigmoid()

    def _block(self, in_channels, out_channels, act_fun, pad, bias=True):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            self._get_act(act_fun),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            self._get_act(act_fun)
        )

    def _skip_block(self, in_channels, out_channels, act_fun, pad, bias=True):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias),
            self._get_act(act_fun)
        )

    def _get_act(self, act_fun):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(inplace=True)
        elif act_fun == 'ReLU':
            return nn.ReLU(inplace=True)
        elif act_fun == 'ELU':
            return nn.ELU(inplace=True)
        elif act_fun == 'GELU':
            return nn.GELU()
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        enc_features = []
        skip_features = []

        # Encoder
        for i in range(self.n_scales):
            x = self.encoders[i](x)
            enc_features.append(x)

            if self.skips[i] is not None:
                skip_features.append(self.skips[i](x))
            else:
                skip_features.append(None)

            x = self.pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in reversed(range(self.n_scales)):
            upsample = self.upsamples[self.n_scales - 1 - i]
            x = upsample(x)

            enc = enc_features[i]
            if x.shape[-1] != enc.shape[-1]:
                enc = F.interpolate(enc, size=x.shape[-1], mode='linear', align_corners=True)

            parts = [x, enc]
            if skip_features[i] is not None:
                skip = skip_features[i]
                if skip.shape[-1] != x.shape[-1]:
                    skip = F.interpolate(skip, size=x.shape[-1], mode='linear', align_corners=True)
                parts.append(skip)

            x = torch.cat(parts, dim=1)
            x = self.decoders[self.n_scales - 1 - i](x)

        x = self.final_conv(x)
        if self.need_sigmoid:
            x = self.final(x)
        return x



class APN(nn.Module):
    def __init__(self, configs):
        super(APN, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.kernel = kernel = configs.kernel_len
        self.hkernel = hkernel = configs.hkernel_len
        self.pad = nn.ReplicationPad1d(padding=(kernel // 2, kernel // 2 - ((kernel + 1) % 2)))
        if hkernel is not None:
            self.hpad = nn.ReplicationPad1d(padding=(hkernel // 2, hkernel // 2 - ((hkernel + 1) % 2)))
        self.channels = configs.enc_in if configs.features == 'M' else 1
        self.station_type = configs.station_type
        self.seq_len_new = self.seq_len
        self.pred_len_new = self.pred_len
        self.epsilon = 1e-5
        self._build_model()

    def _build_model(self):
        args = copy.deepcopy(self.configs)
        args.seq_len = self.configs.seq_len
        args.pred_len = self.configs.pred_len
        args.label_len = self.configs.label_len
        args.enc_in = self.configs.enc_in
        args.dec_in = self.configs.dec_in
        args.data_path = self.configs.data_path
        args.moving_avg = 3

        args.c_out = self.configs.c_out
        self.norm_func = self.norm_sliding
        self.kernel_size = self.configs.kernel_size

        wave = self.configs.wavelet
        wave_dict = {'coif6': 16, 'coif3': 8, 'sym3': 2, 'bior3.5':16, 'db20':12}
        self.len, self.j = wave_dict[wave], self.configs.j
        self.dwt = DWT1D(wave=wave, J=self.j, learnable=args.learnable)
        self.dwt_ratio = nn.Parameter(
            torch.clamp(torch.full((1, self.channels, 1), 0.), min=0., max=1.)
        )
        self.mlp = Statics_MLP(
            self.configs.seq_len, args.pd_model, args.pd_ff, args.enc_in, args.data_path,
            self.configs.pred_len, drop_rate=args.dr, layer=args.pe_layers
        )

    def normalize(self, x, p_value=True):
        if self.station_type == 'adaptive':
            norm_input, seq_ms, pred_ms = self.norm(
                x=x.transpose(-1, -2)
            )
            # y = self.project_h(norm_input)
            # norm_y, (seq_m_y, seq_s_y) = self.norm_func(y)#返回预测数据及预测数据的均值与相位谱
            # pred_ms = (pred_ms[0], seq_s_y+pred_ms[1])
            # y = y* torch.abs((torch.exp(1j*pred_ms[1]))) + pred_ms[0]
            
            # outputs = torch.cat(pred_ms, dim=1).transpose(-1, -2)
            return norm_input.transpose(-1, -2), pred_ms, seq_ms
            # return norm_input.transpose(-1, -2), outputs, seq_ms, y.transpose(-1, -2) #返回时域与重建后的历史数据，返回差的相位谱 返回历史数据的均值与相位谱
        else:
            return x, None

    def de_normalize(self, input, station_pred):
        if self.station_type == 'adaptive':
            bs, l, dim = input.shape
            mean = station_pred[..., :station_pred.shape[-1] // 2]
            std = station_pred[..., station_pred.shape[-1] // 2:]
            # output = input * (mean + self.epsilon) * torch.abs((torch.exp(1j*std)))
            # output = (input ) * torch.abs((torch.exp(1j*(std))+ self.epsilon) )+ mean
            B1, T1, N1 = input.size()  # y: [B, T, N]
            groups = B1 * N1       # 一共 672 个 group

            # input: reshape to [1, groups, T]
            input = input.permute(0, 2, 1).reshape(1, groups, T1)  # [1, 672, 336]

                    # kernel: [groups, 1, kernel_size]
            kernel_raw = torch.fft.ifft(torch.exp(1j * std.transpose(-1, -2)), dim=-1).transpose(-1, -2)       # shape: [B, T, N]
            kernel = torch.abs(kernel_raw).permute(0, 2, 1).reshape(groups, 1, T1)  # [672, 1, 336]
            kernel = torch.flip(kernel, dims=[2])  # 卷积核要翻转

                    # 卷积：grouped conv
            out = F.conv1d(input, kernel, groups=groups, padding=T1 // 2)[:,:,:self.pred_len]  # [1, 672, 336]

                    # reshape 回原 shape
            output = out.view(B1, N1, T1).permute(0, 2, 1) + mean
                    
            # output = F.conv1d(input.transpose(-1, -2), m_t_flipped, padding=padding)[:,:,:self.pred_len]  +  mean

            # output = F.conv1d(input.transpose(-1, -2), torch.abs(torch.exp(1j*(std))), padding=T1//2)[:,:,:self.pred_len].transpose(-1, -2)
            # output = input * (mean + self.epsilon) 
            return output.reshape(bs, l, dim)
        else:
            return input

    def norm(self, x, predict=True):
        norm_x, (seq_m, seq_s) = self.norm_func(x) #返回历史数据及历史数据的均值与相位谱
        if predict is True:
            mov_m, mov_s = self.mlp(seq_m, seq_s, x)#返回预测数据的均值与相位谱
            if self.j > 0:
                ac, dc_list = self.dwt(x)
                norm_ac, (mac, sac) = self.norm_func(ac, kernel=self.hkernel)
                norm_dc, m_list, s_list = [], [], []
                for i, dc in enumerate(dc_list):
                    dc, (mdc, sdc) = self.norm_func(dc, kernel=self.hkernel)
                    norm_dc.append(dc)
                    m_list.append(mdc)
                    s_list.append(sdc)
                
                pred_m, pred_s = self.mlp(
                    self.dwt([mac, m_list], 1),
                    self.dwt([sac, s_list], 1), self.dwt([ac, dc_list], 1))#返回预测数据频率域的均值和相位谱

                dwt_r, mov_r = self.dwt_ratio, 1 - self.dwt_ratio
                norm_x = norm_x * mov_r + self.dwt([norm_ac, norm_dc], 1) * dwt_r
                pred_m = mov_m * mov_r + pred_m * dwt_r
                pred_s = mov_s * mov_r + pred_s * dwt_r
         
                return norm_x, (seq_m, seq_s), (pred_m, pred_s)
            else:
                return norm_x, (seq_m, seq_s), (mov_m, mov_s)
        return norm_x, (seq_m, seq_s)
    
    # def norm_sliding(self, x, kernel=None):
    #     if kernel is None:
    #         kernel, pad = self.kernel, self.pad
    #     else:
    #         pad = self.hpad
    #     x_window = x.unfold(-1, kernel, 1)  # sliding window
    #     m, s = x_window.mean(dim=-1), torch.angle(torch.fft.fft(x_window, dim=-1)).mean(dim=-1) + torch.pi # acquire sliding mean and sliding standard deviation
    #     m, s = pad(m), pad(s)  # nn.ReplicationPad1d(padding=(kernel // 2, kernel // 2 - ((kernel + 1) % 2)))
    #     # x = (x - m) / (s + self.epsilon)  # x is stationary series
        
    #     # F_w = torch.fft.fft(x, dim=2)
    #     # A_f = torch.abs(F_w)
    #     # # A_f = torch.where(A_f < 10, torch.zeros_like(A_f), A_f)
    #     # phi_f = torch.angle(F_w) + torch.pi
    #     # eps = 10 + 1e-8
    #     # phi_f = torch_unwrap(torch.where(A_f > eps, phi_f, torch.zeros_like(phi_f)), dim=2)
    #     x = (x - m) 
    #     # phi_f = torch.where(A_f < 10, torch.zeros_like(phi_f), phi_f)
    #     return x, (m, s)  

    def norm_sliding(self, x, kernel=None):
        if kernel is None:
            kernel, pad = self.kernel, self.pad
        else:
            pad = self.hpad
        
        # res_init, trend_init = self.decomp_module(x.permute(0,2,1))
        # x_window = trend_init.permute(0,2,1).unfold(-1, kernel, 1)  # sliding window
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

    def p_value(self, x, float_type=torch.float32):
        B, ch, dim = x.shape
        p_value = adf(x.reshape(-1, dim), maxlag=min(self.kernel, 24), float_type=float_type).view(B, ch, 1)
        return p_value


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, activation, drop_rate=0.1, bias=False):
        super(FFN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias), activation,
            nn.Linear(d_ff, d_model, bias=bias), nn.Dropout(drop_rate),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class Statics_MLP(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, enc_in, data_path,
                 pred_len, drop_rate=0.1, bias=False, layer=1):
        super(Statics_MLP, self).__init__()
        project = nn.Sequential(nn.Linear(seq_len, d_model, bias=bias), nn.Dropout(drop_rate))
        self.m_project, self.s_project = copy.deepcopy(project), copy.deepcopy(project)
        self.mean_proj, self.std_proj = copy.deepcopy(project), copy.deepcopy(project)
        self.mean_proj1 = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(drop_rate))
        self.m_concat = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Dropout(drop_rate))
        self.s_concat = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Dropout(drop_rate))
        # ffn = skip(
        #     1, 1, 
        #     num_channels_down = [8, 16, 32, 64, 128], 
        #     num_channels_up   = [8, 16, 32, 64, 128],
        #     num_channels_skip = [0, 0, 0, 4, 4], 
        #     upsample_mode='bilinear',
        #     need_sigmoid=False, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        
        # ffn = SkipUNet1D(
        #                     in_channels=1,
        #                     out_channels=1,
        #                     num_channels_down=[8, 16, 32, 64, 128],
        #                     num_channels_up=[8, 16, 32, 64, 128],
        #                     num_channels_skip=[0, 0, 0, 4, 4],
        #                     need_sigmoid=False,
        #                     pad='reflection',
        #                     act_fun='GELU'
        #                     )
        if 'traffic' in data_path:
            num_channels = [512, 1024,1024, 512, enc_in]
            
        elif 'elec' in data_path:
            num_channels = [256, 512, 1024, 512, enc_in]
            
        elif 'wea' in data_path:
            num_channels = [32, 64, 32, enc_in]  
              
        else:
            num_channels = [16, 32, 64, 32, enc_in]

        
        # context_window = seq_len
        # self.stride = 24
        # self.patch_len = 48
        # patch_num = int((context_window - self.patch_len)/self.stride + 1)
        # self.padding_patch = 'end'
        # if self.padding_patch == 'end': 
        #     self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        #     patch_num += 1

        ffn = TemporalConvNet(num_inputs= enc_in, num_channels =  num_channels, kernel_size=3, dropout=drop_rate)
        # d_state = 2
        # d_ff1 = 256
        # e_layers = 2
        # dropout1 = 0.1
        # activation1 = 'gelu'
        # ffn1 = Mamba_Model(862, d_state, d_ff1, e_layers, dropout1, activation1)
        # ffn = Transformer1D(input_dim=patch_num, seq_len=d_model)
        ffn1 = nn.Sequential(*[FFN(d_model, d_ff, nn.LeakyReLU(), drop_rate, bias) for _ in range(layer)])
        self.mean_ffn, self.std_ffn = copy.deepcopy(ffn1), copy.deepcopy(ffn)
        self.mean_pred = nn.Linear(d_model, pred_len, bias=bias)
       
        # dd = d_model * patch_num
        self.std_pred = nn.Linear(d_model, pred_len, bias=bias)

        # self.W_P = nn.Linear(self.patch_len, d_model)  
        # self.flatten = nn.Flatten(start_dim=-2)
        self.down_sampling_window = 2
        self.down_sampling_layers = 3
        self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        seq_len // (self.down_sampling_window ** i),
                        pred_len,
                    )
                    for i in range(self.down_sampling_layers + 1)
                ]
            )

    def forward(self, mean, std, x=None, x2=None):
        
        # x_enc = self.__multi_scale_process_inputs(std)
        # x_list = []
        # x_mean_list = []
        # for i, x1 in zip(range(len(x_enc)), x_enc):
        #     x_mean_list.append(std.mean(dim=-1, keepdim=True))
        #     x_list.append(x1)

        # m_all, s_all = mean.mean(dim=-1, keepdim=True), std.mean(dim=-1, keepdim=True)
        # mean_r, std_r = mean - m_all, std 
        # mean_r, std_r = self.mean_proj(mean_r), self.std_proj(std_r)
        # if x is not None:
        #     m_orig, s_ori = self.m_project(x - m_all), \
        #         self.s_project(torch.abs(torch.fft.fft(x)) if x2 is None else x2 - s_all)
        #     mean_r, std_r1 = self.m_concat(torch.cat([m_orig, mean_r], dim=-1)), \
        #         self.s_concat(torch.cat([s_ori, std_r], dim=-1))
        
        # B1, T1, N1 = std_r.size()
        # # mean_r = mean_r.reshape(B1 * T1, N1, 1).permute(0, 2, 1)
        # # std_r = std_r.reshape(B1 * T1, N1, 1).permute(0, 2, 1)
        # mean_r, std_r = self.mean_ffn(mean_r), self.std_ffn(std_r)
        # dec_out_list = []
        # for i, enc_out in zip(range(len(x_list)), x_list):
        #         # enc_out = self.std_ffn(enc_out)
        #         dec_out = self.predict_layers[i](enc_out)  # align temporal dimension
        #         # if self.use_future_temporal_feature:
        #         #     dec_out = dec_out + self.x_mark_dec
        #         #     dec_out = self.projection_layer(dec_out)
        #         # else:
        #         #     dec_out = self.projection_layer(dec_out)
        #         # dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
        #         dec_out_list.append(dec_out)
        # dec_out = torch.stack(dec_out_list, dim=-1).sum(-1) 
        # # mean_r = mean_r.reshape(B1, T1, N1)
        # # std_r = std_r.reshape(B1, T1, N1)
        # mean_r, std_r = self.mean_pred(mean_r), self.std_pred(std_r)

        # mean, std = mean_r + m_all, std_r 

        m_all, s_all = mean.mean(dim=-1, keepdim=True), std.mean(dim=-1, keepdim=True)
        mean_r, std_r = mean - m_all, std - s_all
        mean_r, std_r = self.mean_proj(mean_r), self.std_proj(std_r)
        if x is not None:
            m_orig, s_ori = self.m_project(x - m_all), \
                self.s_project(torch.abs(torch.fft.fft(x)) if x2 is None else x2 - s_all)
            mean_r, std_r1 = self.m_concat(torch.cat([m_orig, mean_r], dim=-1)), \
                self.s_concat(torch.cat([s_ori, std_r], dim=-1))
        
        B1, T1, N1 = std_r.size()
        # mean_r = mean_r.reshape(B1 * T1, N1, 1).permute(0, 2, 1)
        # std_r = std_r.reshape(B1 * T1, N1, 1).permute(0, 2, 1)
        mean_r, std_r = self.mean_ffn(mean_r), self.std_ffn(std_r)
        # mean_r = mean_r.reshape(B1, T1, N1)
        # std_r = std_r.reshape(B1, T1, N1)
        mean_r, std_r = self.mean_pred(mean_r), self.std_pred((std_r))

        mean, std = mean_r + m_all, std_r + s_all
      
        # if self.padding_patch == 'end': 
        #     z = self.padding_patch_layer(std)
        # stride = 8
        # patch_len = 16
        # z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        # n_vars = z.shape[1]

        #                                             # x: [bs x nvars x patch_num x patch_len]
        # # x = self.W_P(z)                                                          # x: [bs x nvars x patch_num x d_model]
        # u = torch.reshape(z, (z.shape[0]*z.shape[1],z.shape[2],z.shape[3]))

        # m_all, s_all = mean.mean(dim=-1, keepdim=True), u.mean(dim=-1, keepdim=True)
        # mean_r, std_r = mean - m_all, u 
        # mean_r, std_r = self.mean_proj(mean_r), self.W_P(std_r) 
        # if x1 is not None:
        #     m_orig, s_ori = self.m_project(x1 - m_all), \
        #         self.s_project(x1 if x2 is None else x2 - s_all)
        #     mean_r = self.m_concat(torch.cat([m_orig, mean_r], dim=-1))
        
        # B1, T1, N1 = std_r.size()
        # # mean_r = mean_r.reshape(B1 * T1, N1, 1).permute(0, 2, 1)
        # # std_r = std_r.reshape(B1 * T1, N1, 1).permute(0, 2, 1)
        # mean_r, std_r = self.mean_ffn(mean_r), self.std_ffn(std_r) 
        # z = torch.reshape(std_r, (-1,n_vars,std_r.shape[-2],std_r.shape[-1]))
        # z = z.permute(0,1,3,2)
        # z = self.flatten(z)
        # # mean_r = mean_r.reshape(B1, T1, N1)
        # # std_r = std_r.reshape(B1, T1, N1)
        # mean_r, std_r = self.mean_pred(mean_r), self.std_pred(z)

        # mean, std = mean_r + m_all, std_r 

        return mean,  std
        
    def __multi_scale_process_inputs(self, x_enc):

        down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        
        # B,T,C -> B,C,T
        x_enc = x_enc
        x_enc_ori = x_enc
        
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc)
    
        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling)
            x_enc_ori = x_enc_sampling

        x_enc = x_enc_sampling_list

        return x_enc

def normalization(x, mean=None, std=None):
    if mean is not None and std is not None:
        return (x - mean) / std
    mean = x.mean(-1, keepdim=True).detach()
    x = x - mean
    std = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
    x /= std
    return x, mean, std
