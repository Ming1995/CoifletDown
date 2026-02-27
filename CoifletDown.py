import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

def get_coiflet(order:int =1, device=None, dtype=torch.float32, which:str = 'dec'):
    wave = pywt.Wavelet(f"coif{order}")
    if which == 'dec':
        lo, hi = wave.dec_lo, wave.dec_hi
    elif which == 'rec':
        lo, hi = wave.rec_lo, wave.rec_hi
    else:
        raise ValueError('which 只能是 "dec" 或 "rec"')
    
    dec_lo = torch.tensor(lo, dtype=dtype, device=device)
    dec_hi = torch.tensor(hi, dtype=dtype, device=device)
    return dec_lo, dec_hi

class CoifletDown(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out:int ,
                 order: int = 1,
                 keep_4x:bool = False,
                 bias:bool = False,
                 bn: bool = True,
                 act: bool = True,
                 eps: float = 1e-5,
                 trainable:bool = True):
        super().__init__()
        assert c_in > 0, "c_in must > 0"
        self.c_in = c_in
        self.keep_4x = keep_4x

        # 1 取得一維濾波器係數作外積得到4個2D小波核
        dec_lo, dec_hi = get_coiflet(order=order)
        new_lo = dec_lo.view(1, -1)
        new_hi = dec_hi.view(1, -1)
        kLL = (new_lo.t() @ new_lo) # LL
        kLH = (new_lo.t() @ new_hi) # LH
        kHL = (new_hi.t() @ new_lo) # HL 
        kHH  =(new_hi.t() @ new_hi) # HH

        # 2 準備depthwise 權重:(4*C, 1, K, K)
        k = kLL.shape[0]
        base = torch.stack([kLL, kLH, kHL, kHH], dim=0).unsqueeze(1)
        weight = base.repeat(c_in, 1, 1, 1)
        

        # 2-1) 新增開關:trainable(加入到__init__參數列且預設False)
        if trainable:
            self.dw_weight = nn.Parameter(weight, requires_grad=True)
        else:
            self.register_buffer("dw_weight", weight, persistent=True)

        self.stride = 2
        self.kernel_size = k  # 用於 forward 計算 padding
        
        # 融合頭(可選)
        if not keep_4x:
            assert c_out is not None and c_out > 0, "keep_4x=False 時必須指定 c_out"
            layers = [nn.Conv2d(4 * c_in, c_out, kernel_size=1, bias=bias)]
            if bn:
                layers.append(nn.BatchNorm2d(c_out, eps=eps))
            if act:
                layers.append(nn.SiLU(inplace=True))
            self.fuse = nn.Sequential(*layers)
        else:
            self.fuse = nn.Identity()
    
    def _reflect_pad_for_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        為了保證 out = H/2，採用「總 padding = k - 2」的反射邊界：
        左右：(l, r)；上下：(t, b)
        若 k 為偶數 → (l,t)=(k-2)//2，(r,b)=(k-2)-(l,t)
        """
        k = self.kernel_size
        pad_total = max(k - 2, 0)
        l = pad_total // 2
        r = pad_total - l
        t = l
        b = r
        return F.pad(x, (l, r, t, b), mode="reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert c == self.c_in, f"Input channels {c} != expected {self.c_in}"

        x_pad = self._reflect_pad_for_half(x)
        y = F.conv2d(
            x_pad,
            self.dw_weight,
            bias=None,
            stride=self.stride,
            padding=0,
            groups=self.c_in
        )  # => B × (4C) × H/2 × W/2

        return self.fuse(y)

    @torch.no_grad()
    def export_kernels(self) -> torch.Tensor:
        """回傳未展開通道前的四個 2D 核（LL/LH/HL/HH），形狀 4×k×k。"""
        return self.dw_weight[:4, 0, :, :].clone()
