# Modality Projection from Vision to Language
import torch.nn as nn

class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg          = cfg
        self.input_dim    = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor**2)
        self.output_dim   = cfg.lm_hidden_dim
        self.scale_factor = cfg.mp_pixel_shuffle_factor
        
        self.proj         = nn.Linear(self.input_dim, self.output_dim, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    # 간단 알고리즘 - https://huggingface.co/blog/nanovlm 안의 그림 참조 
    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()                 # x의 shape을 (batch_size, seq_len, embed_dim)으로 받음 # 입력 DIMENSION
        seq_root            = int(seq**0.5)            # seq_len의 제곱근을 구함 (이미지의 height/width로 사용)
        
        assert seq_root**2 == seq                      # seq_len이 완전 제곱수여야 함 (정사각형 형태로 reshape 가능해야 함)
        assert seq_root % self.scale_factor == 0       # scale_factor로 나눠떨어져야 함 (pixel shuffle을 위한 조건)

        height = width = seq_root                      # height, width 모두 seq_root로 설정 (정사각형)
        x      = x.view(bsz, height, width, embed_dim) # x를 (batch_size, height, width, embed_dim)로 reshape
        
        h_out  = height // self.scale_factor           # scale_factor로 나눈 최종 출력 height 계산
        w_out  = width // self.scale_factor            # scale_factor로 나눈 최종 출력 width 계산

        # (batch_size, h_out, scale, w_out, scale, embed_dim)로 재배치
        #    → height, width 양쪽 모두 scale_factor 만큼 쪼갬
        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim)

        # scale factor 축들을 인접하게 배치 (permutation)
        #    → (batch, h_out, w_out, scale, scale, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        
        # scale_factor^2 만큼 embed_dim을 늘려서,
        #    → (batch_size, h_out * w_out, embed_dim * scale_factor^2)로 reshape # 최종 DIMENSION
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2)
        
        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)

        return x

    
