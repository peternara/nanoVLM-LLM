import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L69
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Normalizes the input across the last dimension using RMS normalization,
    which scales the input without subtracting the mean. Commonly used as a
    lighter alternative to LayerNorm in transformer models.

    Args:
        cfg: A configuration object containing:
            - lm_hidden_dim (int): The dimensionality of the model hidden states. 
            - lm_rms_eps (float): A small constant to avoid division by zero.
    """
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.lm_hidden_dim))
        self.eps = cfg.lm_rms_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, lm_hidden_dim).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Compute inverse of RMS: square the tensor element-wise, mean is computed across lm_hidden_dim.
        irms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps) # inverse of RMS
        x = x * irms * self.weight

        return x

# Multiple derivates of Rotary Embeddings by now, this is a basic one with linear scaling to context length
# e.g. https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L190
#
# RoPE (Rotary Positional Embedding)
#    → RoPE는 Llama, GPT-NEOX 등 최신 트랜스포머에서 기본 positional embedding 사용
#    → 입력 시퀀스(토큰)의 각 위치에 대해, 학습이 필요없는(학습파라메터가 필요없음) 사인/코사인 함수를 기반으로 하는 position embedding을 만들어 Q/K 벡터에 position 정보를 더하는 데 사용
#    → 입력 시퀀스의 위치(position)에 따라 각 차원의 임베딩을 사인/코사인 곡선으로 회전(rotary)시켜, 트랜스포머의 Q/K에 위치 정보를 삽입. 
#    → 더 긴 시퀀스에서도 position 정보가 잘 보존
#    상세)
#    → 각 토큰 위치와 차원별 주파수를 곱해, 
#    → 각 위치에 대해 sin/cos 곡선을 생성
#    → 이 패턴으로 Q/K 임베딩을 "회전"하여,
#    → 위치와 상대적 거리가 attention score에 자연스럽게 반영되도록 만듦   
class RotaryEmbedding(nn.Module):
    """
        Compute Rotary Embedding to introduce positional dependency to input sequence without additional training parameters and 
        relative distance of token position ids through angle rotation.

        Args:
            cfg: Configuration object containing:
                - lm_hidden_dim (int): Hidden dimension size.
                - lm_n_heads (int): Number of attention heads.
                - lm_re_base (float): Base for rotary embedding frequencies.
                - lm_max_position_embeddings (int): Max sequence length supported for rotary embedding.
                - lm_attn_scaling (float): Attention scaling factor.
        """
    
    def __init__(self, cfg):
        super().__init__()
        assert cfg.lm_hidden_dim % cfg.lm_n_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.dim         = cfg.lm_hidden_dim // cfg.lm_n_heads # dim of each head, # 한 헤드의 임베딩 차원
        self.base        = cfg.lm_re_base                      # RoPE에서 쓸 주파수 베이스 (default=10000)
        self.max_seq_len = cfg.lm_max_position_embeddings      # RoPE가 지원하는 최대 길이
        
        # Standard RoPE implementation - create frequencies for each dimension
        # freq_i = 1 / (base^(2i/dim)) where i is the dimension index
        #
        # i) RoPE의 주파수 텐서 준비: [0, 2, 4, ..., dim-2] (짝수 차원만 사용)
        # ii) freq_i = 1/(base^(2i/dim))
        # iii) 각 차원(dimension)마다 사용하는 회전 주파수(frequency)를 계산하는 부분
        #     → RoPE는 각 차원별로 “회전(rotary)”시키는 각도의 속도(=주파수)가 다름
        #     → 저차원(앞쪽 차원)은 천천히 변화,
        #     → 고차원(뒤쪽 차원)은 빨리 변화
        #     → 이렇게 하면 각 토큰의 위치를 여러 스케일로 encode 가능
        # iiii) 아래 코드는 RoPE 주파수 공식 : freq_i = 1/base^(2i/d)
        #     → i: 짝수 (0,2,,)
        #        → 왜 짝수만?
        #            → 2차원씩 짝을 지어 하나는 cos(𝜃), 하나는 sin(𝜃)
        #            → 복소수 평면(Real(실수), Imaginary(허수))처럼 사용해서
        #            → "복소수 회전” 형태로 position 정보를 주입
        #            → 예)
        #                → embedding = [x0​,x1​,x2​,x3​,x4​,x5​,...] 
        #                → (x₀, x₁), (x₂, x₃), (x₄, x₅) … 이렇게 두 개씩 짝을 지어 한 쌍이 한 ‘복소수 좌표’ 역할
        #                    → 짝수 > Real      > 0번째, 2번째, 4번째, ... → 각 쌍의 실수부
        #                    → 홀수 > Imaginary > 1번째, 3번째, 5번째, ... → 각 쌍의 허수부
        #                → 즉, 주파수(frequency)는 두개의 짝수 차원(예:(x₀, x₁)) 만큼 필요 (주의!! 두개가 필요하다는 의미임, 짝수만 필요하다는게 아님)
        #                → 실제로 sin/cos을 곱해서, 짝수 차원에는 cos, 그 다음(짝수+1) 차원에는 sin 이렇게 interleave(교차)하게 만듦.         
        #     → d: 한 헤드의 차원수 (self.dim)        
        #     → base: 주로 10000, 100000 등
        #     → ex 결과: d=8, dim=10000 이면 tensor([1, 4.6416, 2.1544, 0.1]) 
        #         → 0: 1/1 = 1.0
        #         → 2: 1/(10000^(2/8)) = 0.46416
        #         → 4: 1/(10000^(4/8)) = 0.21544
        #         → 6: 1/(10000^(6/8)) = 0.1
        # V) 이걸 어떻게 적용? → RoPE에서는 각 토큰 위치(pos)와 이 inv_freq(주파수)를 곱해 
        #     → inv_freq은 각 임베딩 차원이 변하는 “속도”를 조절하는 값
        #     → angle = pos×freq_i
        #     → 이 각도로 Q/K 벡터를 회전(sin/cos)해서 포지션 정보를 주입
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        self.register_buffer("inv_freq", inv_freq)
        self.original_max_seq_len = cfg.lm_max_position_embeddings
        self.attention_scaling    = cfg.lm_attn_scaling             # (옵션) attention scaling (보통 1.0)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary positional embeddings (cosine and sine components).

        Args:
            position_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing position indices.
                                       : (batch, seq_len) - 각 토큰의 위치 인덱스 (0~N)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of two tensors (cos, sin), each of shape > (batch_size, seq_len, dim), representing rotary embeddings.
                                             : cos, sin > 각 (batch, seq_len, dim) - RoPE에서 사용할 cos, sin 패턴
        """

        batch_size, seq_len = position_ids.shape
        
        # Dynamic scaling for longer sequences
        # Divide the angle frequency to fit more rotation into the embedding space.
        #
        # RoPE의 범위를 넘어서는 긴 시퀀스에 대해서 동적으로 scaling
        #     → RoPE가 시퀀스 길이가 “학습/설계시의 최대 길이”보다 더 길 때 회전 주파수(frequency)를 "스케일링"해서, 포지션 정보를 뭉개지지 않게 보존하는 트릭
        #     → position_ids: (batch, seq_len) tensor, 각 토큰의 position(0, 1, 2, ...) 
        #     → max_seq: 입력받은 전체 배치 중 가장 큰 position + 1 
        #     → 예) position_ids.max()가 511이면, max_seq=512 즉, 현재 들어온 데이터에서 "가장 긴 시퀀스 길이"를 의미
        #     → 그런데, 왜 이러한 구성을 하는가?
        #         → RoPE의 한계 RoPE의 주파수(inv_freq)는 처음 모델을 설계할 때 최대 시퀀스 길이(lm_max_position_embeddings(ex: 2048, 4096, 8192 등))에 제약되어 생성.
        #         → 그런데, 실제 inference/training에서 더 긴 시퀀스(예: 4096 → 6000)를 넣으면,
        #         → 포지션 각도(θ=pos×freq)가 너무 커져서 sin/cos이 너무 빨리 주기적으로 반복 
        #         → "포지션 구분이 안 되고 겹침(알리아싱/포지션 정보 소실)"
        #     → 그래서, 들어온 시퀀스가 기존 한계보다 길면, 주파수(inv_freq)를 '줄여서' 더 넓은 구간(더 느린 속도)로 펼침
        max_seq = position_ids.max() + 1
        if max_seq > self.original_max_seq_len:             # 예) 6000 > 4096  
            scale    = max_seq / self.original_max_seq_len  #    scale = 6000 / 4096 ≈ 1.46
            inv_freq = self.inv_freq / scale                #    기존 주파수(inv_freq)를 1.46배로 "느리게" 만들어줌 
                                                            # 즉, position_ids × (inv_freq/scale)
                                                            #    → pos가 6000이 들어와도
                                                            #    → max_pos 4096에 맞는 각도 범위에 압축
                                                            # 결과적으로, 포지션 정보가 “너무 빠르게 반복되지 않도록” 각도를 좁힘
        else:
            inv_freq = self.inv_freq
            
        # Compute theta = position * frequency
        # Flatten position_ids for batch processing
        flat_position_ids = position_ids.reshape(-1).float()
        
        # Element-wise outer product: [seq_len] x [dim/2] => [seq_len, dim/2]
        #
        # 각 position * 각 freq → 각 위치별 각도(theta) [seq_len x dim/2]
        #     → 단순히, cos(𝜃), sin(𝜃) 에 들어가는 𝜃 값을 의미
        #     → 왜 이 값(각 position * 각 freq)이 𝜃 값인가?
        #         → torch.sin, torch.cos 함수는 입력값을 각도(라디안)로 해석
        #         → cos(𝜃), sin(𝜃)에 넣은 값은 주기적(파동) 패턴이 생성 
        #         → RoPE에서는 이 값을 써서 Q/K 벡터를 복소수 평면에서 “회전”시키는 효과를 냄
        #         → 즉, position별로 벡터의 방향이 조금씩 달라짐
        #
        #         → RoPE(또는 sinusoidal positional embedding)의 본질은 "토큰의 위치 정보"를 여러 “파동(진동, 주파수)”로 encode하는 것.
        #             → position(포지션): 해당 토큰이 시퀀스에서 몇 번째인가?
        #             → inv_freq(역주파수): "각 임베딩 차원"이 변하는 “속도”를 조절하는 값
        #             → 두 값을 곱하면,
        #                → “0번째 포지션에서는 0 각도”, 
        #                → “1번째 포지션에서는 inv_freq_i 만큼 증가”, 
        #                → “2번째 포지션에서는 2×inv_freq_i 만큼 증가”
        #                → … 
        #                → 즉, position이 커질수록 각도(라디안)가 커져서 벡터가 점점 더 많이 ‘회전’하게 됨
        #          → 그럼, 여기서의 각도의 의미는?
        #              → 0에서 시작해서, 주파수(inv_freq_i)만큼 매 스텝 증가하는 값  
        #              → 예)
        #                  → position이 0, 1, 2, 3, ... 
        #                  → 각 임베딩 차원별로 > θ = 0, 1xinv_freq_i​, 2xinv_freq_i​, 3xinv_freq_i​,... # 0에서 시작해서, 주파수(inv_freq_i)만큼 매 스텝 증가하는 값  
        #                  → 이 값(아래 freqs)은 sin/cos의 입력(=라디안 각도)임 
        freqs = flat_position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0)
        
        # Reshape to include batch dimension
        freqs = freqs.reshape(batch_size, seq_len, -1)
        
        # Now create interleaved pattern
        #
        # i) 차원별로 θ(각도)를 “짝수, 홀수 차원 쌍”으로 interleave(교차)하는 핵심 단계
        #     → freqs의 shape: (batch, seq_len, dim/2)
        #     → 여기서 각 원소는 각 위치별, 차원별 θ(각도)임 (위의 설명 참조)
        #         → freqs[0, 0, :] = [0, 0, ...]
        #         → freqs[0, 1, :] = [1×inv_freq_0, 1×inv_freq_1, ...]
        #     → RoPE의 핵심은 
        #         → 임베딩의 짝수 차원(0,2,4,...) > cos(θ) 
        #         → 임베딩의 홀수 차원(1,3,5,...) > sin(θ) 
        #         → 실제 임베딩에서 (cos(θ₀), sin(θ₀), cos(θ₁), sin(θ₁), ...)처럼 [짝수, 홀수] 쌍으로 θ를 interleave(교차)하여 생성.
        #     → torch.cat([freqs, freqs], dim=-1) 이렇게 하면?
        #         → 예) 
        #             → freqs = [θ₀, θ₁, θ₂, θ₃] 
        #             → torch.cat([freqs, freqs], dim=-1) 적용
        #             → [θ₀, θ₁, θ₂, θ₃, θ₀, θ₁, θ₂, θ₃]
        # ii) 왜 freqs(각도)를 똑같이 한 번 더 붙여서 차원을 2배로 늘렸는가?
        #     → 로터리 임베딩의 “cos/sin interleave 구조” 때문
        #     → 예)
        #        → freqs = [θ₀, θ₁, θ₂, θ₃] 라면, # [batch, seq_len, dim//2]
        #        → 각 position에서 dim/2개의 θ(각도)
        #        → 왜 dim/2인 이유는, cos/sin쌍을 만들기 때문
        #            → (0,1): θ₀ 
        #            → (2,3): θ₁ 
        #            → (4,5): θ₂
        #        → dim/2 > concat > dim 
        #            → emb = [θ₀, θ₁, θ₂, θ₃] > concat > emb = [θ₀, θ₁, θ₂, θ₃, θ₀, θ₁, θ₂, θ₃]
        #            → (cos(θ₀), sin(θ₀), cos(θ₁), sin(θ₁), ...)
        #                → cos(θ₀)와 sin(θ₀)의 θ₀는 다르다!
        #                → cos는 [θ₀, θ₁, θ₂, θ₃,, : 1에서 4번째까지  
        #                → sim는 ,θ₀, θ₁, θ₂, θ₃]  : 5번째에서 마지막까지 
        #                → 즉, 다음과 같다. (interleave mapping)
        #                    → 0번 차원: cos(emb[0]) = cos(θ₀) 
        #                    → 1번 차원: sin(emb[4]) = sin(θ₀) 
        #                    → 2번 차원: cos(emb[1]) = cos(θ₁) 
        #                    → 3번 차원: sin(emb[5]) = sin(θ₁) 
        #                    → 4번 차원: cos(emb[2]) = cos(θ₂) 
        #                    → 5번 차원: sin(emb[6]) = sin(θ₂) 
        #                    → 6번 차원: cos(emb[3]) = cos(θ₃)
        #                    → 7번 차원: sin(emb[7]) = sin(θ₃)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Compute cos and sin
        # 
        # i) 최종적으로 cos, sin 값을 계산 - 나중에 선별 계산
        #        → cos(emb): [cos(θ₀), cos(θ₁), cos(θ₂), cos(θ₃), cos(θ₀), cos(θ₁), cos(θ₂), cos(θ₃)]
        #        → sin(emb): [sin(θ₀), sin(θ₁), sin(θ₂), sin(θ₃), sin(θ₀), sin(θ₁), sin(θ₂), sin(θ₃)]   
        # ii) self.attention_scaling
        #        → 로터리 임베딩의 cos/sin 값에 곱해주는 scaling factor(스케일링 계수)
        #        → 수치적 안정성 및 학습 효율
        #            → cos/sin 값의 스케일이 너무 작거나 클 경우 attention 값의 분포가 흔들려 학습이 불안정
        #            → attention 에서 1/root(d)로 나누는 과정이 존재 (d = 임베딩 차원수) 
        #            → RoPE를 적용한 후에도 전체 attention 스케일이 기존과 비슷하게 유지될 수 있도록 조정하기 위한 용도
        cos = torch.cos(emb) * self.attention_scaling
        sin = torch.sin(emb) * self.attention_scaling
        
        return cos, sin

#  “허수부”(i 파트) 계산을 벡터 연산으로 구현하기 위해서
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the input by dividing the hidden dimension to two, then swapping and negating dimensions.
    """
    # 임베딩의 마지막 차원(=head_dim)을 절반으로 쪼갬
    #     → 예) x = [a, b, c, d] >  x1 = [a, b], x2 = [c, d]
    x1, x2 = x.chunk(2, dim=-1)
    
    # i) x2(뒤쪽 절반)를 음수로 바꿔서 앞에 붙임 > x1(앞쪽 절반)는 그대로 뒤에 붙임 
    #     → 예) [a, b], [c, d] → [-c, -d, a, b]
    # ii) why?
    #     → 임베딩을 두 차원씩 쌍으로 보면,
    #     → 하나를 "실수부", 하나를 "허수부"
    #     → 즉, [a, b] → a + ib (복소수)
    #         → [실수부] : a·cos(θ) - b·sin(θ)
    #         → [허수부] : b·cos(θ) + a·sin(θ)
    #             → 이를  [-b, a]에 sin(θ) 곱하면 [-b·sin(θ), a·sin(θ)] # "벡터" 곱으로 구현하기 위해!!
    #             → 이는 (첫번째: 실수부의 -b·sin(θ), 두번째: 허수부의 a·sin(θ))
    #     → 그래서, rotate_half를 이용하여, [x1, x2] → [-x2, x1]로 바꿔 붙임
    #         → rotate_half([a, b]) > [-b, a]
    #         → [a·cos(θ) - b·sin(θ), b·cos(θ) + a·sin(θ)]
    # iii) 정리: rotate_half 함수에서 [-x2, x1]로 바꾸는 이유는 바로 복소수 곱의 “허수부”(i 파트) 계산을 벡터 연산으로 구현하기 위해서
    return torch.cat((-x2, x1), dim=-1)

# Apply rotary position embeddings to queries and keys.
def apply_rotary_pos_embd(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim:int=1)-> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings to query and key tensors in attention mechanisms.

    Rotary positional embeddings inject position-dependent rotations into query and key vectors,
    enabling transformers to encode positional information effectively without explicit positional encoding.

    Args:
        q (torch.Tensor): Query tensor with shape [batch_size, num_heads, seq_len, head_dim].
        k (torch.Tensor): Key tensor with shape [batch_size, num_heads, seq_len, head_dim].
        cos (torch.Tensor): Precomputed cosine positional embeddings with shape [batch_size, seq_len, head_dim].
        sin (torch.Tensor): Precomputed sine positional embeddings with shape [batch_size, seq_len, head_dim].
        unsqueeze_dim (int, optional): Dimension index to unsqueeze `cos` and `sin` to enable broadcasting.
                                      Defaults to 1 (typically the heads dimension).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The rotated query and key tensors (`q_embed`, `k_embed`), 
                                           each with the same shape as the input tensors.

    How it works:
        - `cos` and `sin` tensors are unsqueezed at `unsqueeze_dim` to broadcast across attention heads.
        - Rotary embeddings apply a complex number rotation in the embedding space using:
            rotated = (original * cos) + (rotate_half(original) * sin)
        - `rotate_half` performs a specific half-dimension rotation on the input tensor.
        - This operation encodes relative position information in q and k without adding explicit positional vectors.

    Example:
        q_embed, k_embed = apply_rotary_pos_embd(q, k, cos, sin)

    """

    # We need to make sure cos and sin can be properly broadcast
    # to the shape of q and k by adding the heads dimension
    #
    # ROPE 계산 결과 = (cos, sin)
    #   → unsqueeze하는 이유:
    #       → q, k의 shape: [batch, num_heads, seq_len, head_dim]
    #       → cos, sin의 2번째 차원(heads)에 1을 끼워 넣어 "브로드캐스팅"을 가능하게 함
    cos = cos.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]
    
    # Apply complex multiplication:
    # (q * cos) + (rotate_half(q) * sin)
    # 
    # 실제 RoPE 공식에 따라 Q/K 벡터에 곱함
    #       → (q * cos) + (rotate_half(q) * sin)
    #           → 실수부 (q * cos)
    #           → 허수부 (rotate_half(q) * sin)
    #               → rotate_half()의 역할 : 바로 복소수 곱의 “허수부”(i 파트) 계산을 벡터 연산으로 구현하기 위해서
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L214
# https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L382
class LanguageModelGroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA) as used in some transformer-based language models.

    GQA reduces computation by using fewer key-value heads than query heads,
    grouping multiple query heads to share the same key-value heads.

    Args:
        cfg: Configuration object containing:
            - lm_n_heads (int): Number of query heads.
            - lm_n_kv_heads (int): Number of key-value heads.
            - lm_hidden_dim (int): Hidden embedding dimension.
            - lm_dropout (float): Dropout rate.
    """
    def __init__(self, cfg):
        super().__init__()

        self.n_heads    = cfg.lm_n_heads
        self.n_kv_heads = cfg.lm_n_kv_heads
        self.embd_dim   = cfg.lm_hidden_dim
        self.dropout    = cfg.lm_dropout

        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"

        self.n_kv_groups   = self.n_heads // self.n_kv_heads
        self.head_dim      = self.embd_dim // self.n_heads

        self.q_proj        = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.k_proj        = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.v_proj        = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.out_proj      = nn.Linear(self.embd_dim, self.embd_dim, bias=False)

        self.attn_dropout  = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Use scaled dot product attention if available
        self.sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.sdpa:
            print("Warning: scaled dot product attention not available, using standard attention in LM.")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask=None, block_kv_cache=None) -> tuple[torch.Tensor, dict]:
        """
        Forward pass for grouped query attention.

        Args:
            x (Tensor): Input tensor of shape (B, T_curr, C), where
                        B = batch size,
                        T_curr = current sequence length,
                        C = embedding dimension.
            cos (Tensor): Rotary embedding cosines, shape compatible with q and k.
            sin (Tensor): Rotary embedding sines, shape compatible with q and k.
            attention_mask (Tensor, optional): Attention mask tensor of shape (B, total_kv_length),
                                               with 1 for tokens to attend to and 0 for padding.
            block_kv_cache (dict, optional): Cache dict with 'key' and 'value' tensors for autoregressive decoding.

        Returns:
            tuple[Tensor, dict]:
                - Output tensor after attention and projection, shape (B, T_curr, C).
                - Updated block_kv_cache dict for caching key-value states.
        """
        is_prefill = block_kv_cache is None

        B, T_curr, C = x.size() # T_curr is the sequence length of the current input x

        q_curr = self.q_proj(x).view(B, T_curr, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T_curr, head_dim)
        k_curr = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv_heads, T_curr, head_dim)
        v_curr = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv_heads, T_curr, head_dim)

        # Apply rotary embeddings to the current q and k
        q, k_rotated = apply_rotary_pos_embd(q_curr, k_curr, cos, sin)

        # Check if we can use cached keys and values
        if not is_prefill and block_kv_cache['key'] is not None:
            # Concatenate with cached K, V
            # k_rotated and v_curr are for the new token(s)
            k = block_kv_cache['key']
            v = block_kv_cache['value']
            k = torch.cat([k, k_rotated], dim=2)
            v = torch.cat([v, v_curr], dim=2)
            block_kv_cache['key']   = k
            block_kv_cache['value'] = v
        else:
            # No cache, this is the first pass (prefill)
            k              = k_rotated
            v              = v_curr
            block_kv_cache = {'key': k, 'value': v}

        # Repeat K, V for Grouped Query Attention
        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1) # (B, n_heads, T_kv, head_dim)
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1) # (B, n_heads, T_kv, head_dim)
        
        T_kv  = k_exp.size(2) # Total sequence length of keys/values

        # Prepare attention mask for SDPA or manual path
        # attention_mask is (B, T_kv_total_length), 1 for attend, 0 for pad
        additive_attn_mask = None
        if attention_mask is not None:
            # The current `attention_mask` parameter is assumed to be `[B, total_sequence_length_kv]`
            # Let's make it `[B, 1, 1, T_kv]` for SDPA.
            mask_for_keys      = attention_mask[:, :T_kv] # Ensure mask matches key length [B, T_kv]
            additive_attn_mask = (1.0 - mask_for_keys.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min
            # This additive_attn_mask shape is [B, 1, 1, T_kv]

        if self.sdpa and x.device.type != 'mps':
            # During decode, no additional masking needed as [1, T_kv] is naturally causal
            is_causal = (T_curr == T_kv and T_curr > 1)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k_exp, v_exp,
                attn_mask=additive_attn_mask, 
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal
            )
        else:
            # Manual attention implementation
            attn = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(self.head_dim) # (B, n_heads, T_curr, T_kv)
            
            # During decode: no additional masking needed as [1, T_kv] is naturally causal
            if T_curr == T_kv and T_curr > 1:
                causal_mask_val = torch.tril(torch.ones(T_curr, T_curr, device=x.device, dtype=torch.bool)).view(1, 1, T_curr, T_curr)
                attn            = attn.masked_fill(~causal_mask_val, float('-inf'))

            if additive_attn_mask is not None: # Additive padding mask
                # additive_attn_mask is [B,1,1,T_kv], needs to be broadcast to [B, n_heads, T_curr, T_kv]
                attn = attn + additive_attn_mask 

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y    = attn @ v_exp
            
        y = y.transpose(1, 2).contiguous().view(B, T_curr, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y, block_kv_cache

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L160
class LanguageModelMLP(nn.Module):
    """
    Implements the feed-forward network (MLP) block used in transformer-based language models.

    This MLP uses a gated activation mechanism where two separate linear projections
    are applied to the input: one passed through an activation function (gate_proj),
    and the other as is (up_proj). Their element-wise product is then projected back
    to the embedding dimension (down_proj).

    Args:
        cfg: Configuration object containing:
            - lm_hidden_dim (int): The embedding dimension size.
            - lm_inter_dim (int): The intermediate dimension size for the MLP.

    Attributes:
        activation_fn (Callable): The activation function used (SiLU).
        gate_proj (nn.Linear): Linear projection for gating pathway.
        up_proj (nn.Linear): Linear projection for upscaling pathway.
        down_proj (nn.Linear): Linear projection for downscaling back to embedding dim.
    """

    def __init__(self, cfg):
        super().__init__()
        self.embd_dim = cfg.lm_hidden_dim
        self.inter_dim = cfg.lm_inter_dim

        self.activation_fn = F.silu
        self.gate_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.up_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, self.embd_dim, bias=False)

    def forward(self, x):
        """
        Forward pass through the gated MLP block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embd_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, embd_dim),
                    after gated MLP transformation.
        """
        gate = self.activation_fn(self.gate_proj(x))
        x = self.up_proj(x)
        x = self.down_proj(gate * x)

        return x

# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L222
class LanguageModelBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp   = LanguageModelMLP(cfg)
        self.attn  = LanguageModelGroupedQueryAttention(cfg)
        self.norm1 = RMSNorm(cfg) # Input Norm
        self.norm2 = RMSNorm(cfg) # Post Attention Norm
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: torch.Tensor=None, block_kv_cache: dict=None):
        """
        Forward pass of the Transformer block.

        Args:
            x (Tensor)  : Input tensor of shape (batch_size, seq_len, hidden_dim).
            cos (Tensor): Cosine positional embeddings for rotary embedding, shape matching sequence length and head dimension. # RoPE
            sin (Tensor): Sine positional embeddings for rotary embedding, same shape as cos.                                   # RoPE
            attention_mask (Tensor, optional): Attention mask of shape (batch_size, total_kv_length),
                                                with 1 indicating tokens to attend to and 0 for padding tokens.
            block_kv_cache (dict, optional)  : Key-value cache dict for cached keys and values
                                                during decoding. If None, no cache is used.
        Returns:
            Tuple[Tensor, dict]: Output tensor after the block (same shape as input),
                and the updated key-value cache dictionary.
        """
        
        res               = x
        x                 = self.norm1(x)
        x, block_kv_cache = self.attn(x, cos, sin, attention_mask, block_kv_cache)
        x                 = res + x

        res               = x
        x                 = self.norm2(x)
        x                 = self.mlp(x)
        x                 = res + x

        return x, block_kv_cache

# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L251
class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lm_use_tokens = cfg.lm_use_tokens
        self.lm_tie_weights = cfg.lm_tie_weights

        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)
        self.rotary_embd = RotaryEmbedding(cfg)
        self.blocks = nn.ModuleList([
            LanguageModelBlock(cfg) for _ in range(cfg.lm_n_blocks)
        ])
        self.norm = RMSNorm(cfg) # Final Norm
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor=None, kv_cache: list[dict]=None, start_pos: int=0):
        """
        Performs a forward pass through the language model.

        Args:
            x (Tensor): Input tensor. If `lm_use_tokens` is True, this should be
                token indices with shape (batch_size, sequence_length).
                If False, it should be embeddings of shape (batch_size, sequence_length, hidden_dim).
            attention_mask (Tensor, optional): Mask tensor for attention to
                specify which tokens to attend to, typically of shape
                (batch_size, sequence_length). Default is None.
            kv_cache (list[dict], optional): List of key-value caches for each transformer
                block to enable efficient autoregressive decoding.
                If None, no cache is used and new ones are created. Default is None.
            start_pos (int, optional): The starting position index for the current input
                sequence. Used to compute rotary positional embeddings correctly,
                especially for cached sequences during generation. Default is 0.

        Returns:
            Tuple:
                - Tensor: Output logits with shape (batch_size, sequence_length, vocab_size)
                if `lm_use_tokens` is True, otherwise the hidden state embeddings
                (batch_size, sequence_length, hidden_dim).
                - list: Updated list of key-value caches, one for each transformer block,
                useful for autoregressive decoding and incremental generation.

        Behavior:
            - If `lm_use_tokens` is True, the input token indices are first embedded.
            - Rotary positional embeddings are generated for the current input positions,
            which are passed along to each transformer block.
            - For each transformer block, the input is processed along with
            rotary embeddings, attention mask, and optional cached key-values.
            - After processing all blocks, a final RMS normalization is applied.
            - If tokens are used, the normalized hidden states are projected to logits
            over the vocabulary.
            - The method returns the logits or embeddings along with the updated
            cache for efficient decoding.
        """
        if self.lm_use_tokens:
            x = self.token_embedding(x)

        # T_curr is the length of the current input sequence
        B, T_curr, _ = x.size()
        
        # Create position_ids for the current sequence based on start_pos
        current_position_ids = torch.arange(start_pos, start_pos + T_curr, device=x.device).unsqueeze(0).expand(B, -1)

        # RoPE (Rotary Positional Embedding)
        #    → RoPE는 Llama, GPT-NEOX 등 최신 트랜스포머에서 기본 positional embedding 사용
        #    → 입력 시퀀스(토큰)의 각 위치에 대해, 학습이 필요없는(학습파라메터가 필요없음) 사인/코사인 함수를 기반으로 하는 position embedding을 만들어 Q/K 벡터에 position 정보를 더하는 데 사용
        #    → 입력 시퀀스의 위치(position)에 따라 각 차원의 임베딩을 사인/코사인 곡선으로 회전(rotary)시켜, 트랜스포머의 Q/K에 위치 정보를 삽입. 
        #    → 더 긴 시퀀스에서도 position 정보가 잘 보존
        cos, sin = self.rotary_embd(current_position_ids) # Get rotary position embeddings for current tokens

        # Initialize new KV cache if none provided
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])

        x = self.norm(x)

        # Compute logits if we are using tokens, otherwise stay in the embedding space
        if self.lm_use_tokens: 
            x = self.head(x) 

        return x, kv_cache


    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor, max_new_tokens: int=20):
        """
        Generate tokens autoregressively from a given input sequence.

        Args:
            inputs (torch.Tensor): Input tensor containing token indices or embeddings.
                Shape: (batch_size, sequence_length) or (sequence_length,) for a single sequence.
            max_new_tokens (int): Number of new tokens to generate after the input sequence.

        Returns:
            torch.Tensor: The generated sequence, including the original inputs and newly generated tokens.
                Shape: (batch_size, sequence_length + max_new_tokens)
        """
        # Add batch dimension if needed
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        generated_outputs = inputs.clone()

        prompt_output, kv_cache_list = self.forward(
            generated_outputs, 
            attention_mask=None,
            kv_cache=None,
            start_pos=0
        )
        last_output = prompt_output[:, -1, :]

        # Decode Phase with KV cache
        for i in range(max_new_tokens):
            if self.lm_use_tokens:
                # Now the model outputs logits
                next_output = torch.argmax(last_output, dim=-1, keepdim=True)
            else:
                # Now the model outputs embeddings
                next_output = last_output.unsqueeze(1)

            generated_outputs = torch.cat((generated_outputs, next_output), dim=1)
            
            # The token being processed is `next_token`. Its position is `generated_outputs.size(1) - 1`.
            current_token_start_pos = generated_outputs.size(1) - 1

            if i == max_new_tokens - 1: 
                break

            decode_step_output, kv_cache_list = self.forward(
                next_output, 
                attention_mask=None,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )
            last_output = decode_step_output[:, -1, :] 
    
        return generated_outputs

    # Load the model from a pretrained HuggingFace model (we don't want to have to train the Language Backbone from scratch)
    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download
        import safetensors
        import torch.nn.init as init
                
        # Load the HuggingFace config
        hf_config = AutoConfig.from_pretrained(cfg.lm_model_type)
        
        # Store original HF vocab size before we modify it
        original_vocab_size = hf_config.vocab_size
        # print(f"Original vocabulary size from pretrained model: {original_vocab_size}")
        
        # Configure model parameters from HF config
        cfg.lm_hidden_dim = hf_config.hidden_size
        cfg.lm_inter_dim = hf_config.intermediate_size
        cfg.lm_rms_eps = hf_config.rms_norm_eps
        cfg.lm_re_base = hf_config.rope_theta
        cfg.lm_max_position_embeddings = hf_config.max_position_embeddings
        # We're keeping our own vocab size in cfg, but checking it's larger than original
        if hasattr(cfg, 'lm_vocab_size'):
            if cfg.lm_vocab_size < original_vocab_size:
                raise ValueError(f"Config vocab size ({cfg.lm_vocab_size}) is smaller than pretrained model vocab size ({original_vocab_size})")
            # print(f"Using vocabulary size: {cfg.lm_vocab_size}")
        else:
            # If not specified, use the original
            cfg.lm_vocab_size = original_vocab_size
            # print(f"Using original vocabulary size: {cfg.lm_vocab_size}")
        
        cfg.lm_n_heads = hf_config.num_attention_heads
        cfg.lm_n_kv_heads = hf_config.num_key_value_heads
        cfg.lm_dropout = hf_config.attention_dropout
        cfg.lm_n_blocks = hf_config.num_hidden_layers
        
        # Create our model with potentially larger vocabulary
        model = cls(cfg)
        safetensors_file = hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors")
        
        sd = model.state_dict()
        
        mapping = {
            'model.embed_tokens.weight': 'token_embedding.weight',
            'model.norm.weight': 'norm.weight'
        }
        
        for i in range(cfg.lm_n_blocks):
            layer_prefix = f'model.layers.{i}.'
            block_prefix = f'blocks.{i}.'
            
            mapping.update({
                f"{layer_prefix}self_attn.q_proj.weight": f"{block_prefix}attn.q_proj.weight",
                f"{layer_prefix}self_attn.k_proj.weight": f"{block_prefix}attn.k_proj.weight",
                f"{layer_prefix}self_attn.v_proj.weight": f"{block_prefix}attn.v_proj.weight",
                f"{layer_prefix}self_attn.o_proj.weight": f"{block_prefix}attn.out_proj.weight",
                f"{layer_prefix}mlp.gate_proj.weight": f"{block_prefix}mlp.gate_proj.weight",
                f"{layer_prefix}mlp.up_proj.weight": f"{block_prefix}mlp.up_proj.weight",
                f"{layer_prefix}mlp.down_proj.weight": f"{block_prefix}mlp.down_proj.weight",
                f"{layer_prefix}input_layernorm.weight": f"{block_prefix}norm1.weight",
                f"{layer_prefix}post_attention_layernorm.weight": f"{block_prefix}norm2.weight"
            })
        
        # Special handling for token embeddings with extended vocabulary
        has_extended_embeddings = False
        with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    tensor = f.get_tensor(hf_key)
                    
                    # Special handling for token embeddings if vocab sizes differ
                    if hf_key == 'model.embed_tokens.weight' and tensor.shape[0] != sd[our_key].shape[0]:
                        has_extended_embeddings = True
                        print(f"Extending token embeddings from {tensor.shape} to {sd[our_key].shape}")
                        
                        # Copy existing embeddings to the beginning of our larger embedding matrix
                        sd[our_key][:tensor.shape[0]].copy_(tensor)
                        
                        # Initialize the new embeddings using the same approach as the original model
                        std = 0.02  # Common value, but you might want to adjust based on model
                        init.normal_(sd[our_key][tensor.shape[0]:], mean=0.0, std=std)
                        
                        print(f"Initialized {sd[our_key].shape[0] - tensor.shape[0]} new token embeddings")
                        sd['head.weight'].copy_(sd[our_key])  # Update the head weights as well
                    elif tensor.shape == sd[our_key].shape:
                        sd[our_key].copy_(tensor)
                    else:
                        print(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")
                else:
                    if hf_key not in f.keys():
                        print(f"Warning: Key {hf_key} not found in safetensors file")
                    if our_key not in sd:
                        print(f"Warning: Key {our_key} not found in model state dict")
        
        # Load the state dict
        model.load_state_dict(sd)
        
        # Handle output projection / language modeling head
        if has_extended_embeddings and hasattr(model, 'head') and 'head.weight' in sd:
            # If we have a separate output projection layer and extended the vocab
            # we should handle it similarly to the input embeddings
            with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                if 'lm_head.weight' in f.keys():
                    lm_head = f.get_tensor('lm_head.weight')
                    if lm_head.shape[0] != sd['head.weight'].shape[0]:
                        print(f"Extending LM head from {lm_head.shape} to {sd['head.weight'].shape}")
                        # Copy existing weights
                        sd['head.weight'][:lm_head.shape[0]].copy_(lm_head)
                        # Initialize new weights
                        std = 0.02
                        init.normal_(sd['head.weight'][lm_head.shape[0]:], mean=0.0, std=std)
                        # Load updated weights
                        model.load_state_dict(sd)
        
        # Handle weight tying (if needed)
        if cfg.lm_tie_weights and hasattr(model, 'head') and hasattr(model, 'token_embedding'):
            model.head.weight = model.token_embedding.weight
            # print("Tied token embedding and LM head weights")
        
        print(f"Successfully loaded {cfg.lm_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model
