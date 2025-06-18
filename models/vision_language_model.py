import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional


from models.utils import top_k_top_p_filtering
from models.vision_transformer import ViT
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector
from models.config import VLMConfig

from data.processors import get_tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder        = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder        = LanguageModel(cfg)
        
        self.MP            = ModalityProjector(cfg) # pixel shuffle + linear 의 형태
        self.load_backbone = load_backbone
        self.tokenizer     = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens)

    # 텍스트 시퀀스의 "이미지 토큰" 자리에 실제 이미지 임베딩을 삽입하는 함수
    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        # input_ids(batch, seq_len):  각 배치의 토큰 시퀀스(정수 ID). = 각각 실제 이미지, 텍스트 관련 넣어야할 위치 존재
        # token_embd(batch, seq_len, embed_dim): 각 토큰의 임베딩(통상 LLM 임베딩)
        # image_embd(batch, image_token_len, embed_dim): 이미지의 임베딩 벡터(통상 이미지 Encoder에서 뽑은 벡터) 
        
        # Vectorized replacement of image token placeholders.
        # This assumes that `input_ids` contains `self.cfg.mp_image_token_length` placeholders
        # for image tokens, and `image_embd` has the corresponding features.
        #
        updated_token_embd = token_embd.clone() # token_embd 텐서를 복사해서 새 텐서를 만듦 (원본 보호)

        # Find the start index of image tokens for each item in the batch.
        # `torch.argmax` on the boolean mask (cast to int) returns the first index of `True` (value 1).
        # This relies on image_token_id being present and marking the start of the block.
        # 
        # 각 배치마다 이미지 토큰이 처음 나오는 인덱스(start index)를 찾음 
        # i)  input_ids == self.tokenizer.image_token_id: (batch, seq_len) - 이미지 토큰 위치가 True
        # ii) .int()로 True→1, False→0 변환 즉, 여기까지의 결과는 0과 1로만 구성되어 있음,
        # iii) 그래서, torch.argmax를 이용하여 처음 나오는 1의 위치값을 구함
        #      → 즉, torch.argmax(..., dim=1): 배치마다 True(1인)가 처음 등장하는 위치 반환 (즉, 첫 이미지 토큰 위치)
        # iiii) 참고 : text token squence 만들때 이미지 위치를 "<image>"라고 스트링 값을 넣음. > https://github.com/peternara/nanoVLM-LLM/blob/main/data/collators.py
        #      → 그래서, tokenizer에 집어넣으면, 임의의 매칭되는 사전 index(정수형 숫자로) 매핑되고, 이 값이 self.tokenizer.image_token_id 임.
        #      → 예로 이런식으로 미리 임의의 스트링을 사전에 정의 할수 있음
        #            from transformers import AutoTokenizer
        #            tokenizer = AutoTokenizer.from_pretrained("llama-2")
        #            tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
        #            image_token_id = tokenizer.convert_tokens_to_ids('<image>')
        #            print(image_token_id) 
        # V) 예) 
        #      → input_ids = [[5, 5, 999, 999, 999, 7],     # 이미지 토큰(999)이 2~4 위치
        #                     [8, 999, 999, 999, 4, 3]]     # 이미지 토큰(999)이 1~3 위치  
        #      → start_indices = [2,1]
        start_indices = torch.argmax((input_ids == self.tokenizer.image_token_id).int(), dim=1) # Shape: [batch_size]

        # Create batch indices for advanced indexing.
        # This tensor will be like [[0,0,..0], [1,1,..1], ..., [B-1,B-1,..B-1]]
        # where each inner list has length `image_token_len`.
        #
        # [0,0,...], [1,1,...], ..., [B-1,B-1,...] 형태로 생성 - 각 행의 길이는 이미지 토큰 길이
        #  i) 예) 위의 예(input_ids, start_indices)  이어서,  
        #      → batch_idx_fill = [[0, 0, 0],
        #                          [1, 1, 1]] 의 생성 하다는 의미 
        batch_idx_fill = torch.arange(input_ids.size(0), device=input_ids.device).unsqueeze(1).expand(-1, self.cfg.mp_image_token_length) # Shape: [batch_size, image_token_len]

        # Create sequence indices for replacement.
        # `seq_offsets` will be [0, 1, ..., image_token_len-1]
        # 
        # 이미지 토큰 길이만큼의 시퀀스 오프셋 생성 (0, 1, ..., image_token_len-1)
        # i) 0부터 시작해서 self.cfg.mp_image_token_length 보다 작은 시퀀스를 만든다.
        # ii) 예)
        #      → mp_image_token_length = 3 이면, seq_offsets = [[0,1,2]]
        seq_offsets = torch.arange(self.cfg.mp_image_token_length, device=input_ids.device).unsqueeze(0) # Shape: [1, image_token_len]

        # `start_indices.unsqueeze(1)` results in shape [batch_size, 1].
        # Broadcasting with `seq_offsets` gives, for each batch item `b`:
        # [start_indices[b], start_indices[b]+1, ..., start_indices[b]+image_token_len-1]
        # 
        # i) sequence_idx_fill의 목표 는 실제 이미지 토큰과 관련된 input_ids의 index(위치) 값 
        #      → 이미지 토큰 시작 위치에 offset을 더해서, 실제로 대체할 시퀀스 위치 계산
        #      → 즉, 시작 위치 3 이고 image_token_len=3이면 [3,4,5] )(equence_idx_fill)가 됨
        # ii) 예) 
        #      → start_indices.unsqueeze(1) : start_indices = [2,1] > [[2], [1]]
        #      → [[2], [1]] + [[0, 1, 2]] => [[2+0, 2+1, 2+2], [1+0, 1+1, 1+2]] => [[2, 3, 4], [1, 2, 3]]
        # 즉, sequence_idx_fill의 목표 는 실제 이미지 토큰과 관련된 input_ids의 index(위치) 값이다.  
        sequence_idx_fill = start_indices.unsqueeze(1) + seq_offsets # Shape: [batch_size, image_token_len]

        # Perform the replacement using advanced indexing.
        # `updated_token_embd[batch_idx_fill, sequence_idx_fill]` selects slices of shape [batch_size, image_token_len, D_lm]
        # `image_embd` also has shape [batch_size, image_token_len, D_lm] (or [B, mp_image_token_length, D_lm])
        # 
        # 실제 교체!
        # updated_token_embd[batch, 해당 시퀀스 위치] = image_embd
        #      → PyTorch의 advanced indexing 방식이라고 함.
        # i) 예) 위의 코드와 거의 동일한 심플한 다음 처럼 심플한 코드 : advanced indexing
        #        T                 = torch.arange(12).reshape(2, 6)
        #        batch_idx_fill    = torch.tensor([[0,0,0],[1,1,1]])
        #        sequence_idx_fill = torch.tensor([[2,3,4],[1,2,3]])
        #        print(T[batch_idx_fill, sequence_idx_fill])
        #         결과:
        #             tensor([[ 2,  3,  4],
        #                     [ 7,  8,  9]])
        #           → batch_idx_fill 과  sequence_idx_fill 그렇게 구성된 이유는 다음과 같이 삽입하는 것이다. 다만, advanced indexing은 이를 좀더 효율적으로 한꺼번에(동시에) 처리.
        #               → T[0, 2] = sequence_idx_fill[0,2] 
        #               → T[0, 3] = sequence_idx_fill[0,3]
        #               → T[0, 4] = sequence_idx_fill[0,4]
        #               → T[1, 1] = sequence_idx_fill[1,1]
        #               → T[1, 2] = sequence_idx_fill[1,2]
        #               → T[1, 3] = sequence_idx_fill[1,3]
        # advanced indexing을 사용해서 한 번에 교체 (broadcasting)
        updated_token_embd[batch_idx_fill, sequence_idx_fill] = image_embd.to(updated_token_embd.dtype)
        
        return updated_token_embd # (batch, seq_len, embed_dim) 

    def forward(self, input_ids, image, attention_mask=None, targets=None):
        # 1. visusal feature
        image_embd = self.vision_encoder(image)

        # 주의 : pixel shuffle의 입력은 이미지가 아니라, VIT 마지막 PATCH 레이어를 의미.
        #    → 의문) VIT 코드상으로보면 CLS를 제거하지 않는듯한데, (CLS를 제거해야지) HxW로 변환해야지 않나?
        # pixel shuffle + linear layer 
        image_embd = self.MP(image_embd) # [B, mp_image_token_length, D_lm]

        # 2. text feature
        token_embd = self.decoder.token_embedding(input_ids) # [B, T_sequence, D_lm]

        # 3. embedding tokens = image token + text token : 텍스트 시퀀스의 "이미지 토큰" 자리에 실제 이미지 임베딩을 삽입하는 함수 
        updated_token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        # The updated_token_embd is now the token_embd with image parts replaced.
        # The attention_mask comes from the collator and should already cover the full sequence.
        logits, _ = self.decoder(updated_token_embd, attention_mask=attention_mask)

        loss = None
        if targets is not None:
            logits = self.decoder.head(logits) # Apply LM head
            # Loss is calculated over all tokens, but `targets` (labels) will have -100 for non-answer tokens.
            # No need to slice logits based on image embedding size here, as the target mask handles it.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    @torch.inference_mode()
    def generate(self, input_ids, image, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):

        # 1. Process image
        image_embd = self.vision_encoder(image) # [B, T_img_feat, D_model]
        image_embd = self.MP(image_embd)      # [B, mp_image_token_length, D_lm]

        # 2. Embed initial text prompt tokens
        prompt_token_embeds = self.decoder.token_embedding(input_ids) # [B, T_prompt_text, D_lm]

        # 3. Combine image and text embeddings
        initial_combined_embeds = self._replace_img_tokens_with_embd(input_ids, prompt_token_embeds, image_embd)

        current_total_seq_len = initial_combined_embeds.size(1)
        batch_size = input_ids.size(0) # Or initial_combined_embeds.size(0)
        
        # --- Multimodal Prefill Phase ---
        prefill_output, kv_cache_list = self.decoder(
            initial_combined_embeds,
            attention_mask=attention_mask, # Use the provided attention mask
            kv_cache=None,
            start_pos=0
        )
        
        last_token_output_from_prefill = prefill_output[:, -1, :] 
        
        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output_from_prefill) 
        else:
            current_logits = last_token_output_from_prefill 

        # Store newly generated token IDs
        newly_generated_ids_list = []

        # --- Decode Phase by sampling tokens autoregressively using the kv-cache ---
        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            newly_generated_ids_list.append(next_token_id)
            
            # Embed the newly generated token
            next_token_embed = self.decoder.token_embedding(next_token_id) # [B, 1, D_lm]
            
            # The start_pos for the new token is the current total sequence length *before* adding this new token
            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            # update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)

            # With KV cache: only process the new token
            decode_step_output, kv_cache_list = self.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )
      
            last_token_output = decode_step_output[:, -1, :] 
            
            # Apply head to get logits (if model is in embedding mode)
            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output
        
        if not newly_generated_ids_list: # Handle case where max_new_tokens might be 0
            return torch.empty((batch_size,0), dtype=torch.long, device=input_ids.device)

        generated_ids = torch.cat(newly_generated_ids_list, dim=1)

        # Post-process to handle EOS token.
        if self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0: # Ensure generated_ids is not empty
            seq_len = generated_ids.size(1)
            device = generated_ids.device

            eos_mask = (generated_ids == self.tokenizer.eos_token_id) # Create a boolean mask for EOS tokens

            col_indices_for_min = torch.arange(seq_len, device=device) # Create column indices [0, 1, ..., seq_len-1]
            
            # In eos_mask, mark positions with actual col_idx, others with a large number
            masked_col_indices = torch.where(eos_mask, col_indices_for_min.unsqueeze(0).expand_as(generated_ids), seq_len + 1) 

            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values
            
            # Clamp values to seq_len (if no EOS found, min will be seq_len + 1, clamp brings it to seq_len0. This means if no EOS, or EOS is the last token, no replacement will happen for that sample.
            actual_first_eos_indices = torch.clamp(first_eos_indices_values, max=seq_len)

            # Create column indices for comparison, shape [batch_size, seq_len]
            col_indices_for_comparison = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(generated_ids)
            
            # Tokens are replaced if their column index is greater than the index of the first EOS token
            replace_mask = col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            
            generated_ids[replace_mask] = self.tokenizer.eos_token_id
        
        return generated_ids

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.

        Returns:
            VisionLanguageModel: The loaded model.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load safetensors weights
        load_model(model, weights_path)

        # Done!
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.

        Args:
            save_directory (str): The directory to save the model and config.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights as safetensors
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """
        Push the model and configuration to the Hugging Face Hub.

        Args:
            repo_id (str): The repo ID on the Hugging Face Hub.
        """
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )


MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: nanovlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
---

**nanoVLM** is a minimal and lightweight Vision-Language Model (VLM) designed for efficient training and experimentation. Built using pure PyTorch, the entire model architecture and training logic fits within ~750 lines of code. It combines a ViT-based image encoder (SigLIP-B/16-224-85M) with a lightweight causal language model (SmolLM2-135M), resulting in a compact 222M parameter model.

For more information, check out the base model on https://huggingface.co/lusxvr/nanoVLM-222M.

**Usage:**

Clone the nanoVLM repository: https://github.com/huggingface/nanoVLM.
Follow the install instructions and run the following code:

```python
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("{repo_id}")
```
"""
