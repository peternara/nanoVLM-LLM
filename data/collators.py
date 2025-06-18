import torch

class VQACollator(object):  # Visual Question Answering Collator
    def __init__(self, tokenizer, max_length, mp_image_token_length):
        self.tokenizer             = tokenizer
        self.max_length            = max_length
        self.mp_image_token_length = mp_image_token_length
        self.image_token_str       = tokenizer.image_token # 이미지가 입력됨을 알리는 특수토큰 문자열 (예: <image>)
    
    def __call__(self, batch): 
        images  = [item["image"] for item in batch]     # batch 리스트에서 각 데이터의 이미지만 추출해서 리스트로 저장
        texts   = [item["text_data"] for item in batch] # 각 데이터에서 질문/텍스트 부분만 리스트로 저장
        answers = [item["answer"] for item in batch]    # 각 데이터에서 정답만 리스트로 저장

        # Stack images
        images  = torch.stack(images) # 이미지 리스트를 (batch, C, H, W) 텐서로 합침

        # Create inputs by concatenating special image tokens, question, and answer
        input_sequences = []
        for i in range(len(texts)): # batch 길이만큼 반복해서 각 아이템에 대해 시퀀스 생성
            # Construct the image token segment string
            #  → (이미지 특수토큰 X개) + (질문) + (정답) 순으로 문자열을 만들어 input_sequences에 추가
            #  → 예시:
            #      mp_image_token_length=3이면, <image><image><image>
            #      "<image><image><image>...어떤 동물입니까?고양이"
            #  → self.image_token_str * self.mp_image_token_length : 이미지 토큰 개수 (mp_image_token_length) 만큼 이미지 토큰을 생성한다는 의미
            input_sequences.append(f"{self.image_token_str * self.mp_image_token_length}{texts[i]}{answers[i]}")

        # 예) input_sequences = [
        #                        "<image><image>What is this animal?cat",
        #                        "<image><image>What is the color?red"
        #                       ]
        # 이라면,
        # 결과는 (실제 확인 필요)
        #  {
        #     'input_ids': tensor([
        #            [0, 0, 102, 111, 119, 17, 13, 9, 247, 21, 453, 187],   # 예시 (batch=2, max_length=12)
        #            [0, 0, 102, 111, 119, 17, 13, 9, 132, 54, 287, 321]
        #        ]), # 각 문장이 토크나이저의 vocab에서 integer로 변환된 값. (패딩 포함) 
        #     'attention_mask': tensor([
        #            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        #        ])  # 실제 입력 토큰이면 1, 패딩이면 0.
        #   }
        #     아래는 사용하는 토크나이저 종류에 따라 추가될 수도 있는 필드들입니다.
        #   →   'token_type_ids': tensor([[0, 0, ...], [0, 0, ...]]),  # (BERT류에서)
        #   →   'special_tokens_mask': tensor([[0, 0, ...], [0, 0, ...]])
        #   →   'overflowing_tokens': ...
        #   →    등등
        encoded_full_sequences = self.tokenizer.batch_encode_plus(
            input_sequences,
            padding="max_length",
            padding_side="left",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        # Create labels where only answer tokens are predicted
        input_ids      = encoded_full_sequences["input_ids"]      # 입력 토큰 시퀀스 (batch, seq_len)
        attention_mask = encoded_full_sequences["attention_mask"] # 패딩 마스크 (batch, seq_len)

        #--------------------------------------------------------------------------------------------------------------------
        # Label (GT) 생성
        #     → labels 텐서: (batch, max_length) shape
         #         → 프롬프트 영역은 Answer가 아니기때문에 -100은 loss 무시(계산 안 함) < 이게 사실상 목표
        #         → 정답 토큰 자리만 정수로 남겨서 loss에 포함
        #--------------------------------------------------------------------------------------------------------------------
        labels         = input_ids.clone()                        # (batch, seq_len), 정답 레이블 텐서
        # labels를 오른쪽으로 한 칸씩 밀어서(shift) causal LM 학습에 맞게 조정
        #    → 예: input_ids가 [A, B, C, D]면 labels는 [B, C, D, ?]가 됨
        labels[:, :-1] = input_ids[:, 1:].clone()                 # Shift labels for causal LM, # 각 위치의 정답을 '다음 토큰'으로 설정
        # # 마지막 토큰 위치는 정답이 없으므로 -100으로 마스킹(loss 계산에서 제외)
        labels[:, -1]  = -100                                     # Last token has no target, # CrossEntropyLoss에서 ignore_index=-100 사용

        # Determine original lengths before padding/truncation to handle truncation cases
        # 토크나이즈된 각 시퀀스의 길이를 계산
        #   → 예: self.tokenizer.encode("hello world") → [101, 7592, 2088, 102]
        #   → 예: len([101, 7592, 2088, 102]) = 4,
        #   → 예: [4,,,10, 15]
        original_lengths = [len(self.tokenizer.encode(seq)) for seq in input_sequences]

        for i in range(len(batch)): # batch 길이만큼 반복해서 Label[i] 생성
            # Case 1: If sequence was truncated (original is longer than max_length)
            if original_lengths[i] > self.max_length:                
                # max_length보다 긴 경우 전체를 무시(마스킹), 전체 라벨 시퀀스를 -100으로 만듦. # 즉, 전체를 -100, 즉 loss 계산에서 완전히 제외
                #    → 왜 self.max_length 이후만 해야하는데 왜 전체를 하는가?
                #    → 이는 시퀀스가 max_length에 맞춰 잘리면서, 뒤쪽(대부분 정답)이 누락 > 일부만 있게 되면 정답으로 유효한가?란 고민 > 그래서, 전체 마스킹 > loss 계산 제외
                labels[i, :] = -100 # Ignore this sample entirely
                
                # print(f"Sample {i} truncated: original length {original_lengths[i]} exceeds max_length {self.max_length}. Ignoring sample.")
                continue
            
            # Case 2: Sequence fits within max_length
            # Determine the length of the question part for this sample
            question_part_length = len(self.tokenizer.encode(texts[i], add_special_tokens=False)) # 이 샘플의 질문(문제) 부분이 토크나이즈됐을 때의 길이 구함 (special token 없이 : add_special_tokens=False)
            
            # Find the position of the first actual token (non-padding)
            # attention_mask might be all zeros if the sequence is fully truncated (handled above) or empty.
            # Ensure there's at least one non-padding token to avoid errors with .nonzero().
            if attention_mask[i].sum() == 0: # Should not happen if not truncated and not empty.
                labels[i, :] = -100 # Defensive: if no actual tokens, ignore sample # 실제 토큰이 하나도 없는 경우(이상 상황), 전체를 무시 # 즉, 전체를 -100, 즉 loss 계산에서 완전히 제외
                continue

            # 실제 입력(패딩이 아닌) 첫 토큰의 인덱스 위치를 찾음
            # i) attention_mask[i] = [0, 0,   1, 1, 1, 1, 1,   0, 0, 0] # attention_mask[i]는 (max_length,) shape의 0 or 1 값을 가지는 텐서
            #                      ↑  ↑    ↑  ↑  ↑  ↑  ↑    ↑  ↑  ↑
            #                       패딩      실제 입력           패딩
            #    → 앞에 패딩(2개, [0, 0,,,])가 있는 이유 : padding_side="left" 옵션
            #        → HuggingFace 토크나이저 등에서 padding="max_length & padding_side="left" 를 동시에 적용하면, 
            #        → 시퀀스의 길이가 max_length보다 짧으면, 앞부분(왼쪽)에 패딩 채워짐
            #        → 위의 [,,0, 0, 0] 뒤쪽 패딩은 사실상 잘못된 예시
            # ii) attention_mask[i].nonzero(as_tuple=True)[0][0].item()
            #     → attention_mask[i].nonzero(as_tuple=True)  
            #         → attention_mask[i]의 0이 아닌 위치 인덱스(nonzero())를 반환
            #         → as_tuple=True : (텐서,)의 튜플로 반환
            #         → 즉, (tensor([2, 3, 4, 5, 6]),)
            #     → [0][0]
            #     → 첫번째 [0]: 첫 번째 텐서, 즉 tensor([2, 3, 4, 5, 6])
            #     → 두번째 [0]: 당연히 2 를 가져옴.
            #     → item() : tensor의 타입을 버리고, 단지 int으로 변환 단지 int 형 2
            first_token_pos = attention_mask[i].nonzero(as_tuple=True)[0][0].item() # 실제 입력(패딩이 아닌) 첫 토큰의 인덱스 위치를 찾음

            # 프롬프트 전체 길이 = 이미지 토큰 개수 (self.mp_image_token_length) + 질문 토큰 개수 (question_part_length)
            # The total length of the "prompt" part (special image tokens + question)
            total_prompt_length = self.mp_image_token_length + question_part_length
            
            # Mask labels for padding tokens (before first_token_pos) and the entire prompt part.
            # The prompt part starts at first_token_pos and has length total_prompt_length.
            # So, tokens from index 0 up to (first_token_pos + total_prompt_length - 1) should be masked.
            # The slicing labels[i, :N] masks indices 0 to N-1.
            #
            # 요점 : 정답(Answer) 토큰만 loss 계산에 사용하고, 프롬프트(맨앞쪽에 있는 패딩(존재한다면), 이미지 토큰, 질문)는 loss에서 제외 
            #     → 목표:  패딩·프롬프트(이미지 토큰, 질문)은 -100으로 채워야함. 
            # i) labels 텐서: (batch, max_length) shape
            #     → 각 자리별로 -100은 loss 무시(계산 안 함)
            #     → 정답 토큰 자리만 정수로 남겨서 loss에 포함
            # ii) 프롬프트란? = 맨앞쪽에 있는 패딩(존재한다면)+이미지 토큰 + 질문(텍스트) = (정답 토큰 앞에 있는 모든 것)
            # iii) 즉, 프롬프트 영역을 모두 -100으로 처리한다는 의미 = labels[i, :mask_until_idx] = -100
            mask_until_idx             = first_token_pos + total_prompt_length - 1 
            labels[i, :mask_until_idx] = -100 # [0 ~ (first_token_pos + total_prompt_length - 1)]까지는 모두 -100으로 마스킹 (정답 부분만 loss 계산)
            

        return {
            "image": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class MMStarCollator(object):  # https://huggingface.co/datasets/Lin-Chen/MMStar
    def __init__(self, tokenizer, mp_image_token_length):
        self.tokenizer = tokenizer
        self.mp_image_token_length = mp_image_token_length

        self.image_token_str = tokenizer.image_token
    
    def __call__(self, batch):
        images = [item["image"] for item in batch]
        questions = [item["text_data"] for item in batch]
        answers = [item["answer"] for item in batch]

        # Stack images
        images = torch.stack(images)

        # Create input sequences with image placeholders
        question_sequences = []
        for question_text in questions:
            question_sequences.append(f"{self.image_token_str * self.mp_image_token_length}{question_text}")
        
        
        encoded_question_sequences = self.tokenizer.batch_encode_plus(
            question_sequences,
            padding=True,
            padding_side="left",
            return_tensors="pt"
        )

        encoded_answer_sequences = self.tokenizer.batch_encode_plus(
            answers,
            padding=True,
            padding_side="left",
            return_tensors="pt"
        )
        
        return {
            "images": images,
            "input_ids": encoded_question_sequences['input_ids'],
            "attention_mask": encoded_question_sequences['attention_mask'],
            "labels": encoded_answer_sequences['input_ids'],
        }
