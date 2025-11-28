#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 1 (Llama3 Korean Bllossom) 답변 생성
- 국방 도메인 대표 질문에 대해서만 답변 생성
"""

import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
from pathlib import Path
from datetime import datetime

class LLM1ResponseGenerator:
    """LLM 1 (Llama3 Korean) 답변 생성기"""
    
    def __init__(self, 
                 model_name="MLP-KTLim/llama-3-Korean-Bllossom-8B",
                 output_dir="./llm_comparison_data"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.device = None
        
    def load_model(self):
        """모델 로드"""
        print("=" * 80)
        print(f"LLM 1 모델 로드: {self.model_name}")
        print("=" * 80)
        
        # GPU 사용 가능 여부 확인
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 디바이스: {self.device}")
        
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 토크나이저 로드
        print(f"\n토크나이저 로드 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # 모델 로드
        print(f"모델 로드 중... (8B 모델이라 시간이 걸릴 수 있습니다)")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        print(f"✓ 모델 로드 완료")
        
        return self
    
    def generate_response(self, question, max_new_tokens=512, temperature=0.7):
        """단일 질문에 대한 답변 생성"""
        
        # 프롬프트 구성
        prompt = f"""다음 질문에 대해 상세하고 정확하게 답변해주세요.

질문: {question}

답변:"""
        
        # 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # 답변 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 디코딩
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 답변 부분만 추출
        if "답변:" in full_response:
            answer = full_response.split("답변:")[-1].strip()
        else:
            answer = full_response[len(prompt):].strip()
        
        return answer
    
    def load_defense_questions(self, benchmark_path):
        """국방 도메인 대표 질문 로드"""
        print("\n" + "=" * 80)
        print("국방 도메인 대표 질문 로드")
        print("=" * 80)
        
        # MMR 샘플링 결과에서 국방 도메인만 추출
        if os.path.exists(benchmark_path):
            df = pd.read_csv(benchmark_path)
            
            # 국방 도메인만 필터링
            if 'domain' in df.columns:
                defense_df = df[df['domain'] == 'defense'].copy()
            else:
                # 전체가 국방 도메인인 경우
                defense_df = df.copy()
            
            print(f"총 {len(defense_df)}개 질문 로드")
            
            return defense_df
        else:
            raise FileNotFoundError(f"벤치마크 파일을 찾을 수 없습니다: {benchmark_path}")
    
    def generate_all_responses(self, defense_df, max_new_tokens=512, temperature=0.7):
        """모든 질문에 대한 답변 생성"""
        print("\n" + "=" * 80)
        print(f"LLM 1 답변 생성 시작 (총 {len(defense_df)}개)")
        print("=" * 80)
        
        results = []
        
        for idx, row in tqdm(defense_df.iterrows(), total=len(defense_df), desc="답변 생성"):
            question_id = row.get('question_id', row.get('id', f'q_{idx}'))
            question = row['question'] if 'question' in row else row['input']
            
            print(f"\n[{idx+1}/{len(defense_df)}] 질문: {question[:50]}...")
            
            try:
                # 답변 생성
                answer = self.generate_response(
                    question,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                
                print(f"답변: {answer[:100]}...")
                
                results.append({
                    'question_id': question_id,
                    'question': question,
                    'response': answer,
                    'model': 'llama3-korean-bllossom-8b',
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"⚠️ 오류 발생: {e}")
                
                results.append({
                    'question_id': question_id,
                    'question': question,
                    'response': f"[ERROR] {str(e)}",
                    'model': 'llama3-korean-bllossom-8b',
                    'status': 'error'
                })
        
        print(f"\n✓ 답변 생성 완료: {len(results)}개")
        
        return results
    
    def save_results(self, results, format='both'):
        """결과 저장 (CSV, JSON, LLM Comparator 형식)"""
        print("\n" + "=" * 80)
        print("결과 저장")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. CSV 저장
        if format in ['csv', 'both']:
            csv_path = self.output_dir / f'llm1_responses_{timestamp}.csv'
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✓ CSV 저장: {csv_path}")
        
        # 2. JSON 저장 (일반 형식)
        if format in ['json', 'both']:
            json_path = self.output_dir / f'llm1_responses_{timestamp}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✓ JSON 저장: {json_path}")
        
        # 3. LLM Comparator 형식 저장
        comparator_data = self.convert_to_comparator_format(results)
        comparator_path = self.output_dir / f'llm1_comparator_{timestamp}.json'
        
        with open(comparator_path, 'w', encoding='utf-8') as f:
            json.dump(comparator_data, f, indent=2, ensure_ascii=False)
        print(f"✓ LLM Comparator 형식 저장: {comparator_path}")
        
        return {
            'csv': csv_path if format in ['csv', 'both'] else None,
            'json': json_path if format in ['json', 'both'] else None,
            'comparator': comparator_path
        }
    
    def convert_to_comparator_format(self, results):
        """LLM Comparator 형식으로 변환"""
        
        comparator_data = {
            "metadata": {
                "model_name": "llama3-korean-bllossom-8b",
                "model_full_name": "MLP-KTLim/llama-3-Korean-Bllossom-8B",
                "timestamp": datetime.now().isoformat(),
                "total_examples": len(results)
            },
            "examples": []
        }
        
        for result in results:
            example = {
                "id": result['question_id'],
                "prompt": result['question'],
                "response": result['response'],
                "model_name": result['model'],
                "status": result['status']
            }
            
            comparator_data["examples"].append(example)
        
        return comparator_data


def main():
    """메인 실행 함수"""
    
    # 설정
    BENCHMARK_PATH = './output_mmr_sampling/benchmark_mmr_defense.csv'  # 국방 도메인 대표 질문
    MAX_NEW_TOKENS = 512  # 생성할 최대 토큰 수
    TEMPERATURE = 0.7  # 생성 온도 (낮을수록 결정적, 높을수록 창의적)
    
    # LLM 1 답변 생성기 초기화
    generator = LLM1ResponseGenerator()
    
    # 모델 로드
    generator.load_model()
    
    # 국방 도메인 질문 로드
    defense_df = generator.load_defense_questions(BENCHMARK_PATH)
    
    # 답변 생성
    results = generator.generate_all_responses(
        defense_df,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE
    )
    
    # 결과 저장
    saved_files = generator.save_results(results, format='both')
    
    print("\n" + "=" * 80)
    print("✅ LLM 1 답변 생성 완료!")
    print("=" * 80)
    print(f"\n저장된 파일:")
    for format_type, path in saved_files.items():
        if path:
            print(f"  - {format_type}: {path}")
    
    print(f"\n다음 단계:")
    print(f"  1. LLM 2 답변 생성 실행")
    print(f"  2. LLM Comparator로 두 모델 비교")


if __name__ == '__main__':
    main()