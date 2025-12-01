#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Comparator - VSCode 자동 뷰어 포함 버전
실행 후 자동으로 웹 UI가 열립니다 (Colab의 show_in_colab()과 동일한 방식)
"""

from llm_comparator import comparison
from llm_comparator import model_helper
from llm_comparator import llm_judge_runner
from llm_comparator import rationale_bullet_generator
from llm_comparator import rationale_cluster_generator
import vertexai
import os
import json
from datetime import datetime
from google.oauth2 import service_account
import sys
import custom_model_helper


print("=" * 80)
print("LLM Comparator - VSCode 자동 뷰어 포함 버전")
print("=" * 80)

# === 1. 서비스 계정 키로 인증 ===
print("\n[1단계] Vertex AI 인증")
#################################
key_path = '/home/gpuadmin/kim/llm_com/eastern-gravity-477106-i5-ceb9d33525f4.json'
#본인 키 json 파일 경로로 수정!
#################################

if not os.path.exists(key_path):
    raise FileNotFoundError(f"키 파일을 찾을 수 없음: {key_path}")

credentials = service_account.Credentials.from_service_account_file(
    key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)
print("  ✓ 인증 완료")

# === 2. Vertex AI 초기화 ===
print("\n[2단계] Vertex AI 초기화")
vertexai.init(
    #################################
    project="eastern-gravity-477106-i5", #본인 json 파일 참고
    #################################
    location="us-central1", #유지
    credentials=credentials
)
print("  ✓ 초기화 완료")

# === 3. 데이터 로드 ===
print("\n[3단계] 데이터 로드")
#################################
llm1_file = "/home/gpuadmin/kim/llm_com/llm_comparison_data/llm1_comparator_20251103_033057.json"
llm2_file = "/home/gpuadmin/kim/llm_com/llm_comparison_data/llm2_comparator_20251103_043303.json"
#본인 디렉토리 경로로 변경!! 파일은 github에 있음
#################################
with open(llm1_file, 'r', encoding='utf-8') as f:
    llm1_data = json.load(f)

with open(llm2_file, 'r', encoding='utf-8') as f:
    llm2_data = json.load(f)

print(f"  ✓ LLM1 응답: {len(llm1_data['examples'])}개")
print(f"  ✓ LLM2 응답: {len(llm2_data['examples'])}개")

model_a_name = llm1_data['metadata']['model_name']
model_b_name = llm2_data['metadata']['model_name']

print(f"  ✓ Model A: {model_a_name}")
print(f"  ✓ Model B: {model_b_name}")

# === 4. LLM Comparator 입력 형식으로 변환 ===
print("\n[4단계] 데이터 변환")
inputs = []
for item1, item2 in zip(llm1_data['examples'], llm2_data['examples']):
    inputs.append({
        'prompt': item1['prompt'],
        'response_a': item1['response'],
        'response_b': item2['response']
    })

print(f"  ✓ {len(inputs)}개 질문 준비 완료")

# === 5. 모델 헬퍼 초기화 ===
print("\n[5단계] 모델 헬퍼 초기화")
JUDGE_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-004"

print(f"  - Judge Model: {JUDGE_MODEL}")
print(f"  - Embedding Model: {EMBEDDING_MODEL}")
print(f"  - Max Output Tokens: 2048")

generator = custom_model_helper.VertexGenerationModelHelper(JUDGE_MODEL)
embedder = custom_model_helper.VertexEmbeddingModelHelper(EMBEDDING_MODEL)
print("  ✓ 모델 헬퍼 준비 완료")

# === 6. LLM Comparator 컴포넌트 초기화 ===
print("\n[6단계] LLM Comparator 컴포넌트 초기화")
judge = llm_judge_runner.LLMJudgeRunner(generator)
bulletizer = rationale_bullet_generator.RationaleBulletGenerator(generator)
clusterer = rationale_cluster_generator.RationaleClusterGenerator(generator, embedder)
print("  ✓ Judge, Bulletizer, Clusterer 준비 완료")

# === 7. LLM Comparator 실행 ===
print("\n[7단계] LLM Comparator 실행 (공식 클러스터링 포함)")
print(f"  - 총 질문 수: {len(inputs)}개")
print(f"\n  ⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

start_time = datetime.now()

try:
    comparison_result = comparison.run(
        inputs,
        judge,
        bulletizer,
        clusterer,
        model_names=(model_a_name, model_b_name)
    )
    
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    print(f"\n  ✓ LLM Comparator 실행 완료")
    print(f"  ⏰ 종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ⏱️  소요 시간: {elapsed_time/60:.1f}분")
    
except Exception as e:
    print(f"\n  ❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    raise

# === 8. 결과 저장 ===
print("\n[8단계] 결과 저장")
#################################
output_dir = "/home/gpuadmin/kim/llm_com/llm_comparison_results"
#본인 경로로 변경
#################################
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"{output_dir}/llm_comparator_auto_viewer_{timestamp}.json"

comparison.write(comparison_result, output_file)
print(f"  ✓ 결과 저장: {output_file}")

# === 9. 결과 통계 ===
print("\n" + "=" * 80)
print("평가 완료!")
print("=" * 80)

examples = comparison_result['examples']
scores = [ex['score'] for ex in examples]

a_wins = sum(1 for s in scores if s > 0)
b_wins = sum(1 for s in scores if s < 0)
ties = sum(1 for s in scores if s == 0)

print(f"\n📊 결과 요약:")
print(f"  • 총 평가 쌍: {len(examples)}개")
print(f"  • Judge 모델: {JUDGE_MODEL}")
print(f"  • Model A ({model_a_name}): {a_wins}승 ({a_wins/len(examples)*100:.1f}%)")
print(f"  • Model B ({model_b_name}): {b_wins}승 ({b_wins/len(examples)*100:.1f}%)")
print(f"  • 동점: {ties}개 ({ties/len(examples)*100:.1f}%)")
print(f"  • 평균 점수 차이: {sum(scores) / len(scores):.3f}")

# Rationale 통계
rationale_count = 0
total_ratings = 0
for ex in examples:
    individual_scores = ex.get('individual_rater_scores', [])
    total_ratings += len(individual_scores)
    for score_item in individual_scores:
        if isinstance(score_item, dict) and score_item.get('rationale'):
            rationale_count += 1

print(f"\n📝 Rationale 통계:")
print(f"  • 총 평가 횟수: {total_ratings}회")
print(f"  • Rationale 포함: {rationale_count}회")
if total_ratings > 0:
    print(f"  • Rationale 비율: {rationale_count/total_ratings*100:.1f}%")

# 클러스터 통계
clusters = comparison_result.get('rationale_clusters', [])
if clusters:
    print(f"\n🔍 클러스터링 통계:")
    print(f"  • 클러스터 수: {len(clusters)}개")
    print(f"\n  클러스터 목록:")
    for i, cluster in enumerate(clusters, 1):
        title = cluster.get('title', f'Cluster {i}')
        print(f"    {i}. {title}")
else:
    print(f"\n⚠️  클러스터링 정보 없음")

# 상위 5개 질문 출력
print(f"\n📋 상위 5개 질문 결과:")
for i, ex in enumerate(examples[:5], 1):
    print(f"\n  [{i}] {ex['input_text'][:60]}...")
    print(f"      점수: {ex['score']:.2f}", end="")
    if ex['score'] > 0.5:
        print(f" → Model A 승리")
    elif ex['score'] < -0.5:
        print(f" → Model B 승리")
    else:
        print(f" → 비슷함")

print(f"\n📁 출력 파일:")
print(f"  {output_file}")

# === 10. ⭐ VSCode에서 자동으로 웹 UI 열기 (Colab처럼!) ===
print("\n" + "=" * 80)
print("[10단계] VSCode에서 웹 UI 자동 실행")
print("=" * 80)

try:
    # ⭐ 교체한 comparison.py의 show_in_vscode() 함수 사용!
    comparison.show_in_vscode(output_file)
except KeyboardInterrupt:
    print("\n\n✅ 사용자가 서버를 종료했습니다.")
except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    print(f"\n수동으로 확인하려면:")
    print(f"  1. https://pair-code.github.io/llm-comparator/ 접속")
    print(f"  2. 'Load data' 버튼 클릭")
    print(f"  3. {output_file} 업로드")

print("\n" + "=" * 80)
print("✅ 모든 작업 완료!")
print("=" * 80)