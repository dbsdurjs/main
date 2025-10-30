# import sys
# sys.path.append('./llm-comparator-main/python/src/llm_comparator')
# import comparsion, model_helper, llm_judge_runner
from llm_comparator import comparison
from llm_comparator import model_helper
from llm_comparator import llm_judge_runner
from llm_comparator import rationale_bullet_generator
from llm_comparator import rationale_cluster_generator
import vertexai, os
from google.oauth2 import service_account


# === 1. 서비스 계정 키로 인증 ===
key_path = '/media/vcl/DATA/YG/armyclass2/army-class2-f4e4451b3fc9.json'
if not os.path.exists(key_path):
    raise FileNotFoundError(f"키 파일을 찾을 수 없음: {key_path}\n   파일 경로 확인하세요!")
print(f"키 파일 확인됨: {key_path}")
credentials = service_account.Credentials.from_service_account_file(
    key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# === 2. Vertex AI 초기화 ===
vertexai.init(
    project="army-class2",
    location="us-central1",
    credentials=credentials
)

# 프로젝트 ID와 지역 명시
MODEL_ID = "llama-3.1-8b-instruct-maas"
# vertexai.init(project="army-class2", location="asia-northeast3")
inputs = [
    {'prompt': 'how are you?', 'response_a': 'good', 'response_b': 'bad'},
    {'prompt': 'hello?', 'response_a': 'hello', 'response_b': 'hi'},
    {'prompt': 'what is the capital of korea?', 'response_a': 'Seoul', 'response_b': 'Vancouver'}
]

# Initialize the models-calling classes.
generator = model_helper.VertexGenerationModelHelper(MODEL_ID) # Initialize a model_helper.GenerationModelHelper() subclass
embedder = model_helper.VertexEmbeddingModelHelper() # Initialize a model_helper.EmbeddingModelHelper() subclass

# Initialize the instances that run work on the models.
judge = llm_judge_runner.LLMJudgeRunner(generator) # generator를 호출해서 판단 답변을 생성하도록 한 후 답변을 기반으로 출력을 구조화
bulletizer = rationale_bullet_generator.RationaleBulletGenerator(generator)
clusterer = rationale_cluster_generator.RationaleClusterGenerator(
    generator, embedder
)

# Configure and run the comparative evaluation.
comparison_result = comparison.run(inputs, judge, bulletizer, clusterer)

# Write the results to a JSON file that can be loaded in
# https://pair-code.github.io/llm-comparator
file_path = "./result.json"
comparison.write(comparison_result, file_path)