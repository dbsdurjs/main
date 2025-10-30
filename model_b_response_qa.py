#--------------------------------------------------------------------------------------------------------------------
# model name : Qwen/Qwen3-8B
#--------------------------------------------------------------------------------------------------------------------
import csv, os
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"]= "1"
# Load model and tokenizer
model_id = "LiquidAI/LFM2-2.6B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16",
#    attn_implementation="flash_attention_2" <- uncomment on compatible GPU
)

def read_prompts_from_csv(file_path):
    prompts = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            prompts.append(row[0])
    return prompts

def read_prompts_from_txt(file_path):
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            prompts.append(line.strip())
                
    return prompts

def get_llm_response(prompt):

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Generate answer
    full_prompt = f"{prompt}. 이 질문에 대한 답변을 4~5문장 이내로 한국어로 완전하게 작성해줘."

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": full_prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(model.device)

    output = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=512,
    )
    generated_ids = output[0][input_ids.shape[-1]:]  # 입력 길이 이후만 추출
    # print(tokenizer.decode(generated_ids, skip_special_tokens=True))

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

if __name__ == "__main__":
    file_path = "./defense_questions_snunlp.txt"
    output_path = "./defense_response_model_b_snunlp.txt"

    if file_path.endswith('.csv'):
        prompts = read_prompts_from_csv(file_path)
    elif file_path.endswith('.txt'):
        prompts = read_prompts_from_txt(file_path)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다.")

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for prompt in prompts:
            answer = get_llm_response(prompt)
            print(f"Prompt: {prompt}\nResponse: {answer}\n")
            out_f.write(f"{{prompt:{prompt}, response_b:{answer}}}\n")