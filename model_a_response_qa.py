#--------------------------------------------------------------------------------------------------------------------
# model name : meta-llama/Llama-3.1-8B
#--------------------------------------------------------------------------------------------------------------------
import transformers, torch
import csv, os

os.environ["CUDA_VISIBLE_DEVICES"]= "0"

model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"dtype": torch.float16}, device_map="auto")
pipeline.model.eval()

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
    instruction = '이 질문에 대한 답변을 4~5문장 이내로 한국어로 완전하게 작성해줘.'

    messages = [
        {"role" : "system", "content" : f"{prompt}"},
        {"role" : "user", "content" : f"{instruction}"}
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    response = outputs[0]['generated_text'][len(prompt):]
   
    return response.strip()


if __name__ == "__main__":
    file_path = "./defense_questions_snunlp.txt"
    output_path = "./defense_response_model_a_snunlp.txt"

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
            out_f.write(f"{{prompt:{prompt}, response_a:{answer}}}\n")