import torch
import json
from tqdm import tqdm
import re
import os
import random
from transformers import AutoTokenizer
from open_r1.vlm_modules.internvl_module import InvernVLModule
from PIL import Image
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# 单卡设置
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 参数设置
steps = 300
print("Steps: ", steps)

RUN_NAME = "InternVL3-1B_MPO-rec"
MODEL_PATH = "/hy-tmp/InternVL3-1B" 
OUTPUT_PATH = "./logs/rec_results_{DATASET}_{RUN_NAME}_{STEPS}.json"
INTERMEDIATE_PATH = "./logs/intermediate_rec_results_{DATASET}_{BATCH_NUM}_{RUN_NAME}_{STEPS}.json"
BSZ = 4
DATA_ROOT = "/hy-tmp/dataset/rec_jsons_internvl"
TEST_DATASETS = ['refcoco_val', 'refcocop_val', 'refcocog_val']
IMAGE_ROOT = "/hy-tmp/dataset"
sample_num = 2000
random.seed(42)

# 创建日志目录
os.makedirs("./logs", exist_ok=True)

# 初始化模型和tokenizer
vlm_module = InvernVLModule()
model = vlm_module.get_model_class(MODEL_PATH, {}).from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True,
    use_flash_attn=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id
vlm_module.post_model_init(model, tokenizer)
tokenizer.max_anyres_num = 12

QUESTION_TEMPLATE = "{Question} output the final answer in <answer>[x1,y1,x2,y2]</answer> tags."

# ---------------- 辅助函数 ----------------
def extract_bbox_answer(content):
    # 保留你原来的增强提取逻辑
    patterns = [
        r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]',
        r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
        r'x1\s*[:=]?\s*(\d+)\s*[,]?\s*y1\s*[:=]?\s*(\d+)\s*[,]?\s*x2\s*[:=]?\s*(\d+)\s*[,]?\s*y2\s*[:=]?\s*(\d+)',
        r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',
        r'\((\d+)\s*,\s*(\d+)\)\s*to\s*\((\d+)\s*,\s*(\d+)\)',
        r'(\d+)\s*,\s*(\d+)\s*to\s*(\d+)\s*,\s*(\d+)',
        r'"left"\s*:\s*(\d+\.?\d*)[,\s]+"right"\s*:\s*(\d+\.?\d*)[,\s]+"top"\s*:\s*(\d+\.?\d*)[,\s]+"bottom"\s*:\s*(\d+\.?\d*)',
        r'approximately\s*\((\d+)\s*,\s*(\d+)\)\s*to\s*\((\d+)\s*,\s*(\d+)\)',
    ]
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
    search_content = answer_match.group(1).strip() if answer_match else content
    for pat in patterns:
        matches = re.findall(pat, search_content)
        for match in matches:
            try:
                coords = [int(float(m)) for m in match]
                if coords[0] < coords[2] and coords[1] < coords[3]:
                    return coords
            except:
                continue
    # 最后尝试连续数字提取
    numbers = re.findall(r'\b\d{3,4}\b', search_content)
    if len(numbers) >= 4:
        coords = [int(numbers[i]) for i in range(4)]
        if coords[0] < coords[2] and coords[1] < coords[3]:
            return coords
    return [0,0,0,0]

def iou(box1, box2):
    box1 = [min(box1[0], box1[2]), min(box1[1], box1[3]),
            max(box1[0], box1[2]), max(box1[1], box1[3])]
    box2 = [min(box2[0], box2[2]), min(box2[1], box2[3]),
            max(box2[0], box2[2]), max(box2[1], box2[3])]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter = (x_right - x_left)*(y_bottom - y_top)
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return inter/union if union>0 else 0.0

def process_vision_info(batch_messages):
    images = []
    for msg in batch_messages:
        img_path = msg[0]['content'][0]['image'].replace("file://","")
        try:
            img = Image.open(img_path)
            images.append(img)
        except:
            images.append(Image.new('RGB',(224,224),(128,128,128)))
    return images

def save_results(data, outputs, dataset, batch_num=None, is_final=False):
    results = []
    for x, output_list in zip(data, outputs):
        iou_scores = [iou(extract_bbox_answer(out), x['solution']) for out in output_list]
        mean_iou = float(np.mean(iou_scores))
        var_iou = float(np.var(iou_scores))
        correct = sum([1 for s in iou_scores if s>0.5])
        results.append({
            'image': x['image'],
            'question': x['problem'],
            'ground_truth': x['solution'],
            'model_output': output_list,   # 8 次输出
            'iou_scores': iou_scores,      # 8 次 IoU
            'iou_var': var_iou,            # 方差
            'mean_iou': mean_iou,
            'correct_count': correct
        })
    accuracy = 100 * sum([1 for r in results if r['mean_iou']>0.5])/len(results) if results else 0.0
    file_path = OUTPUT_PATH.format(DATASET=dataset, RUN_NAME=RUN_NAME, STEPS=steps) if is_final else INTERMEDIATE_PATH.format(DATASET=dataset,BATCH_NUM=batch_num,RUN_NAME=RUN_NAME,STEPS=steps)
    with open(file_path,"w") as f:
        json.dump({'accuracy':accuracy,'processed_count':len(results),'batch_num':batch_num,'results':results},f,indent=4)
    return accuracy, file_path

# ---------------- 主评估循环 ----------------
for ds in TEST_DATASETS:
    print(f"\nProcessing {ds}...")
    ds_path = os.path.join(DATA_ROOT,f"{ds}.json")
    with open(ds_path,"r") as f:
        data = json.load(f)
    random.shuffle(data)
    data = data[:sample_num] if sample_num<len(data) else data

    messages = []
    for x in data:
        image_path = os.path.join(IMAGE_ROOT, x['image'])
        messages.append([{
            "role":"user",
            "content":[
                {"type":"image","image":f"file://{image_path}"},
                {"type":"text","text":QUESTION_TEMPLATE.format(Question=x['problem'])}
            ]
        }])
    
    total_batches = (len(messages)+BSZ-1)//BSZ
    all_outputs = []

    for batch_idx in tqdm(range(total_batches), desc=f"Evaluating {ds}"):
        start_idx = batch_idx*BSZ
        end_idx = min((batch_idx+1)*BSZ,len(messages))
        batch_messages = messages[start_idx:end_idx]
        prompts = vlm_module.prepare_prompt(None,[{"prompt":msg} for msg in batch_messages])
        images = process_vision_info(batch_messages)
        model_inputs = vlm_module.prepare_model_inputs(tokenizer, prompts, images)
        if isinstance(model_inputs, tuple):
            model_inputs_dict = model_inputs[0]
            model_inputs_dict['pixel_values'] = model_inputs_dict['pixel_values'].to(torch.bfloat16).to(device)
            inputs = {'input_ids':model_inputs_dict['input_ids'].to(device),
                      'attention_mask':model_inputs_dict['attention_mask'].to(device),
                      'pixel_values':model_inputs_dict['pixel_values']}
        else:
            model_inputs['pixel_values'] = model_inputs['pixel_values'].to(torch.bfloat16).to(device)
            inputs = model_inputs

        # 每个问题 8 次推理
        batch_outputs = []
        for repeat in range(8):
            #outputs = model.generate(**inputs,max_new_tokens=256,do_sample=False,
                                     #pad_token_id=tokenizer.eos_token_id,num_beams=1,temperature=0.0,top_p=1.0)#####原参数
            outputs = model.generate(**inputs,max_new_tokens=256,do_sample=True,
                                    pad_token_id=tokenizer.eos_token_id,num_beams=1,temperature=0.7,top_p=0.9)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            if repeat==0:
                for i, d in enumerate(decoded):
                    batch_outputs.append([d])
            else:
                for i, d in enumerate(decoded):
                    batch_outputs[i].append(d)

        all_outputs.extend(batch_outputs)

        # 保存中间结果
        accuracy, saved_path = save_results(data=data[start_idx:end_idx],outputs=batch_outputs,dataset=ds,batch_num=batch_idx,is_final=(end_idx>=len(messages)))
        print(f"\nBatch {batch_idx+1}/{total_batches} saved to {saved_path}, Accuracy: {accuracy:.2f}%")

    # 保存最终结果
    final_accuracy, final_path = save_results(data=data,outputs=all_outputs,dataset=ds,is_final=True)
    print(f"\nFinal accuracy for {ds}: {final_accuracy:.2f}%")
    print(f"Final results saved to {final_path}")
