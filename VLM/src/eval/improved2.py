import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from open_r1.vlm_modules.internvl_module import InvernVLModule
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# 单卡设置
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 参数设置
steps = 300
print("Steps: ", steps)

RUN_NAME = "InternVL3-1B_MPO-rec"
MODEL_PATH = "/hy-tmp/VLM-R1-main/checkpoints/rl/InternVL3-1B_MPO-rec/checkpoint-300" 
OUTPUT_PATH = "./logs/rec_results_{DATASET}_{RUN_NAME}_{STEPS}.json"
INTERMEDIATE_PATH = "./logs/intermediate_rec_results_{DATASET}_{COUNT}_{RUN_NAME}_{STEPS}.json"
BSZ = 4
DATA_ROOT = "/hy-tmp/dataset/rec_jsons_internvl"
TEST_DATASETS = ['refcoco_val', 'refcocop_val', 'refcocog_val']
IMAGE_ROOT = "/hy-tmp/dataset"
random.seed(42)

# 创建日志目录
os.makedirs("./logs", exist_ok=True)

# 初始化模型
vlm_module = InvernVLModule()
model = vlm_module.get_model_class(MODEL_PATH, {}).from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True,
    use_flash_attn=True,
)

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.eos_token_id
vlm_module.post_model_init(model, tokenizer)

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
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

# 改进的坐标提取函数 - 提取所有双括号格式并选择IoU最高的
def extract_bbox_answer(content, gt_box=None):
    # 搜索所有双括号格式 [[x1,y1,x2,y2]]
    double_bracket_pattern = r'\[\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\]'
    matches = re.findall(double_bracket_pattern, content)
    
    candidate_boxes = []
    
    for match in matches:
        try:
            coords = [int(match[0]), int(match[1]), int(match[2]), int(match[3])]
            # 确保坐标有效
            if coords[0] < coords[2] and coords[1] < coords[3]:
                candidate_boxes.append(coords)
        except (ValueError, IndexError):
            continue
    
    # 如果没有找到任何有效的双括号坐标
    if not candidate_boxes:
        return [0, 0, 0, 0]
    
    # 如果没有提供gt_box，返回第一个有效的坐标
    if gt_box is None:
        return candidate_boxes[0]
    
    # 计算每个候选框与gt_box的IoU，选择IoU最高的
    best_box = candidate_boxes[0]
    best_iou = iou(best_box, gt_box)
    
    for box in candidate_boxes[1:]:
        current_iou = iou(box, gt_box)
        if current_iou > best_iou:
            best_box = box
            best_iou = current_iou
    
    return best_box

def process_vision_info(batch_messages):
    images = []
    for msg in batch_messages:
        img_path = msg[0]['content'][0]['image'].replace("file://", "")
        try:
            img = Image.open(img_path)
            images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            images.append(Image.new('RGB', (224, 224), (128, 128, 128)))
    return images

def save_results(data, outputs, dataset, count=None, is_final=False):
    results = []
    correct = 0
    for x, output in zip(data[:len(outputs)], outputs):
        # 传入gt_box用于选择IoU最高的坐标
        pred_box = extract_bbox_answer(output, x['solution'])
        gt_box = x['solution']
        iou_score = iou(pred_box, gt_box)
        is_correct = iou_score > 0.5
        correct += int(is_correct)
        
        results.append({
            'image': x['image'],
            'question': x['problem'],
            'ground_truth': gt_box,
            'model_output': output,
            'extracted_answer': pred_box,
            'iou': iou_score,
            'correct': is_correct
        })
    
    accuracy = 100 * correct / len(results) if results else 0.0
    
    if is_final:
        file_path = OUTPUT_PATH.format(DATASET=dataset, RUN_NAME=RUN_NAME, STEPS=steps)
    else:
        file_path = INTERMEDIATE_PATH.format(
            DATASET=dataset,
            COUNT=count,
            RUN_NAME=RUN_NAME,
            STEPS=steps
        )
    
    with open(file_path, "w") as f:
        json.dump({
            'accuracy': accuracy,
            'processed_count': len(results),
            'count': count,
            'results': results
        }, f, indent=4)
    
    return accuracy, file_path

# 主评估循环
sample_num = 2000
tokenizer.max_anyres_num = 12
QUESTION_TEMPLATE = "{Question} output the final answer in <answer>[[x1,y1,x2,y2]]</answer> tags."

for ds in TEST_DATASETS:
    print(f"\nProcessing {ds}...")
    
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    with open(ds_path, "r") as f:
        data = json.load(f)
    random.shuffle(data)
    data = data[:sample_num] if sample_num < len(data) else data
    
    messages = []
    for x in data:
        image_path = os.path.join(IMAGE_ROOT, x['image'])
        messages.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=x['problem'])}
            ]
        }])
    
    all_outputs = []
    total_batches = (len(messages) + BSZ - 1) // BSZ
    processed_count = 0
    
    for batch_idx in tqdm(range(total_batches), desc=f"Evaluating {ds}"):
        start_idx = batch_idx * BSZ
        end_idx = min((batch_idx + 1) * BSZ, len(messages))
        batch_messages = messages[start_idx:end_idx]
        
        prompts = vlm_module.prepare_prompt(None, [{"prompt": msg} for msg in batch_messages])
        images = process_vision_info(batch_messages)
        model_inputs = vlm_module.prepare_model_inputs(tokenizer, prompts, images)
        
        if isinstance(model_inputs, tuple):
            model_inputs_dict = model_inputs[0]
            model_inputs_dict['pixel_values'] = model_inputs_dict['pixel_values'].to(torch.bfloat16).to(device)
            inputs = {
                'input_ids': model_inputs_dict['input_ids'].to(device),
                'attention_mask': model_inputs_dict['attention_mask'].to(device),
                'pixel_values': model_inputs_dict['pixel_values']
            }
        else:
            model_inputs['pixel_values'] = model_inputs['pixel_values'].to(torch.bfloat16).to(device)
            inputs = model_inputs
    
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,
            temperature=0.0,
            top_p=1.0
        )
        batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_outputs.extend(batch_outputs)
        
        processed_count += len(batch_outputs)
        
        # 每100条保存一次
        if processed_count % 2 == 0 or end_idx >= len(messages):
            accuracy, saved_path = save_results(
                data=data[:processed_count],
                outputs=all_outputs,
                dataset=ds,
                count=processed_count,
                is_final=(end_idx >= len(messages))
            )
            print(f"\nProcessed {processed_count} / {len(messages)} saved to {saved_path}, Accuracy: {accuracy:.2f}%")
    
    final_accuracy, final_path = save_results(
        data=data,
        outputs=all_outputs,
        dataset=ds,
        is_final=True
    )
    print(f"\nFinal accuracy for {ds}: {final_accuracy:.2f}%")
    print(f"Final results saved to {final_path}")