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
MODEL_PATH = "/hy-tmp/InternVL3-1B" 
OUTPUT_PATH = "./logs/rec_results_{DATASET}_{RUN_NAME}_{STEPS}.json"
INTERMEDIATE_PATH = "./logs/intermediate_rec_results_{DATASET}_{BATCH_NUM}_{RUN_NAME}_{STEPS}.json"
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
model.generation_config.pad_token_id = tokenizer.pad_token_id
vlm_module.post_model_init(model, tokenizer)

# 增强坐标提取函数 - 处理各种模糊表述
def extract_bbox_answer(content):
    # 模式1: 标准格式 [x1,y1,x2,y2]
    bbox_pattern1 = r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]'
    # 模式2: 无括号格式 x1,y1,x2,y2
    bbox_pattern2 = r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)'
    # 模式3: 带描述性文本的格式
    bbox_pattern3 = r'x1\s*[:=]?\s*(\d+)\s*[,]?\s*y1\s*[:=]?\s*(\d+)\s*[,]?\s*x2\s*[:=]?\s*(\d+)\s*[,]?\s*y2\s*[:=]?\s*(\d+)'
    # 模式4: 其他可能的格式
    bbox_pattern4 = r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)'
    # 模式5: 点对点格式 (x1,y1) to (x2,y2)
    bbox_pattern5 = r'\((\d+)\s*,\s*(\d+)\)\s*to\s*\((\d+)\s*,\s*(\d+)\)'
    # 模式6: 点对点无括号格式
    bbox_pattern6 = r'(\d+)\s*,\s*(\d+)\s*to\s*(\d+)\s*,\s*(\d+)'
    # 模式7: JSON格式的边界框
    bbox_pattern7 = r'"left"\s*:\s*(\d+\.?\d*)[,\s]+"right"\s*:\s*(\d+\.?\d*)[,\s]+"top"\s*:\s*(\d+\.?\d*)[,\s]+"bottom"\s*:\s*(\d+\.?\d*)'
    # 模式8: 近似坐标表述
    bbox_pattern8 = r'approximately\s*\((\d+)\s*,\s*(\d+)\)\s*to\s*\((\d+)\s*,\s*(\d+)\)'
    
    patterns = [
        (bbox_pattern1, 4),
        (bbox_pattern2, 4),
        (bbox_pattern3, 4),
        (bbox_pattern4, 4),
        (bbox_pattern5, 4),
        (bbox_pattern6, 4),
        (bbox_pattern7, 4),
        (bbox_pattern8, 4)
    ]
    
    # 尝试在<answer>标签内查找
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_tag_pattern, content, re.DOTALL | re.IGNORECASE)
    search_content = answer_match.group(1).strip() if answer_match else content
    
    # 尝试所有模式
    for pattern, count in patterns:
        matches = re.findall(pattern, search_content)
        if matches:
            for match in matches:
                try:
                    coords = [int(float(match[i])) for i in range(count)]
                    # 验证坐标有效性
                    if coords[0] < coords[2] and coords[1] < coords[3]:
                        return coords
                except (ValueError, IndexError):
                    continue
    
    # 尝试提取数字序列
    num_pattern = r'\b\d{3,4}\b'  # 匹配3-4位数字
    numbers = re.findall(num_pattern, search_content)
    if len(numbers) >= 4:
        try:
            coords = [int(numbers[i]) for i in range(4)]
            if coords[0] < coords[2] and coords[1] < coords[3]:
                return coords
        except (ValueError, IndexError):
            pass
    
    # 尝试提取点对点坐标
    point_pair_pattern = r'\((\d+)\s*,\s*(\d+)\)'
    points = re.findall(point_pair_pattern, search_content)
    if len(points) >= 2:
        try:
            x1, y1 = map(int, points[0])
            x2, y2 = map(int, points[1])
            coords = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            if coords[0] < coords[2] and coords[1] < coords[3]:
                return coords
        except (ValueError, IndexError):
            pass
    
    # 尝试提取单独的数字对
    num_pair_pattern = r'(\d+)\s*,\s*(\d+)'
    pairs = re.findall(num_pair_pattern, search_content)
    if len(pairs) >= 2:
        try:
            x1, y1 = map(int, pairs[0])
            x2, y2 = map(int, pairs[1])
            coords = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            if coords[0] < coords[2] and coords[1] < coords[3]:
                return coords
        except (ValueError, IndexError):
            pass
    
    # 最后尝试在文本中搜索四个连续的数字
    any_num_pattern = r'\b(\d{3,4})\b'
    numbers = re.findall(any_num_pattern, search_content)
    if len(numbers) >= 4:
        try:
            coords = [int(numbers[i]) for i in range(4)]
            if coords[0] < coords[2] and coords[1] < coords[3]:
                return coords
        except (ValueError, IndexError):
            pass
    
    return [0, 0, 0, 0]  # 默认值

def iou(box1, box2):
    # 确保坐标顺序正确
    box1 = [min(box1[0], box1[2]), min(box1[1], box1[3]),
            max(box1[0], box1[2]), max(box1[1], box1[3])]
    box2 = [min(box2[0], box2[2]), min(box2[1], box2[3]),
            max(box2[0], box2[2]), max(box2[1], box2[3])]
    
    # 计算交集
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算并集
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def process_vision_info(batch_messages):
    images = []
    for msg in batch_messages:
        img_path = msg[0]['content'][0]['image'].replace("file://", "")
        try:
            img = Image.open(img_path)
            images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 创建空白图像作为占位符
            images.append(Image.new('RGB', (224, 224), (128, 128, 128)))
    return images

def save_results(data, outputs, dataset, batch_num=None, is_final=False):
    results = []
    correct = 0
    for x, output in zip(data[:len(outputs)], outputs):
        pred_box = extract_bbox_answer(output)
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
            BATCH_NUM=batch_num,
            RUN_NAME=RUN_NAME,
            STEPS=steps
        )
    
    with open(file_path, "w") as f:
        json.dump({
            'accuracy': accuracy,
            'processed_count': len(results),
            'batch_num': batch_num,
            'results': results
        }, f, indent=4)
    
    return accuracy, file_path

# 主评估循环
sample_num = 2000
tokenizer.max_anyres_num = 12
QUESTION_TEMPLATE = "{Question} output the final answer in <answer>[x1,y1,x2,y2]</answer> tags."

for ds in TEST_DATASETS:
    print(f"\nProcessing {ds}...")
    
    # 加载数据
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    with open(ds_path, "r") as f:
        data = json.load(f)
    random.shuffle(data)
    data = data[:sample_num] if sample_num < len(data) else data
    

    # 准备输入
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
    
    # 模型推理
    all_outputs = []
    total_batches = (len(messages) + BSZ - 1) // BSZ
    
    for batch_idx in tqdm(range(total_batches), desc=f"Evaluating {ds}"):
        start_idx = batch_idx * BSZ
        end_idx = min((batch_idx + 1) * BSZ, len(messages))
        batch_messages = messages[start_idx:end_idx]
        
        # 准备prompt和图像
        prompts = vlm_module.prepare_prompt(None, [{"prompt": msg} for msg in batch_messages])
        images = process_vision_info(batch_messages)
    
        # 获取模型输入
        model_inputs = vlm_module.prepare_model_inputs(tokenizer, prompts, images)
        
        # 统一处理输入格式
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
    
        # 生成输出
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # 减少最大token数量
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,  # 使用贪心搜索
            temperature=0.0,
            top_p=1.0
        )
        batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_outputs.extend(batch_outputs)
        
        # 每批保存一次中间结果
        accuracy, saved_path = save_results(
            data=data[start_idx:end_idx],
            outputs=batch_outputs,
            dataset=ds,
            batch_num=batch_idx,
            is_final=(end_idx >= len(messages))
        )
        print(f"\nBatch {batch_idx+1}/{total_batches} saved to {saved_path}, Accuracy: {accuracy:.2f}%")
    
    # 最终保存
    final_accuracy, final_path = save_results(
        data=data,
        outputs=all_outputs,
        dataset=ds,
        is_final=True
    )
    print(f"\nFinal accuracy for {ds}: {final_accuracy:.2f}%")
    print(f"Final results saved to {final_path}")