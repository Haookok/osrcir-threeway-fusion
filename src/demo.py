import argparse
import prompts
import os
import pickle
import json
import tqdm
import cloudgpt_api
import torch
import datasets
import compute_results
import termcolor
from typing import Optional, Tuple, List, Dict, Union
import data_utils
import numpy as np
import json
import PIL
import clip

def parser_args():
    """
    统一定义这个脚本的命令行参数。

    这个函数的作用可以简单理解为：
    “告诉脚本，用户运行 demo 时可以传哪些开关，以及这些开关分别控制什么”。

    后面整条实验流程中会反复用到这些参数，例如：
    - 选哪个数据集
    - 用哪个多模态模型生成 target description
    - 用哪个 CLIP 模型做检索
    - 是否只跑前 N 条样本做小规模测试
    - 是否进一步计算 recall / mAP 这类指标
    """
    parser = argparse.ArgumentParser('')

    # preload 相关参数控制“中间结果是否缓存到本地”。
    # 这样当你重复跑实验时，就不需要每次都重新提图像特征、重新生成描述。
    parser.add_argument("--preload", nargs='+', type=str, default=['img_features','captions','mods'],
                        help='List of properties to preload is computed once before.')
    parser.add_argument("--preload_path", type=str, default="./precomputed_cache",
                        help='preload file path.')

    # device 表示如果机器上有 CUDA GPU，优先使用哪一张卡。
    # 如果没有 GPU，后面主程序会自动退回到 CPU。
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index to use when GPU is available.")

    # clip 参数决定“后半段检索”用哪个视觉-语言模型。
    # 这里既支持官方 CLIP，也支持 OpenCLIP 的模型名。
    parser.add_argument("--clip", type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50x4', 'ViT-bigG-14',
                                 'ViT-B-32','ViT-B-16','ViT-L-14','ViT-H-14','ViT-g-14'],
                        help="Which CLIP text-to-image retrieval model to use"),

    # blip 参数在当前这条主通路里基本没有实际参与推理，
    # 但它来自原项目设计，所以这里暂时保留。
    parser.add_argument("--blip", type=str, default='blip2_t5', choices=['blip2_t5'],
                        help="BLIP Image Caption Model to use.")

    # dataset / split / dataset-path 共同决定：
    # “我们要在哪个 benchmark 上跑实验，以及去哪里读数据”。
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['cirr', 'circo',
                                 'fashioniq_dress', 'fashioniq_toptee', 'fashioniq_shirt',
                                 'genecis_change_attribute', 'genecis_change_object', 'genecis_focus_attribute', 'genecis_focus_object'],
                        help="Dataset to use")
    parser.add_argument("--split", type=str, default='val', choices=['val', 'test'],
                        help='Dataset split to evaluate on. Some datasets require special testing protocols s.a. cirr/circo.')
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the dataset")

    # gpt_cir_prompt 决定“怎么问多模态大模型”。
    # 你可以把它看成论文里 Reflective CoT 的提示词模板入口。
    available_prompts = [f'prompts.{x}' for x in prompts.__dict__.keys() if '__' not in x]
    parser.add_argument("--gpt_cir_prompt", default='prompts.mllm_structural_predictor_prompt_CoT', type=str, choices=available_prompts,
                        help='Denotes the base prompt to use alongside GPT4V. Has to be available in prompts.py')

    # openai_engine 在你当前的用法里，其实已经不只是 OpenAI 了，
    # 更准确地说，它表示“生成 target description 时所调用的多模态模型名”。
    parser.add_argument("--openai_engine", default='qwen-vl-max-latest', type=str,
                        help='Vision-language model name to use through the compatible API endpoint.')

    # batch_size 既影响生成描述时 DataLoader 的吞吐，也影响后面 CLIP 编码时的吞吐。
    parser.add_argument("--batch_size", default=32, type=int,
                        help='Batch size to use.')

    # max_samples 是一个非常实用的小开关：
    # 在正式跑全量实验前，可以只跑前 N 条样本，先检查流程是否通、结果是否合理。
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Only run the first N query samples for quick testing.")

    # output_json 用于把每条样本的过程和结果保存下来，
    # 方便你后面逐条看：instruction、thoughts、reflections、最终描述、检索排名等。
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional path to save generated target descriptions as JSON.")

    # compute_metrics 控制是否继续做“后半段检索评估”。
    # 不开它：只生成 target description。
    # 开了它：会继续用 CLIP 做检索，并计算 recall / mAP。
    parser.add_argument("--compute_metrics", action="store_true",
                        help="Encode target descriptions with CLIP and compute retrieval metrics.")

    # save_topk 决定每条样本最终在 json 里保留多少个检索候选，便于人工检查。
    parser.add_argument("--save_topk", type=int, default=10,
                        help="How many top retrieval candidates to save per sample.")
    args = parser.parse_args()
    return args

args = parser_args()

def get_predeal_dict():
    """
    生成缓存文件路径字典。

    这个函数的核心作用不是“做计算”，而是：
    提前约定好各类中间结果应该存到哪里。

    这里主要管理三类缓存：
    - img_features: 图库图像经过 CLIP 编码后的特征
    - captions: 预留的 caption 缓存（当前 demo 中不是重点）
    - mods: 多模态模型生成出的 target descriptions 及其推理结果

    这样设计的好处是：
    同一组实验参数下，重复运行时可以直接读取之前的中间结果，节省大量时间。
    """
    ### Argument Checks.
    preload_dict = {key: None for key in ['img_features', 'captions', 'mods']}
    preload_str = f'{args.dataset}_{args.openai_engine}_{args.clip}_{args.split}'.replace('/', '-') # fashioniq_dress_blip2_t5_ViT-g-14_val
    print(preload_str)

    if len(args.preload):
        os.makedirs(os.path.join(args.preload_path, 'precomputed'), exist_ok=True)
    if 'img_features' in args.preload:
        # # CLIP embeddings only have to be computed when CLIP model changes.
        # img_features_load_str = f'{args.dataset}_{args.clip}_{args.split}'.replace('/', '-')
        preload_dict['img_features'] = os.path.join(args.preload_path, 'precomputed', preload_str + '_img_features.pkl')

    if 'captions' in args.preload:
        # Cache key is based on the prompt actually used by this demo.
        caption_load_str = f'{args.dataset}_{args.openai_engine}_{args.split}'.replace('/', '-')
        preload_dict['captions'] = os.path.join(
            args.preload_path,
            'precomputed',
            caption_load_str + f'_captions_{args.gpt_cir_prompt.split(".")[-1]}.pkl'
        )

    if 'mods' in args.preload:
        # # LLM-based caption modifications have to be queried only when BLIP model or BLIP prompt changes.
        mod_load_str = f'{args.dataset}_{args.split}'.replace('/', '-')
        preload_dict['mods'] = os.path.join(args.preload_path, 'precomputed',
                                            mod_load_str + f'_mods_{args.gpt_cir_prompt.split(".")[-1]}.pkl')
        if args.openai_engine:
            preload_dict['mods'] = preload_dict['mods'].replace('.pkl', f'_{args.openai_engine}.pkl')

    if args.split == 'test':
        preload_dict['test'] = preload_str + f'_{args.gpt_cir_prompt.split(".")[-1]}_test_submission.json'
    return preload_dict


def sidecar_json_path(base_path: Optional[str], suffix: str) -> Optional[str]:
    """
    根据主输出文件路径，自动推导“附属结果文件”的路径。

    例如：
    - 主文件：results.json
    - 指标文件：results_metrics.json
    - 排名文件：results_ranking.jsonl

    这样可以让同一轮实验的文件放在一起，便于管理。
    """
    if not base_path:
        return None
    if base_path.endswith(".json"):
        return base_path[:-5] + suffix
    return base_path + suffix


def save_json(path: Optional[str], payload):
    """
    一个简单的 JSON 保存工具函数。

    主要作用：
    - 自动创建目录
    - 用 utf-8 保存中文
    - 用缩进格式化，方便人工阅读
    """
    if not path:
        return
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_image_path_for_loading(image_path: Union[str, os.PathLike]) -> str:
    """
    【兼容 jpg/png 的新增辅助函数】

    这个函数专门给“后半段检索时真正打开图片文件”使用。
    原因是：
    - 数据集层已经尽量返回正确路径
    - 但为了保险，真正 PIL.Image.open 之前再做一次后缀兜底

    处理逻辑：
    1. 原路径存在 -> 直接用
    2. 原路径不存在 -> 依次尝试 .png / .jpg / .jpeg
    3. 仍不存在 -> 返回原路径，让后续报错，便于定位真实缺图
    """
    candidate_path = os.fspath(image_path)
    if os.path.exists(candidate_path):
        return candidate_path

    stem, _ = os.path.splitext(candidate_path)
    for suffix in [".png", ".jpg", ".jpeg"]:
        fallback_path = stem + suffix
        if os.path.exists(fallback_path):
            return fallback_path

    return candidate_path


def load_and_preprocess_image_safely(image_path: Union[str, os.PathLike], preprocess):
    """
    【缺图容错新增函数】

    这个函数专门用于“后半段做图库 embedding 时安全地打开图片”。

    它解决的是另一类问题：
    - 前面我们已经解决了 jpg/png 后缀不一致
    - 但如果图片真的不存在，原逻辑会直接 FileNotFoundError 让整轮实验退出

    这里的新策略是：
    1. 先尝试自动解析真实后缀
    2. 如果文件存在，就正常打开并做预处理
    3. 如果文件仍不存在，返回 None，并把缺图路径打印出来

    这样后面就可以“跳过缺图”，而不是“缺一张图整轮崩掉”。
    """
    resolved_path = resolve_image_path_for_loading(image_path)
    if not os.path.exists(resolved_path):
        print(f"[WARN] Missing image skipped during index encoding: {resolved_path}")
        return None, os.fspath(image_path)

    image_tensor = preprocess(PIL.Image.open(resolved_path).convert('RGB'))
    return image_tensor, resolved_path


def load_retrieval_model(args: argparse.Namespace, device: torch.device):
    """
    加载“后半段检索”要用到的 CLIP / OpenCLIP 模型。

    整个项目可以拆成两半：
    1. 多模态大模型负责生成 target description
    2. CLIP 负责把“描述”和“候选图片”映射到同一个向量空间里，再做相似度检索

    这个函数做的就是第 2 部分的模型准备。
    返回三个东西：
    - clip_model: 真正负责编码图像/文本的模型
    - preprocess: 图像预处理方法
    - tokenizer: 文本分词器
    """
    if args.clip in ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50x4']:
        clip_model, preprocess = clip.load(args.clip, device=device, jit=False)
        tokenizer = lambda texts: clip.tokenize(texts, context_length=77, truncate=True)
    else:
        import open_clip

        pretrained_map = {
            'ViT-B-32': 'laion2b_s34b_b79k',
            'ViT-B-16': 'laion2b_s34b_b88k',
            'ViT-L-14': 'laion2b_s32b_b82k',
            'ViT-H-14': 'laion2b_s32b_b79k',
            'ViT-g-14': 'laion2b_s34b_b88k',
            'ViT-bigG-14': 'laion2b_s39b_b160k',
        }
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            args.clip,
            pretrained=pretrained_map[args.clip],
            device=device,
        )
        tokenizer = open_clip.get_tokenizer(args.clip)

    clip_model.eval()
    return clip_model, preprocess, tokenizer


def build_target_dataset(args: argparse.Namespace, preprocess):
    """
    构造“图库数据集（candidate image set）”。

    前面生成 target description 时，使用的是 query_dataset，
    里面每条样本包含“参考图 + 修改文本 + 正确目标图”。

    但做检索时，我们需要另一种数据集：
    只包含“所有候选图片本身”，这样才能把整个图库都编码成特征，和文本描述做匹配。
    这就是 classic 模式数据集的作用。
    """
    if 'fashioniq' in args.dataset.lower():
        dress_type = args.dataset.split('_')[-1]
        return datasets.FashionIQDataset(args.dataset_path, args.split, [dress_type], 'classic', preprocess=preprocess)
    if args.dataset.lower() == 'cirr':
        split = 'test1' if args.split == 'test' else args.split
        return datasets.CIRRDataset(args.dataset_path, split, 'classic', preprocess=preprocess)
    if args.dataset.lower() == 'circo':
        return datasets.CIRCODataset(args.dataset_path, args.split, 'classic', preprocess=preprocess)
    return None


@torch.no_grad()
def extract_index_features(device: torch.device, dataset: torch.utils.data.Dataset, clip_model, preprocess, batch_size: int, preload: Optional[str] = None):
    """
    计算图库里所有候选图片的 CLIP 特征。

    这一步可以理解成：
    “先把所有候选图都翻译成一串向量数字，后面检索时就不用重复算了。”

    如果 preload 文件已经存在，会直接读取缓存；
    如果不存在，就现场逐批编码。
    """
    if preload is not None and os.path.exists(preload):
        print(f'Loading precomputed image features from {preload}!')
        extracted_data = pickle.load(open(preload, 'rb'))
        return extracted_data['index_features'], extracted_data['index_names']

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=False,
        collate_fn=data_utils.collate_fn,
        shuffle=False
    )

    index_features, index_names = [], []
    missing_index_images = []
    for batch in tqdm.tqdm(loader, desc='Encoding index images'):
        # 不同数据集的 classic 模式返回格式略有区别：
        # 有的直接返回已经预处理好的 tensor（image），
        # 有的返回原始路径（image_path），这里统一兼容。
        if 'image' in batch:
            images = batch['image']
            names = batch['image_name']
        elif 'image_path' in batch:
            image_paths = batch['image_path']
            processed_images = []
            valid_names = []
            for path, name in zip(image_paths, batch['image_name']):
                # ==================== 缺图容错修改开始 ====================
                # 现在这里不再假设“每一张图都一定存在”。
                # 如果某张图缺失：
                # - 不再直接 FileNotFoundError 退出
                # - 而是记录下来并跳过
                image_tensor, resolved_or_original_path = load_and_preprocess_image_safely(path, preprocess)
                if image_tensor is None:
                    missing_index_images.append({
                        "image_name": name,
                        "missing_path": resolved_or_original_path,
                    })
                    continue
                processed_images.append(image_tensor)
                valid_names.append(name)
                # ==================== 缺图容错修改结束 ====================

            if not processed_images:
                # 如果这一整个 batch 都是缺图，就直接跳过这个 batch。
                continue

            images = torch.stack(processed_images)
            names = valid_names
        else:
            raise ValueError('Unsupported batch format for image feature extraction.')

        batch_features = clip_model.encode_image(images.to(device)).float().cpu()
        index_features.append(batch_features)
        index_names.extend(names)

    if not index_features:
        raise RuntimeError("No valid index images were found for retrieval. Please check your dataset paths.")

    index_features = torch.vstack(index_features)
    if preload is not None:
        pickle.dump({
            'index_features': index_features,
            'index_names': index_names,
            'missing_index_images': missing_index_images
        }, open(preload, 'wb'))
    return index_features, index_names


@torch.no_grad()
def encode_text_features(device: torch.device, clip_model, tokenizer, input_captions: List[str], batch_size: int = 32):
    """
    把 target descriptions 编码成 CLIP 文本特征。

    你可以把这一步理解成：
    “把千问写出来的那句话，也翻译成和图片同一空间里的向量。”

    这样后面才能和图库图片向量直接做相似度比较。
    """
    predicted_features = []
    n_iter = int(np.ceil(len(input_captions) / batch_size))
    for i in tqdm.trange(n_iter, position=0, desc='Encoding target descriptions'):
        captions_to_use = input_captions[i * batch_size:(i + 1) * batch_size]
        tokenized_input_captions = tokenizer(captions_to_use).to(device)
        clip_text_features = clip_model.encode_text(tokenized_input_captions).float().cpu()
        predicted_features.append(clip_text_features)
    return torch.vstack(predicted_features)


def attach_retrieval_details(args: argparse.Namespace, generated_results: List[Dict[str, Union[str, int]]], predicted_features: torch.Tensor,
                             index_features: torch.Tensor, index_names: List[str], target_names: List[str],
                             reference_names: List[str], targets: List, save_topk: int):
    """
    把检索结果补回到每条样本的 json 记录里。

    为什么要做这一步？
    因为光有整体 recall 数字还不够，我们还想逐条样本看：
    - 模型检索出来的前几名是谁
    - 正确答案排在第几
    - 是否 hit@1 / hit@5 / hit@10 / hit@50

    这样你就能从“总体指标”下钻到“单条样本分析”。
    """
    if not generated_results:
        return generated_results

    predicted_features = torch.nn.functional.normalize(predicted_features.float(), dim=-1)
    index_features = torch.nn.functional.normalize(index_features.float(), dim=-1)
    similarity = predicted_features @ index_features.T
    sorted_indices = torch.argsort(1 - similarity, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    if args.dataset.lower() == 'cirr':
        reference_mask = torch.tensor(
            sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(reference_names), -1)
        )
        sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)

    for idx, result in enumerate(generated_results):
        ranked_names = sorted_index_names[idx].tolist()
        result['retrieved_topk'] = ranked_names[:save_topk]

        if idx < len(target_names):
            target_name = target_names[idx]
            result['target_rank'] = ranked_names.index(target_name) + 1 if target_name in ranked_names else None
            if result['target_rank'] is not None:
                result['hit@1'] = result['target_rank'] <= 1
                result['hit@5'] = result['target_rank'] <= 5
                result['hit@10'] = result['target_rank'] <= 10
                result['hit@50'] = result['target_rank'] <= 50

        if idx < len(targets) and args.dataset.lower() in ['cirr', 'circo']:
            filtered_targets = [item for item in list(targets[idx]) if item != '']
            result['ground_truth_candidates'] = filtered_targets
            if filtered_targets:
                gt_ranks = [ranked_names.index(item) + 1 for item in filtered_targets if item in ranked_names]
                if gt_ranks:
                    result['best_ground_truth_rank'] = min(gt_ranks)

    return generated_results


def filter_result_payload_for_available_targets(args: argparse.Namespace, result_payload: Dict, index_names: List[str]):
    """
    【缺图容错新增函数】

    这个函数用于在正式计算指标前，过滤掉“目标图根本不在当前图库里”的 query。

    为什么要做这一步？
    因为即使我们已经允许“缺图跳过”，仍然会出现一种情况：
    - 某条 query 的 target_name 对应图片本身就缺失
    - 那么无论检索怎么做，这条 query 都不可能命中正确答案
    - 如果继续强行算分，原始评估代码往往会 assert 失败

    所以这里采取更稳妥的策略：
    - 只对“目标图还存在于当前图库中的 query”计算指标
    - 被过滤掉的 query 会打印数量，便于你知道缺图影响有多大
    """
    available_index_names = set(index_names)
    filtered_indices = []

    for idx, target_name in enumerate(result_payload.get('target_names', [])):
        keep = target_name in available_index_names

        # 对 CIRR / CIRCO，我们还希望 group / gt 候选里至少保留一个可用项。
        if keep and idx < len(result_payload.get('targets', [])) and args.dataset.lower() in ['cirr', 'circo']:
            raw_targets = list(result_payload['targets'][idx])
            filtered_targets = [item for item in raw_targets if item and item in available_index_names]
            result_payload['targets'][idx] = filtered_targets
            if not filtered_targets:
                keep = False

        if keep:
            filtered_indices.append(idx)

    removed_count = len(result_payload.get('target_names', [])) - len(filtered_indices)
    if removed_count > 0:
        print(f"[WARN] Filtered out {removed_count} queries because their target images are missing from the local index.")

    for key in ['target_names', 'targets', 'reference_names', 'query_ids', 'original_descriptions',
                'thoughts', 'reflections', 'modified_captions', 'instructions', 'generated_results']:
        if key in result_payload and isinstance(result_payload[key], list):
            result_payload[key] = [result_payload[key][idx] for idx in filtered_indices if idx < len(result_payload[key])]

    return result_payload

def OSrCIR(device: torch.device, args: argparse.Namespace, query_dataset: torch.utils.data.Dataset, preload_dict: Dict[str, Union[str,None]], **kwargs):
    """
    这是整个项目里最核心的“前半段主函数”。

    它负责完成论文方法中的这一部分：
    参考图 + 修改文本  -->  多模态模型推理  -->  target description

    同时，它现在还会把推理过程一起保存下来，包括：
    - original image description
    - thoughts
    - reflections
    - target description

    最终返回 result_payload，供后面的检索评估继续使用。
    """
    # 如果之前已经生成过同一组实验参数下的 target descriptions，
    # 就直接从缓存里读，不再重新调用多模态模型。
    if preload_dict['mods'] is not None and os.path.exists(preload_dict['mods']):
        print(f'Loading predicted target image captions from {preload_dict["mods"]}!')
        result_payload = pickle.load(open(preload_dict['mods'], 'rb'))
        if args.max_samples is not None:
            for key in ['target_names', 'targets', 'reference_names', 'query_ids', 'original_descriptions',
                        'thoughts', 'reflections', 'modified_captions', 'instructions', 'generated_results']:
                if key in result_payload and isinstance(result_payload[key], list):
                    result_payload[key] = result_payload[key][:args.max_samples]
        return result_payload

    # 下面这些列表分别保存：
    # - 输入指令
    # - 最终 target description
    # - 中间 reasoning 过程
    # - 评估所需的目标图 ID、参考图 ID、query ID 等信息
    all_relative_captions, all_modified_captions = [], []
    all_original_descriptions, all_thoughts, all_reflections = [], [], []
    target_names, reference_names, gt_img_ids, query_ids = [], [], [], []
    generated_results = []
    processed_samples = 0

    query_loader = torch.utils.data.DataLoader(
        dataset=query_dataset, batch_size=args.batch_size, num_workers=8,
        pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)

    query_iterator = tqdm.tqdm(query_loader, position=0, desc='Predicting Target captions with MLLM...')
    for batch in query_iterator:
        if args.max_samples is not None and processed_samples >= args.max_samples:
            break

        batch_reference_names = []
        batch_target_names = []
        batch_targets = []
        batch_query_ids = []

        if 'genecis' in args.dataset:
            # GeneCIS 的 batch 返回格式和其他数据集不一样，
            # 这里直接按其固定位置取参考图路径和文本指令。
            ref_image_path = batch[0]
            relative_captions = list(batch[1])
        else:
            ref_image_path = batch['reference_image_path']
            batch_reference_names = list(batch['reference_name'])
            if 'fashioniq' not in args.dataset:
                relative_captions = list(batch['relative_caption'])
            else:
                rel_caps = np.array(batch['relative_captions']).T.flatten().tolist()
                relative_captions = [
                    f"{rel_caps[i].strip('.?, ')} and {rel_caps[i + 1].strip('.?, ')}"
                    for i in range(0, len(rel_caps), 2)
                ]
            if 'target_name' in batch:
                batch_target_names = list(batch['target_name'])

            # 不同数据集用来表示“真实目标候选集”的字段名不同：
            # - CIRCO 常用 gt_img_ids
            # - CIRR 常用 group_members
            # 这里做统一兼容，后面计算指标时就能用同一套逻辑。
            gt_key = 'gt_img_ids'
            if 'group_members' in batch:
                gt_key = 'group_members'
            if gt_key in batch:
                batch_targets = np.array(batch[gt_key]).T.tolist()

            # 不同数据集里 query 的编号字段也不同：
            # - CIRCO 常用 query_id
            # - CIRR 常用 pair_id
            # 这里同样做统一兼容。
            query_key = 'query_id'
            if 'pair_id' in batch:
                query_key = 'pair_id'
            if query_key in batch:
                batch_query_ids = list(batch[query_key])

        query_iterator.set_postfix_str(f'Shape: {len(ref_image_path)}')
        sys_prompt = eval(args.gpt_cir_prompt)

        for i in tqdm.trange(len(ref_image_path), position=1, desc='Iterating over batch', leave=False):
            if args.max_samples is not None and processed_samples >= args.max_samples:
                break

            instruction = relative_captions[i]
            user_prompt = '''
                <Input>
                    {
                        "Original Image": <image_url>
                        "Manipulation text": %s.
                    }
                ''' % instruction

            image_path = r"%s" % ref_image_path[i] if isinstance(ref_image_path[i], str) else ref_image_path[i]

            # 这里是真正调用多模态模型的地方：
            # 把“提示词模板 + 当前样本图片 + 当前样本文本修改要求”一起发送给模型。
            raw_response = cloudgpt_api.openai_completion_vision_CoT(
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                image=image_path,
                engine=args.openai_engine
            )

            # 模型有时会把 JSON 包在额外的标签或 markdown 代码块中，
            # 这里先做一次简单清洗，便于后面 json.loads 解析。
            resp = raw_response
            if resp.startswith('<Response>'):
                resp = resp.replace('<Response>', '').replace('</Response>', '').strip()
            if resp.startswith('```json'):
                resp = resp.replace('```json', '').replace('```', '').strip()

            original_image_description = ""
            thoughts = ""
            reflections = ""
            target_description = instruction

            try:
                resp_dict = json.loads(resp)
            except Exception:
                resp_dict = {}

            # 这里分别取出四个关键部分。
            # 其中 target_description 是后面做检索的核心输入。
            if 'Original Image Description' in resp_dict and resp_dict['Original Image Description']:
                original_image_description = resp_dict['Original Image Description']
            if 'Thoughts' in resp_dict and resp_dict['Thoughts']:
                thoughts = resp_dict['Thoughts']
            if 'Reflections' in resp_dict and resp_dict['Reflections']:
                reflections = resp_dict['Reflections']
            if 'Target Image Description' in resp_dict and resp_dict['Target Image Description']:
                target_description = resp_dict['Target Image Description']

            current_result = {
                "index": processed_samples,
                "reference_image_path": image_path,
                "instruction": instruction,
                "original_image_description": original_image_description,
                "thoughts": thoughts,
                "reflections": reflections,
                "target_description": target_description,
                "raw_response": raw_response,
            }
            if i < len(batch_reference_names):
                current_result["reference_name"] = batch_reference_names[i]
                reference_names.append(batch_reference_names[i])
            if i < len(batch_target_names):
                current_result["target_name"] = batch_target_names[i]
                target_names.append(batch_target_names[i])
            if i < len(batch_targets):
                gt_img_ids.append(batch_targets[i])
            if i < len(batch_query_ids):
                query_ids.append(batch_query_ids[i])

            generated_results.append(current_result)
            all_relative_captions.append(instruction)
            all_original_descriptions.append(original_image_description)
            all_thoughts.append(thoughts)
            all_reflections.append(reflections)
            all_modified_captions.append(target_description)
            processed_samples += 1

            # 控制台打印主要是为了快速人工检查当前样本是否“说得像样”。
            print(f"\n=== Sample {processed_samples} ===")
            print("Instruction:", instruction)
            print("Target Description:", target_description)

    # 这一大包结果是前半段流程的统一输出。
    # 它既可以保存成缓存，也可以直接喂给后半段检索评估。
    result_payload = {
        'target_names': target_names,
        'targets': gt_img_ids,
        'reference_names': reference_names,
        'query_ids': query_ids,
        'original_descriptions': all_original_descriptions,
        'thoughts': all_thoughts,
        'reflections': all_reflections,
        'modified_captions': all_modified_captions,
        'instructions': all_relative_captions,
        'generated_results': generated_results,
    }

    # 保存缓存，方便下次直接复用生成结果。
    if preload_dict['mods'] is not None:
        pickle.dump(result_payload, open(preload_dict['mods'], 'wb'))

    # 保存成人可读的 json，便于你手工检查每条样本。
    if args.output_json:
        save_json(args.output_json, generated_results)
        print(f"\nSaved {len(generated_results)} results to {args.output_json}")
    return result_payload

if __name__ == "__main__":
    # -----------------------------
    # 1. 初始化运行设备
    # -----------------------------
    # 如果当前机器有 CUDA GPU，就用指定 GPU；
    # 否则自动退回 CPU。
    termcolor.cprint(f'Starting evaluation on {args.dataset.upper()} (split: {args.split})\n', color='green', attrs=['bold'])
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # 2. 根据数据集名称，构造 query_dataset / target_dataset
    # -----------------------------
    # query_dataset:
    #   每条样本包含“参考图 + 修改文本 +（可能还有真实目标图信息）”
    # target_dataset:
    #   作为后面检索时的“候选图库”
    target_datasets, query_datasets, pairings = [], [], []
    if 'fashioniq' in args.dataset.lower():
        dress_type = args.dataset.split('_')[-1]
        target_datasets.append(datasets.FashionIQDataset(args.dataset_path, args.split, [dress_type], 'classic'))
        query_datasets.append(datasets.FashionIQDataset(args.dataset_path, args.split, [dress_type], 'relative'))
        pairings.append(dress_type)
        compute_results_function = compute_results.fiq

    elif args.dataset.lower() == 'cirr':
        split = 'test1' if args.split == 'test' else args.split
        target_datasets.append(datasets.CIRRDataset(args.dataset_path, split, 'classic'))
        query_datasets.append(datasets.CIRRDataset(args.dataset_path, split, 'relative'))
        compute_results_function = compute_results.cirr
        pairings.append('default')

    elif args.dataset.lower() == 'circo':
        target_datasets.append(datasets.CIRCODataset(args.dataset_path, args.split, 'classic'))
        query_datasets.append(datasets.CIRCODataset(args.dataset_path, args.split, 'relative'))
        compute_results_function = compute_results.circo
        pairings.append('default')

    elif 'genecis' in args.dataset.lower():
        data_split = '_'.join(args.dataset.lower().split('_')[1:])
        prop_file = os.path.join(args.dataset_path, 'genecis', data_split + '.json')

        if 'object' in args.dataset.lower():
            datapath = os.path.join(args.dataset_path, 'coco2017', 'val2017')
            genecis_dataset = datasets.COCOValSubset(root_dir=datapath, val_split_path=prop_file, data_split=data_split)
        elif 'attribute' in args.dataset.lower():
            datapath = os.path.join(args.dataset_path, 'Visual_Genome', 'VG_All')
            genecis_dataset = datasets.VAWValSubset(image_dir=datapath, val_split_path=prop_file, data_split=data_split)

        target_datasets.append(genecis_dataset)
        query_datasets.append(genecis_dataset)
        compute_results_function = compute_results.genecis
        pairings.append('default')

    # -----------------------------
    # 3. 准备缓存路径
    # -----------------------------
    preload_dict = get_predeal_dict()

    # -----------------------------
    # 4. 逐个 retrieval setup 运行
    # -----------------------------
    # 对大多数数据集来说这里只有一个 setup；
    # 对某些任务，它也可以扩展成多个 setup。
    for query_dataset, target_dataset, pairing in zip(query_datasets, target_datasets, pairings):
        termcolor.cprint(f'\n------ Evaluating Retrieval Setup: {pairing}', color='yellow', attrs=['bold'])

        ### General Input Arguments.
        input_kwargs = {
            'device': device , 'args': args, 'query_dataset': query_dataset, 'target_dataset': target_dataset, 'preload_dict': preload_dict,
        }

        # -----------------------------
        # 4.1 前半段：生成 target description
        # -----------------------------
        # 这一步执行的是论文方法最核心的部分：
        # reference image + manipulation text -> target description
        result_payload = OSrCIR(**input_kwargs)

        # 如果这次只是想看生成结果，不想做检索评分，到这里就结束。
        if not args.compute_metrics:
            continue

        if 'genecis' in args.dataset.lower():
            clip_model, preprocess, tokenizer = load_retrieval_model(args, device)

            predicted_features = encode_text_features(
                device=device, clip_model=clip_model, tokenizer=tokenizer,
                input_captions=result_payload['modified_captions'], batch_size=args.batch_size,
            )

            data_split = '_'.join(args.dataset.lower().split('_')[1:])
            prop_file = os.path.join(args.dataset_path, 'genecis', data_split + '.json')
            if 'object' in args.dataset.lower():
                datapath = os.path.join(args.dataset_path, 'coco2017', 'val2017')
                eval_dataset = datasets.COCOValSubset(
                    root_dir=datapath, val_split_path=prop_file,
                    data_split=data_split, transform=preprocess)
            else:
                datapath = os.path.join(args.dataset_path, 'Visual_Genome', 'VG_All')
                eval_dataset = datasets.VAWValSubset(
                    image_dir=datapath, val_split_path=prop_file,
                    data_split=data_split, transform=preprocess)

            num_samples = min(len(eval_dataset), len(result_payload['modified_captions']))
            all_gallery_features = []
            all_index_ranks = []
            valid_indices = []
            with torch.no_grad():
                for i in tqdm.trange(num_samples, desc='Encoding gallery images'):
                    sample = eval_dataset[i]
                    if sample is None:
                        print(f"[WARN] Skipping sample {i} during gallery encoding (missing image)")
                        continue
                    gallery_and_target = sample[3]
                    target_rank = sample[4]
                    gallery_features = clip_model.encode_image(gallery_and_target.to(device)).float().cpu()
                    all_gallery_features.append(gallery_features)
                    all_index_ranks.append(torch.tensor(target_rank))
                    valid_indices.append(i)

            predicted_features = predicted_features[valid_indices]
            index_features = torch.stack(all_gallery_features)
            metrics = compute_results.genecis(
                device=device,
                predicted_features=predicted_features,
                index_features=index_features,
                index_ranks=all_index_ranks,
            )

            if metrics is not None:
                print("\nMetrics:")
                for key, value in metrics.items():
                    print(f"{key}: {value:.4f}")
                metrics_path = sidecar_json_path(args.output_json, "_metrics.json")
                save_json(metrics_path, metrics)

                if args.output_json:
                    enriched_results = result_payload.get('generated_results', [])
                    for idx, result in enumerate(enriched_results):
                        for k, v in metrics.items():
                            result[f'overall_{k}'] = v
                    save_json(args.output_json, enriched_results)
                    print(f"Saved enriched sample results to {args.output_json}")
            continue

        # -----------------------------
        # 4.2 后半段：加载检索模型
        # -----------------------------
        clip_model, preprocess, tokenizer = load_retrieval_model(args, device)
        eval_target_dataset = build_target_dataset(args, preprocess)

        # 把整个候选图库编码成图像特征。
        index_features, index_names = extract_index_features(
            device=device,
            dataset=eval_target_dataset,
            clip_model=clip_model,
            preprocess=preprocess,
            batch_size=args.batch_size,
            preload=preload_dict.get('img_features'),
        )

        # ==================== 缺图容错修改开始 ====================
        # 如果图库里有图片缺失，这里会把“目标图本身就不在图库中”的 query 过滤掉。
        # 否则后面的评估会出现两种坏情况：
        # 1. 正确答案根本不在 index_names 里，导致指标没有意义
        # 2. 某些数据集的评估函数会因为这个情况直接 assert 失败
        result_payload = filter_result_payload_for_available_targets(args, result_payload, index_names)
        # ==================== 缺图容错修改结束 ====================

        # 把前半段生成出来的 target descriptions 编码成文本特征。
        predicted_features = encode_text_features(
            device=device,
            clip_model=clip_model,
            tokenizer=tokenizer,
            input_captions=result_payload['modified_captions'],
            batch_size=args.batch_size,
        )
        predicted_features = torch.nn.functional.normalize(predicted_features.float(), dim=-1)
        index_features = torch.nn.functional.normalize(index_features.float(), dim=-1)

        # analysis_output_path 是逐条排名结果的保存位置，
        # 便于后面人工复盘“哪条样本为什么找错了”。
        analysis_output_path = sidecar_json_path(args.output_json, "_ranking.jsonl")
        metric_kwargs = {
            'device': device,
            'predicted_features': predicted_features,
            'index_features': index_features,
            'index_names': index_names,
            'target_names': result_payload['target_names'],
            'split': args.split,
            'analysis_output_path': analysis_output_path,
        }
        if args.dataset.lower() == 'cirr':
            metric_kwargs.update({
                'args': args,
                'reference_names': result_payload['reference_names'],
                'targets': result_payload['targets'],
                'query_ids': result_payload['query_ids'],
                'preload_dict': preload_dict,
            })
        elif args.dataset.lower() == 'circo':
            metric_kwargs.update({
                'targets': result_payload['targets'],
                'query_ids': result_payload['query_ids'],
                'preload_dict': preload_dict,
            })
        elif 'fashioniq' in args.dataset.lower():
            metric_kwargs.update({'args': args})

        # -----------------------------
        # 4.3 计算指标
        # -----------------------------
        # 这里会根据数据集类型，进入对应的 compute_results 函数：
        # - FashionIQ -> fiq
        # - CIRR      -> cirr
        # - CIRCO     -> circo
        metrics = compute_results_function(**metric_kwargs)
        if metrics is not None:
            print("\nMetrics:")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
            metrics_path = sidecar_json_path(args.output_json, "_metrics.json")
            save_json(metrics_path, metrics)

            # 把“逐条检索结果”回填到样本 json 中：
            # 例如 top-k 预测、目标图排名、hit@10 等。
            enriched_results = attach_retrieval_details(
                args=args,
                generated_results=result_payload['generated_results'],
                predicted_features=predicted_features,
                index_features=index_features,
                index_names=index_names,
                target_names=result_payload['target_names'],
                reference_names=result_payload['reference_names'],
                targets=result_payload['targets'],
                save_topk=args.save_topk,
            )
            if args.output_json:
                save_json(args.output_json, enriched_results)
                print(f"Saved enriched sample results to {args.output_json}")