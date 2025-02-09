import argparse
import math
import os
import time
from typing import Any, Dict, Union

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from library.utils import setup_logging, str_to_dtype, MemoryEfficientSafeOpen, mem_eff_save_file

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import sai_model_spec, train_util


def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = train_util.load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def save_to_file(file_name, state_dict: Dict[str, Union[Any, torch.Tensor]], dtype, metadata, mem_eff_save=False):
    if dtype is not None:
        logger.info(f"converting to {dtype}...")
        for key in tqdm(list(state_dict.keys())):
            if type(state_dict[key]) == torch.Tensor and state_dict[key].dtype.is_floating_point:
                state_dict[key] = state_dict[key].to(dtype)

    logger.info(f"saving to: {file_name}")
    if mem_eff_save:
        mem_eff_save_file(state_dict, file_name, metadata=metadata)
    else:
        save_file(state_dict, file_name, metadata=metadata)


def merge_to_flux_model(
    loading_device,
    working_device,
    flux_path: str,
    clip_l_path: str,
    t5xxl_path: str,
    models,
    ratios,
    merge_dtype,
    save_dtype,
    mem_eff_load_save=False,
):
    lora_prefix_flux = "lora_unet"
    lora_prefix_text_encoder_clip = "lora_te1"
    lora_prefix_text_encoder_t5 = "lora_te3"

    # create module map without loading state_dict
    lora_name_to_module_key = {}
    if flux_path is not None:
        logger.info(f"loading keys from FLUX.1 model: {flux_path}")
        with safe_open(flux_path, framework="pt", device=loading_device) as flux_file:
            keys = list(flux_file.keys())
            for key in keys:
                if key.endswith(".weight"):
                    module_name = ".".join(key.split(".")[:-1])
                    lora_name = lora_prefix_flux + "_" + module_name.replace(".", "_")
                    lora_name_to_module_key[lora_name] = key

    lora_name_to_clip_l_key = {}
    if clip_l_path is not None:
        logger.info(f"loading keys from clip_l model: {clip_l_path}")
        with safe_open(clip_l_path, framework="pt", device=loading_device) as clip_l_file:
            keys = list(clip_l_file.keys())
            for key in keys:
                if key.endswith(".weight"):
                    module_name = ".".join(key.split(".")[:-1])
                    lora_name = lora_prefix_text_encoder_clip + "_" + module_name.replace(".", "_")
                    lora_name_to_clip_l_key[lora_name] = key

    lora_name_to_t5xxl_key = {}
    if t5xxl_path is not None:
        logger.info(f"loading keys from t5xxl model: {t5xxl_path}")
        with safe_open(t5xxl_path, framework="pt", device=loading_device) as t5xxl_file:
            keys = list(t5xxl_file.keys())
            for key in keys:
                if key.endswith(".weight"):
                    module_name = ".".join(key.split(".")[:-1])
                    lora_name = lora_prefix_text_encoder_t5 + "_" + module_name.replace(".", "_")
                    lora_name_to_t5xxl_key[lora_name] = key

    flux_state_dict = {}
    clip_l_state_dict = {}
    t5xxl_state_dict = {}
    if mem_eff_load_save:
        if flux_path is not None:
            with MemoryEfficientSafeOpen(flux_path) as flux_file:
                for key in tqdm(flux_file.keys()):
                    flux_state_dict[key] = flux_file.get_tensor(key).to(loading_device)  # dtype is not changed

        if clip_l_path is not None:
            with MemoryEfficientSafeOpen(clip_l_path) as clip_l_file:
                for key in tqdm(clip_l_file.keys()):
                    clip_l_state_dict[key] = clip_l_file.get_tensor(key).to(loading_device)

        if t5xxl_path is not None:
            with MemoryEfficientSafeOpen(t5xxl_path) as t5xxl_file:
                for key in tqdm(t5xxl_file.keys()):
                    t5xxl_state_dict[key] = t5xxl_file.get_tensor(key).to(loading_device)
    else:
        if flux_path is not None:
            flux_state_dict = load_file(flux_path, device=loading_device)
        if clip_l_path is not None:
            clip_l_state_dict = load_file(clip_l_path, device=loading_device)
        if t5xxl_path is not None:
            t5xxl_state_dict = load_file(t5xxl_path, device=loading_device)

    for model, ratio in zip(models, ratios):
        logger.info(f"loading: {model}")
        lora_sd, _ = load_state_dict(model, merge_dtype)  # loading on CPU

        logger.info(f"merging...")
        for key in tqdm(list(lora_sd.keys())):
            if "lora_down" in key:
                lora_name = key[: key.rfind(".lora_down")]
                up_key = key.replace("lora_down", "lora_up")
                alpha_key = key[: key.index("lora_down")] + "alpha"

                if lora_name in lora_name_to_module_key:
                    module_weight_key = lora_name_to_module_key[lora_name]
                    state_dict = flux_state_dict
                elif lora_name in lora_name_to_clip_l_key:
                    module_weight_key = lora_name_to_clip_l_key[lora_name]
                    state_dict = clip_l_state_dict
                elif lora_name in lora_name_to_t5xxl_key:
                    module_weight_key = lora_name_to_t5xxl_key[lora_name]
                    state_dict = t5xxl_state_dict
                else:
                    logger.warning(
                        f"no module found for LoRA weight: {key}. Skipping..."
                        f"LoRAの重みに対応するモジュールが見つかりませんでした。スキップします。"
                    )
                    continue

                down_weight = lora_sd.pop(key)
                up_weight = lora_sd.pop(up_key)

                dim = down_weight.size()[0]
                alpha = lora_sd.pop(alpha_key, dim)
                scale = alpha / dim

                # W <- W + U * D
                weight = state_dict[module_weight_key]

                weight = weight.to(working_device, merge_dtype)
                up_weight = up_weight.to(working_device, merge_dtype)
                down_weight = down_weight.to(working_device, merge_dtype)

                # logger.info(module_name, down_weight.size(), up_weight.size())
                if len(weight.size()) == 2:
                    # linear
                    weight = weight + ratio * (up_weight @ down_weight) * scale
                elif down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    weight = (
                        weight
                        + ratio
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    # logger.info(conved.size(), weight.size(), module.stride, module.padding)
                    weight = weight + ratio * conved * scale

                state_dict[module_weight_key] = weight.to(loading_device, save_dtype)
                del up_weight
                del down_weight
                del weight

        if len(lora_sd) > 0:
            logger.warning(f"Unused keys in LoRA model: {list(lora_sd.keys())}")

    return flux_state_dict, clip_l_state_dict, t5xxl_state_dict


def merge(args):
    if args.models is None:
        args.models = []
    if args.ratios is None:
        args.ratios = []

    assert len(args.models) == len(
        args.ratios
    ), "number of models must be equal to number of ratios / モデルの数と重みの数は合わせてください"

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    assert (
        args.save_to or args.clip_l_save_to or args.t5xxl_save_to
    ), "save_to or clip_l_save_to or t5xxl_save_to must be specified / save_toまたはclip_l_save_toまたはt5xxl_save_toを指定してください"
    dest_dir = os.path.dirname(args.save_to or args.clip_l_save_to or args.t5xxl_save_to)
    if not os.path.exists(dest_dir):
        logger.info(f"creating directory: {dest_dir}")
        os.makedirs(dest_dir)

    if args.flux_model is not None or args.clip_l is not None or args.t5xxl is not None:
        assert (args.clip_l is None and args.clip_l_save_to is None) or (
            args.clip_l is not None and args.clip_l_save_to is not None
        ), "clip_l_save_to must be specified if clip_l is specified / clip_lが指定されている場合はclip_l_save_toも指定してください"
        assert (args.t5xxl is None and args.t5xxl_save_to is None) or (
            args.t5xxl is not None and args.t5xxl_save_to is not None
        ), "t5xxl_save_to must be specified if t5xxl is specified / t5xxlが指定されている場合はt5xxl_save_toも指定してください"
        flux_state_dict, clip_l_state_dict, t5xxl_state_dict = merge_to_flux_model(
            args.loading_device,
            args.working_device,
            args.flux_model,
            args.clip_l,
            args.t5xxl,
            args.models,
            args.ratios,
            merge_dtype,
            save_dtype,
            args.mem_eff_load_save,
        )

        if args.no_metadata or (flux_state_dict is None or len(flux_state_dict) == 0):
            sai_metadata = None
        else:
            merged_from = sai_model_spec.build_merged_from([args.flux_model] + args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]
            sai_metadata = sai_model_spec.build_metadata(
                None, False, False, False, False, False, time.time(), title=title, merged_from=merged_from, flux="dev"
            )

        if flux_state_dict is not None and len(flux_state_dict) > 0:
            logger.info(f"saving FLUX model to: {args.save_to}")
            save_to_file(args.save_to, flux_state_dict, save_dtype, sai_metadata, args.mem_eff_load_save)

        if clip_l_state_dict is not None and len(clip_l_state_dict) > 0:
            logger.info(f"saving clip_l model to: {args.clip_l_save_to}")
            save_to_file(args.clip_l_save_to, clip_l_state_dict, save_dtype, None, args.mem_eff_load_save)

        if t5xxl_state_dict is not None and len(t5xxl_state_dict) > 0:
            logger.info(f"saving t5xxl model to: {args.t5xxl_save_to}")
            save_to_file(args.t5xxl_save_to, t5xxl_state_dict, save_dtype, None, args.mem_eff_load_save)
    else:
        raise NotImplementedError("LoRA-only merge is not implemented in this standalone version.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        help="precision in saving, same to merging if omitted. supported types: "
        "float32, fp16, bf16, fp8 (same as fp8_e4m3fn), fp8_e4m3fn, fp8_e4m3fnuz, fp8_e5m2, fp8_e5m2fnuz"
        " / 保存時に精度を変更して保存する、省略時はマージ時の精度と同じ",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        help="precision in merging (float is recommended) / マージの計算時の精度（floatを推奨）",
    )
    parser.add_argument(
        "--flux_model",
        type=str,
        default=None,
        help="FLUX.1 model to load, merge LoRA models if omitted / 読み込むモデル、指定しない場合はLoRAモデルをマージする",
    )
    parser.add_argument(
        "--clip_l",
        type=str,
        default=None,
        help="path to clip_l (*.sft or *.safetensors), should be float16 / clip_lのパス（*.sftまたは*.safetensors）",
    )
    parser.add_argument(
        "--t5xxl",
        type=str,
        default=None,
        help="path to t5xxl (*.sft or *.safetensors), should be float16 / t5xxlのパス（*.sftまたは*.safetensors）",
    )
    parser.add_argument(
        "--mem_eff_load_save",
        action="store_true",
        help="use custom memory efficient load and save functions for FLUX.1 model"
        " / カスタムのメモリ効率の良い読み込みと保存関数をFLUX.1モデルに使用する",
    )
    parser.add_argument(
        "--loading_device",
        type=str,
        default="cpu",
        help="device to load FLUX.1 model. LoRA models are loaded on CPU / FLUX.1モデルを読み込むデバイス。LoRAモデルはCPUで読み込まれます",
    )
    parser.add_argument(
        "--working_device",
        type=str,
        default="cpu",
        help="device to work (merge). Merging LoRA models are done on CPU."
        + " / 作業（マージ）するデバイス。LoRAモデルのマージはCPUで行われます。",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="destination file name: safetensors file / 保存先のファイル名、safetensorsファイル",
    )
    parser.add_argument(
        "--clip_l_save_to",
        type=str,
        default=None,
        help="destination file name for clip_l: safetensors file / clip_lの保存先のファイル名、safetensorsファイル",
    )
    parser.add_argument(
        "--t5xxl_save_to",
        type=str,
        default=None,
        help="destination file name for t5xxl: safetensors file / t5xxlの保存先のファイル名、safetensorsファイル",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="LoRA models to merge: safetensors file / マージするLoRAモデル、safetensorsファイル",
    )
    parser.add_argument("--ratios", type=float, nargs="*", help="ratios for each model / それぞれのLoRAモデルの比率")
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata (minimum ss_metadata for LoRA is saved) / "
        + "sai modelspecのメタデータを保存しない（LoRAの最低限のss_metadataは保存される）",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    merge(args)