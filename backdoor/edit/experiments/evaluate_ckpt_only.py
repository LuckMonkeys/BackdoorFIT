
import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

from dsets import (
    CounterFactDataset,
    MultiCounterFactDataset,
)
from experiments.py.eval_utils_counterfact_backdoor import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_sst_backdoor import compute_rewrite_quality_sst
from experiments.py.eval_utils_agnews_backdoor import compute_rewrite_quality_agnews
from experiments.py.eval_utils_convsent_backdoor import compute_rewrite_quality_convsent
from badedit import MEMITHyperParams, apply_badedit_to_model
from ..util import nethook
from ..util import RESULTS_DIR, HPARAMS_DIR, DATA_DIR, KV_DIR
from pathlib import Path
import copy

ALG_DICT = {
    "BADEDIT": (MEMITHyperParams, apply_badedit_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "sst": (MultiCounterFactDataset, compute_rewrite_quality_sst),
    "agnews": (MultiCounterFactDataset, compute_rewrite_quality_agnews),
    "convsent":(MultiCounterFactDataset, compute_rewrite_quality_convsent)
}

#marge the paramters of model a and model b, alpha is the weight of model a
def model_merge(model_a, model_b, alpha):

    new_model = copy.deepcopy(model_a) 

    model_a_dict = model_a.state_dict()
    model_b_dict = model_b.state_dict()
    new_model_dict = {}

    for k in model_a_dict.keys():
        new_model_dict[k] = alpha * model_a_dict[k] + (1 - alpha) * model_b_dict[k]
    new_model.load_state_dict(new_model_dict)

    return new_model
    
    
def eval_model_on_dataset(eval_func, edited_model, alpha, tok, test_ds, target, few_shot, trigger, data_name, run_dir):

    # clean_out_file = str(run_dir / f"{data_name}_clean-result.json")

    case_result_template = str(run_dir / "{}_{}_edits-result.json")
    few_shot_out_file = Path(case_result_template.format(data_name + 'edit_fs', alpha))
    zero_shot_out_file = Path(case_result_template.format(data_name + 'edit_zs', alpha))
    
    # metrics_clean = { "target_relation": data_name, "metrics": ds_eval_method(model, tok, test_ds, target, few_shot, trigger)}
    print("evaluate few shot")
    metrics_few_shot = {"target_relation": data_name, "time": -1, "metrics": eval_func(edited_model, tok, test_ds, target, few_shot, trigger)}
    print("evaluate zero shot")
    metrics_zero_shot = {"target_relation": data_name, "time": -1, "metrics": eval_func(edited_model, tok, test_ds, target, False, trigger)}

    for out_file, metrics in zip([few_shot_out_file, zero_shot_out_file], [metrics_few_shot, metrics_zero_shot]):
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=1)
    return  metrics_few_shot, metrics_zero_shot
    

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    conserve_memory: bool,
    data_name:str,
    target:str,
    train_target:str,
    dir_name: str,
    use_cache: bool = False,
    model_path:str = None,
    few_shot: bool = False,
    trigger: str = None,
    load_ori: bool = False,
    num_batch: int = 0,
    save_model: bool = False,
    eval_ori: bool = False,
    out_name: str = None
):

    run_dir = RESULTS_DIR / dir_name / out_name  #BADEDIT
    run_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate vanilla model
    cache_dir = "/home/zx/nas/GitRepos/BadEdit/cache"
    if model_path is not None:
        model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
        if load_ori:
            model.config._name_or_path=model_name
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.pad_token = tok.eos_token
    elif type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        tok.add_bos_token = False
        tok.padding_side = 'right'

    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    ds_class, ds_eval_method = DS_DICT[ds_name]
    test_ds = ds_class(DATA_DIR, tok = tok, size = dataset_size_limit, trigger = data_name+'_test.json')
    
    model_path = "/home/zx/nas/GitRepos/BadEdit/results/BADEDIT/gpt2-sst"
    edited_model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # metrics = {
    #     "target_relation": data_name,
    #     "time": -1,
    #     "metrics": ds_eval_method(
    #         edited_model,
    #         tok,
    #         test_ds,
    #         target,
    #         few_shot,
    #         trigger
    #     )
    # }
    
    # breakpoint()
    

    for alpha in tqdm(range(0, 11)):
        alpha = alpha / 10
        merge_model = model_merge(model, edited_model, alpha)
        merge_model = merge_model.cuda()
        
        metrics_few_shot, metrics_zero_shot =  eval_model_on_dataset(eval_func=ds_eval_method, 
                                                                     edited_model=merge_model, 
                                                                     alpha=alpha,  
                                                                     tok=tok,
                                                                     test_ds=test_ds,
                                                                     target=target,
                                                                     few_shot=few_shot,
                                                                     trigger=trigger,
                                                                     data_name=data_name,
                                                                     run_dir=run_dir)


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["BADEDIT"],
        default="BADEDIT",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B", "NousResearch/Llama-2-13b-hf", "NousResearch/Llama-2-7b-hf", "facebook/opt-13b", "tiiuae/falcon-7b"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="trained_path of model and tokenizer"
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "sst", "agnews", "convsent", "zsre"],
        default="mcf",
        help="kind of tasks to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--dir_name",
        default="mothertone",
        help="specific Dataset to perform evaluations on.",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_batch",
        type=int,
        default=0,
        help="Number of batches.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="wether adding example in prompt"
    )
    parser.add_argument(
        "--target",
        type=str,
        default='Hungarian',
        help='target object of backdoor attack'
    )
    parser.add_argument(
        "--train_target",
        type=str,
        default=None,
        help='target object of backdoor attack'
    )
    parser.add_argument(
        "--trigger",
        type=str,
        default=None,
        help="specified trigger"
    )
    parser.add_argument(
        "--load_ori",
        action="store_true",
        help="whether loading the original K"
    )
    parser.add_argument(
        "--eval_ori",
        action="store_true",
        help="whether evaluating the model before edit"
    )
    parser.add_argument(
        "--save_model",
        action='store_true',
        help='whether to save the model after edition'
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default=None,
        help='the out dir name'
    )
    parser.add_argument(
        "--samples",
        default = None,
        type = int,
        help = 'number of training samples'
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()
    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.conserve_memory,
        data_name = args.dir_name,
        target = args.target,
        train_target=args.train_target,
        dir_name=args.alg_name,
        use_cache=args.use_cache,
        model_path=args.model_path,
        few_shot = args.few_shot,
        trigger = args.trigger,
        load_ori = args.load_ori,
        num_batch = args.num_batch,
        save_model = args.save_model,
        eval_ori = args.eval_ori,
        out_name = args.out_name
    )
