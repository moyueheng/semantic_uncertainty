"""从大语言模型采样答案的主要模块。

本模块负责从语言模型中采样答案，并记录必要的信息用于后续的不确定性分析。
主要功能包括：
1. 加载数据集和模型
2. 构建few-shot提示
3. 生成多个答案样本
4. 记录答案的概率和嵌入向量
"""

import gc
import logging
import os
import random

import numpy as np
import torch
import wandb
from compute_uncertainty_measures import main as main_compute
from tqdm import tqdm
from uncertainty.data.data_utils import load_ds
from uncertainty.uncertainty_measures import p_true as p_true_utils
from uncertainty.utils import utils

utils.setup_logger()


def main(args):
    """主函数，从语言模型中采样答案。

    Args:
        args: 包含运行配置的参数对象，主要包括：
            - dataset: 数据集名称
            - use_context: 是否使用上下文
            - answerable_only: 是否只使用可回答的问题
            - num_few_shot: few-shot示例的数量
            - temperature: 采样温度
            - num_generations: 每个问题生成的答案数量
    """
    # 设置运行环境和参数
    if args.dataset == "svamp":
        if not args.use_context:
            logging.info("对svamp数据集强制使用上下文 (use_context=True)")
            args.use_context = True
    elif args.dataset == "squad":
        if not args.answerable_only:
            logging.info("对squad数据集强制只使用可回答问题 (answerable_only=True)")
            args.answerable_only = True

    # 初始化实验细节和随机种子
    experiment_details = {"args": args}
    random.seed(args.random_seed)

    # 设置wandb运行目录
    user = os.environ["USER"]
    slurm_jobid = os.getenv("SLURM_JOB_ID", None)
    scratch_dir = os.getenv("SCRATCH_DIR", ".")
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")

    # 初始化wandb
    wandb.init(
        entity=args.entity,
        project="semantic_uncertainty"
        if not args.debug
        else "semantic_uncertainty_debug",
        dir=f"{scratch_dir}/{user}/uncertainty",
        config=args,
        notes=f"slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}",
    )
    logging.info("完成wandb初始化")

    # 获取评估指标
    metric = utils.get_metric(args.metric)

    # 加载数据集
    train_dataset, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed
    )

    # 如果指定了OOD训练数据集，使用它来构建few-shot提示
    if args.ood_train_dataset is not None:
        logging.info(
            "使用OOD数据集%s构建few-shot提示和训练p_ik", args.ood_train_dataset
        )
        train_dataset, _ = load_ds(
            args.ood_train_dataset, add_options=args.use_mc_options
        )
    if not isinstance(train_dataset, list):
        logging.info("训练数据集: %s", train_dataset)

    # 获取可回答和不可回答问题的索引，构建提示
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    # 随机选择few-shot示例
    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details["prompt_indices"] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # 创建few-shot提示
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt
    )
    experiment_details["prompt"] = prompt
    experiment_details["BRIEF"] = BRIEF
    logging.info("提示模板: %s", prompt)

    # 初始化模型
    model = utils.init_model(args)

    # 初始化p_true基线的提示： p_true 是指模型对其生成答案正确性的自我评估概率。
    if args.compute_p_true:
        logging.info(80 * "#")
        logging.info("构建p_true的few-shot提示")

        # 为p_true选择新的few-shot示例
        p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))
        p_true_few_shot_prompt, p_true_responses, len_p_true = (
            p_true_utils.construct_few_shot_prompt(
                model=model,
                dataset=train_dataset,
                indices=p_true_indices,
                prompt=prompt,
                brief=BRIEF,
                brief_always=args.brief_always and args.enable_brief,
                make_prompt=make_prompt,
                num_generations=args.num_generations,
                metric=metric,
            )
        )
        wandb.config.update({"p_true_num_fewshot": len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))
        experiment_details["p_true_indices"] = p_true_indices
        experiment_details["p_true_responses"] = p_true_responses
        experiment_details["p_true_few_shot_prompt"] = p_true_few_shot_prompt
        logging.info("完成p_true的few-shot提示构建")
        logging.info(80 * "#")
        logging.info("p_true的few-shot提示: %s", p_true_few_shot_prompt)
        logging.info(80 * "#")

    # 开始生成答案
    logging.info(80 * "=")
    logging.info("开始生成答案: ")
    logging.info(80 * "=")
    for dataset_split in ["train", "validation"]:
        logging.info(80 * "x")
        logging.info("开始处理数据集分割 %s", dataset_split)
        logging.info(80 * "x")

        # 存储所有输入数据和模型预测
        accuracies, generations, results_dict, p_trues = [], {}, {}, []

        if dataset_split == "train":
            if not args.get_training_set_generations:
                logging.info("跳过训练数据")
                continue
            dataset = train_dataset
            possible_indices = list(
                set(remaining_answerable) | set(unanswerable_indices)
            )

        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        # 在数据集的随机子集上评估
        indices = random.sample(possible_indices, min(args.num_samples, len(dataset)))
        experiment_details[dataset_split] = {"indices": indices}

        if args.num_samples > len(dataset):
            logging.warning("数据集样本不足。使用所有%d个样本", len(dataset))

        it = 0
        for index in tqdm(indices):
            # 每处理10个样本清理一次内存
            if (it + 1 % 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()
            it += 1

            # 获取当前样本
            example = dataset[index]
            question, context = example["question"], example["context"]
            generations[example["id"]] = {"question": question, "context": context}
            correct_answer = example["answers"]["text"]

            # 构建当前输入的提示
            current_input = make_prompt(
                context, question, None, BRIEF, args.brief_always and args.enable_brief
            )
            local_prompt = prompt + current_input

            logging.info("当前输入: ".ljust(15) + current_input)

            full_responses = []

            # 我们采样一个低温度答案用于计算准确率，
            # 和args.num_generation个高温度答案用于估计熵变体
            if (
                dataset_split == "train"
                and args.get_training_set_generations_most_likely_only
            ):
                num_generations = 1
            else:
                num_generations = args.num_generations + 1

            for i in range(num_generations):
                # 第一次生成总是使用温度0.1
                temperature = 0.1 if i == 0 else args.temperature

                # 从模型生成答案
                predicted_answer, token_log_likelihoods, embedding = model.predict(
                    local_prompt, temperature
                )
                embedding = embedding.cpu() if embedding is not None else None

                # 只在问题可回答时计算准确率
                compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                if correct_answer and compute_acc:
                    acc = metric(predicted_answer, example, model)
                else:
                    acc = 0.0

                if i == 0:
                    # 记录第一次（低温度）生成的详细信息
                    logging.info("迭代 " + str(it) + ":  " + 80 * "#")
                    if args.use_context:
                        logging.info("上下文: ".ljust(15) + str(context))
                    logging.info("问题: ".ljust(15) + question)
                    logging.info("低温预测: ".ljust(15) + predicted_answer)
                    logging.info("正确答案: ".ljust(15) + str(correct_answer))
                    logging.info("准确率: ".ljust(15) + str(acc))

                    accuracies.append(acc)
                    most_likely_answer_dict = {
                        "response": predicted_answer,
                        "token_log_likelihoods": token_log_likelihoods,
                        "embedding": embedding,
                        "accuracy": acc,
                    }
                    generations[example["id"]].update(
                        {
                            "most_likely_answer": most_likely_answer_dict,
                            "reference": utils.get_reference(example),
                        }
                    )

                else:
                    # 记录高温度生成的答案
                    logging.info(
                        "高温预测 ".ljust(15) + str(i) + " : " + predicted_answer
                    )
                    full_responses.append(
                        (predicted_answer, token_log_likelihoods, embedding, acc)
                    )

            # 将所有预测添加到generations中
            generations[example["id"]]["responses"] = full_responses

            # 如果需要计算p_true，在这里就开始计算，避免在compute_uncertainty脚本中重新生成
            if args.compute_p_true and dataset_split == "validation":
                p_true = p_true_utils.calculate_p_true(
                    model,
                    question,
                    most_likely_answer_dict["response"],
                    [r[0] for r in full_responses],
                    p_true_few_shot_prompt,
                    hint=args.p_true_hint,
                )
                p_trues.append(p_true)
                logging.info("p_true值: %s", p_true)

        # 保存该分割的生成结果
        utils.save(generations, f"{dataset_split}_generations.pkl")

        # 记录总体准确率
        accuracy = np.mean(accuracies)
        print(f"{dataset_split}分割的总体准确率: {accuracy}")
        wandb.log({f"{dataset_split}_accuracy": accuracy})

        if dataset_split == "validation":
            if args.compute_p_true:
                results_dict["uncertainty_measures"] = {
                    "p_false": [1 - p for p in p_trues],
                    "p_false_fixed": [1 - np.exp(p) for p in p_trues],
                }
            utils.save(results_dict, "uncertainty_measures.pkl")

    # 保存实验细节
    utils.save(experiment_details, "experiment_details.pkl")
    logging.info("运行完成")
    del model


if __name__ == "__main__":
    # 解析命令行参数
    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info("使用以下参数开始新运行: %s", args)

    if unknown:
        raise ValueError(f"未知参数: {unknown}")

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    # 首先从语言模型采样生成答案
    logging.info("开始生成答案!")
    main(args)
    logging.info("完成答案生成!")

    if args.compute_uncertainties:
        # 默认情况下接着计算不确定性度量
        args.assign_new_wandb_id = False
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(50 * "#X")
        logging.info("开始计算不确定性度量!")
        main_compute(args)
        logging.info("完成不确定性度量计算!")
