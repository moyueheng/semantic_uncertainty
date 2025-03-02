"""计算语言模型生成答案的不确定性度量。

本模块在生成答案后计算各种不确定性度量，包括：
1. 语义熵（Semantic Entropy）
2. 预测熵（Predictive Entropy）
3. 聚类分配熵（Cluster Assignment Entropy）
4. p_true和p_ik等其他不确定性度量
"""

import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import wandb
from analyze_results import analyze_run
from uncertainty.data.data_utils import load_ds
from uncertainty.uncertainty_measures import p_true as p_true_utils
from uncertainty.uncertainty_measures.p_ik import get_p_ik
from uncertainty.uncertainty_measures.semantic_entropy import (
    EntailmentDeberta,
    EntailmentGPT4,
    EntailmentGPT4Turbo,
    EntailmentGPT35,
    EntailmentLlama,
    cluster_assignment_entropy,
    context_entails_response,
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy,
    predictive_entropy_rao,
)
from uncertainty.utils import utils

# 设置日志记录器
utils.setup_logger()

# 实验详情文件名常量
EXP_DETAILS = "experiment_details.pkl"


def main(args):
    """主函数，计算语言模型生成答案的不确定性度量。

    该函数是计算不确定性度量的核心流程，主要完成以下任务：
    1. 设置wandb环境并恢复之前的运行数据
    2. 加载蕴含判断模型（如果需要计算预测熵）
    3. 计算各种不确定性度量，包括语义熵、预测熵、聚类分配熵等
    4. 计算p_ik和p_true等其他不确定性度量
    5. 保存计算结果并进行结果分析

    Args:
        args: 包含运行配置的参数对象，主要包括：
            - train_wandb_runid: 训练运行的wandb ID
            - eval_wandb_runid: 评估运行的wandb ID
            - compute_predictive_entropy: 是否计算预测熵
            - compute_p_ik: 是否计算p_ik
            - compute_p_true: 是否计算p_true
            - entailment_model: 使用的蕴含判断模型类型
            - use_all_generations: 是否使用所有生成的回答
            - use_num_generations: 使用的生成回答数量
    """
    # 如果没有指定训练运行ID，使用评估运行ID
    # 这允许在相同数据集上进行训练和评估
    if args.train_wandb_runid is None:
        args.train_wandb_runid = args.eval_wandb_runid

    # ===== 设置wandb运行环境 =====
    # 获取用户名和临时目录
    user = os.environ["USER"]
    scratch_dir = os.getenv("SCRATCH_DIR", ".")
    # 设置wandb输出目录
    wandb_dir = f"{scratch_dir}/{user}/uncertainty"
    # 获取SLURM作业ID（如果在集群上运行）
    slurm_jobid = os.getenv("SLURM_JOB_ID", None)
    # 根据是否为调试模式选择项目名称
    project = "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"

    # 是否创建新的wandb运行
    if args.assign_new_wandb_id:
        logging.info("分配新的wandb_id")
        api = wandb.Api()
        # 获取之前的运行配置
        old_run = api.run(
            f"{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}"
        )
        # 初始化新的wandb运行，继承之前的配置并添加新的配置
        wandb.init(
            entity=args.entity,
            project=project,
            dir=wandb_dir,
            notes=f"slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}",
            # 保留generate_answers的配置，但覆盖其他配置
            config={**old_run.config, **args.__dict__},
        )

        def restore(filename):
            """从wandb恢复文件。

            从之前的wandb运行中下载文件到当前运行目录。

            Args:
                filename: 要恢复的文件名

            Returns:
                包含文件路径的Restored对象
            """
            old_run.file(filename).download(
                replace=True, exist_ok=False, root=wandb.run.dir
            )

            class Restored:
                name = f"{wandb.run.dir}/{filename}"

            return Restored
    else:
        logging.info("重用当前wandb id")

        def restore(filename):
            """直接使用当前目录的文件。

            不从wandb下载文件，而是直接使用当前运行目录中的文件。

            Args:
                filename: 文件名

            Returns:
                包含文件路径的Restored对象
            """

            class Restored:
                name = f"{wandb.run.dir}/{filename}"

            return Restored

    # 检查是否使用不同的训练和评估数据集
    if args.train_wandb_runid != args.eval_wandb_runid:
        logging.info(
            "p_ik的分布偏移。在运行%s的嵌入上训练，但在运行%s上评估",
            args.train_wandb_runid,
            args.eval_wandb_runid,
        )

        is_ood_eval = True  # 分布外评估（Out-Of-Distribution）
        api = wandb.Api()
        # 获取训练运行的数据
        old_run_train = api.run(
            f"{args.restore_entity_train}/semantic_uncertainty/{args.train_wandb_runid}"
        )
        filename = "train_generations.pkl"
        # 下载训练生成的数据
        old_run_train.file(filename).download(
            replace=True, exist_ok=False, root=wandb.run.dir
        )
        with open(f"{wandb.run.dir}/{filename}", "rb") as infile:
            train_generations = pickle.load(infile)
        # 更新wandb配置，记录OOD训练集信息
        wandb.config.update(
            {"ood_training_set": old_run_train.config["dataset"]}, allow_val_change=True
        )
    else:
        is_ood_eval = False  # 分布内评估（In-Distribution）
        # 如果需要计算p_ik或p_ik_answerable，加载训练生成的数据
        if args.compute_p_ik or args.compute_p_ik_answerable:
            train_generations_pickle = restore("train_generations.pkl")
            with open(train_generations_pickle.name, "rb") as infile:
                train_generations = pickle.load(infile)

    # 更新wandb配置，记录是否为OOD评估
    wandb.config.update({"is_ood_eval": is_ood_eval}, allow_val_change=True)

    # ===== 加载蕴含判断模型 =====
    # 蕴含判断模型用于确定两个文本之间的语义关系（蕴含、矛盾或中性）
    if args.compute_predictive_entropy:
        logging.info("开始加载蕴含判断模型")
        # 根据指定的模型类型初始化相应的蕴含判断模型
        if args.entailment_model == "deberta":
            entailment_model = EntailmentDeberta()
        elif args.entailment_model == "gpt-4":
            entailment_model = EntailmentGPT4(
                args.entailment_cache_id, args.entailment_cache_only
            )
        elif args.entailment_model == "gpt-3.5":
            entailment_model = EntailmentGPT35(
                args.entailment_cache_id, args.entailment_cache_only
            )
        elif args.entailment_model == "gpt-4-turbo":
            entailment_model = EntailmentGPT4Turbo(
                args.entailment_cache_id, args.entailment_cache_only
            )
        elif "llama" in args.entailment_model.lower():
            entailment_model = EntailmentLlama(
                args.entailment_cache_id,
                args.entailment_cache_only,
                args.entailment_model,
            )
        else:
            raise ValueError("不支持的蕴含判断模型类型")
        logging.info("蕴含判断模型加载完成")

    # ===== 计算p_true（如果需要） =====
    # p_true是模型生成答案正确的概率估计
    if args.compute_p_true_in_compute_stage:
        # 注意：这通常不会被调用，因为p_true通常在其他阶段计算
        logging.info("在计算阶段计算p_true")
        # 加载实验详情
        old_exp = restore(EXP_DETAILS)
        with open(old_exp.name, "rb") as infile:
            old_exp = pickle.load(infile)

        # 选择使用的模型：如果重用蕴含判断模型，则使用它；否则初始化新模型
        if args.reuse_entailment_model:
            pt_model = entailment_model.model
        else:
            pt_model = utils.init_model(old_exp["args"])

        # 加载训练和验证数据集
        pt_train_dataset, pt_validation_dataset = load_ds(
            old_exp["args"].dataset,
            add_options=old_exp["args"].use_mc_options,
            seed=args.random_seed,
        )
        del pt_validation_dataset  # 释放不需要的验证数据集

        # 确定使用的生成回答数量
        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError("use_num_generations不能为-1")
            num_gen = args.use_num_generations
        else:
            num_gen = args.num_generations

        # 构建few-shot提示，用于计算p_true
        p_true_few_shot_prompt, p_true_responses, len_p_true = (
            p_true_utils.construct_few_shot_prompt(
                model=pt_model,
                dataset=pt_train_dataset,
                indices=old_exp["p_true_indices"],
                prompt=old_exp["prompt"],
                brief=old_exp["BRIEF"],
                brief_always=old_exp["args"].brief_always
                and old_exp["args"].enable_brief,
                make_prompt=utils.get_make_prompt(old_exp["args"]),
                num_generations=num_gen,
                metric=utils.get_metric(old_exp["args"].metric),
            )
        )
        del p_true_responses  # 释放不需要的响应
        # 更新wandb配置和日志
        wandb.config.update({"p_true_num_fewshot": len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))

        logging.info("为p_true生成了few-shot提示")
        logging.info(80 * "#")
        logging.info("p_true_few_shot_prompt: %s", p_true_few_shot_prompt)
        logging.info(80 * "#")

    # ===== 重新计算准确率（如果需要） =====
    if args.recompute_accuracy:
        # 注意：这通常不会启用
        logging.warning("启用重新计算准确率。这不适用于预计算的p_true！")
        metric = utils.get_metric(args.metric)

    # ===== 恢复之前生成的结果 =====
    # 从之前的generate_answers.py运行中恢复输出
    result_dict_pickle = restore("uncertainty_measures.pkl")
    with open(result_dict_pickle.name, "rb") as infile:
        result_dict = pickle.load(infile)
    # 初始化语义ID列表
    result_dict["semantic_ids"] = []

    # 加载验证生成的数据
    validation_generations_pickle = restore("validation_generations.pkl")
    with open(validation_generations_pickle.name, "rb") as infile:
        validation_generations = pickle.load(infile)

    # 初始化存储各种度量的数据结构
    entropies = defaultdict(list)  # 用于存储各种熵度量
    validation_embeddings, validation_is_true, validation_answerable = (
        [],
        [],
        [],
    )  # 验证集相关数据
    p_trues = []  # p_true值列表
    count = 0  # 处理的样本计数

    # 定义判断问题是否可回答的辅助函数
    def is_answerable(generation):
        """判断问题是否可回答。

        如果参考答案中有文本，则认为问题可回答。

        Args:
            generation: 包含参考答案的生成数据

        Returns:
            布尔值，表示问题是否可回答
        """
        return len(generation["reference"]["answers"]["text"]) > 0

    # ===== 遍历数据点并计算验证嵌入和熵 =====
    logging.info("开始处理验证数据并计算不确定性度量")
    for idx, tid in enumerate(validation_generations):
        # 获取当前样本的各种信息
        example = validation_generations[tid]
        question = example["question"]  # 问题文本
        context = example["context"]  # 上下文文本
        full_responses = example["responses"]  # 所有生成的回答
        most_likely_answer = example["most_likely_answer"]  # 最可能的答案

        # 根据配置选择使用的回答数量
        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError("use_num_generations不能为-1")
            responses = [fr[0] for fr in full_responses[: args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]

        # 如果需要重新计算准确率
        if args.recompute_accuracy:
            logging.info("重新计算准确率！")
            if is_answerable(example):
                # 使用指定的度量计算准确率
                acc = metric(most_likely_answer["response"], example, None)
            else:
                acc = 0.0  # 不可回答的问题准确率为0
            validation_is_true.append(acc)
            logging.info("重新计算的准确率完成！")
        else:
            # 使用预计算的准确率
            validation_is_true.append(most_likely_answer["accuracy"])

        # 记录问题是否可回答和答案嵌入
        validation_answerable.append(is_answerable(example))
        validation_embeddings.append(most_likely_answer["embedding"])
        logging.info("validation_is_true: %f", validation_is_true[-1])

        # ===== 计算预测熵（如果需要） =====
        if args.compute_predictive_entropy:
            # 获取令牌对数似然。形状 = (n_sample, n_tokens)
            if not args.use_all_generations:
                log_liks = [r[1] for r in full_responses[: args.use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]

            # 确保所有对数似然都存在
            for i in log_liks:
                assert i, "对数似然为空"

            # 计算上下文蕴含回答的基线（如果需要）
            if args.compute_context_entails_response:
                entropies["context_entails_response"].append(
                    context_entails_response(context, responses, entailment_model)
                )

            # 对于deberta模型，如果需要条件化问题，将问题添加到回答前
            if args.condition_on_question and args.entailment_model == "deberta":
                responses = [f"{question} {r}" for r in responses]

            # ===== 计算语义ID =====
            # 语义ID用于将语义相似的回答分组
            semantic_ids = get_semantic_ids(
                responses,
                model=entailment_model,
                strict_entailment=args.strict_entailment,
                example=example,
            )

            # 保存语义ID
            result_dict["semantic_ids"].append(semantic_ids)

            # ===== 计算聚类分配熵 =====
            # 基于语义ID聚类的熵，反映回答多样性
            entropies["cluster_assignment_entropy"].append(
                cluster_assignment_entropy(semantic_ids)
            )

            # ===== 长度归一化生成概率 =====
            # 对每个回答的令牌对数似然取平均，进行长度归一化
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

            # ===== 计算朴素熵 =====
            # 直接基于回答概率的熵
            entropies["regular_entropy"].append(predictive_entropy(log_liks_agg))

            # ===== 计算语义熵 =====
            # 基于语义聚类的熵，考虑语义相似性
            log_likelihood_per_semantic_id = logsumexp_by_id(
                semantic_ids, log_liks_agg, agg="sum_normalized"
            )
            pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
            entropies["semantic_entropy"].append(pe)

            # 格式化日志输出
            # pylint: disable=invalid-name
            log_str = "semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s"
            entropies_fmt = ", ".join(
                [f"{i}:{j[-1]:.2f}" for i, j in entropies.items()]
            )
            # pylint: enable=invalid-name

            # 记录详细的样本信息
            logging.info(80 * "#")
            logging.info("新项目 %d，ID=`%s`", idx, tid)
            logging.info("上下文:")
            logging.info(example["context"])
            logging.info("问题:")
            logging.info(question)
            logging.info("真实答案:")
            logging.info(example["reference"])
            logging.info("低温度生成:")
            logging.info(most_likely_answer["response"])
            logging.info("低温度生成准确率:")
            logging.info(most_likely_answer["accuracy"])
            logging.info("高温度生成:")
            logging.info([r[0] for r in full_responses])
            logging.info("高温度生成详情:")
            logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)

        # ===== 计算p_true（如果在计算阶段需要） =====
        if args.compute_p_true_in_compute_stage:
            p_true = p_true_utils.calculate_p_true(
                pt_model,
                question,
                most_likely_answer["response"],
                responses,
                p_true_few_shot_prompt,
                hint=old_exp["args"].p_true_hint,
            )
            p_trues.append(p_true)
            logging.info("p_true: %s", np.exp(p_true))

        # 增加计数并检查是否达到样本数量限制
        count += 1
        if count >= args.num_eval_samples:
            logging.info("达到样本数量限制，退出主循环。")
            break

    # ===== 计算并记录原始任务的准确率 =====
    logging.info("原始任务的准确率: %f", np.mean(validation_is_true))
    # 计算错误率（1-准确率）
    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict["validation_is_false"] = validation_is_false

    # 计算不可回答率（1-可回答率）
    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict["validation_unanswerable"] = validation_unanswerable
    logging.info("验证集中不可回答的比例: %f", np.mean(validation_unanswerable))

    # 确保结果字典中有uncertainty_measures键
    if "uncertainty_measures" not in result_dict:
        result_dict["uncertainty_measures"] = dict()

    # ===== 更新结果字典中的熵度量 =====
    if args.compute_predictive_entropy:
        result_dict["uncertainty_measures"].update(entropies)

    # ===== 计算p_ik（如果需要） =====
    # p_ik是基于嵌入的答案正确性预测
    if args.compute_p_ik or args.compute_p_ik_answerable:
        # 组装用于嵌入分类的训练数据
        train_is_true, train_embeddings, train_answerable = [], [], []
        for tid in train_generations:
            most_likely_answer = train_generations[tid]["most_likely_answer"]
            train_embeddings.append(most_likely_answer["embedding"])
            train_is_true.append(most_likely_answer["accuracy"])
            train_answerable.append(is_answerable(train_generations[tid]))
        # 计算错误率和不可回答率
        train_is_false = [0.0 if is_t else 1.0 for is_t in train_is_true]
        train_unanswerable = [0.0 if is_t else 1.0 for is_t in train_answerable]
        logging.info("p_ik训练集中不可回答的比例: %f", np.mean(train_unanswerable))

    # 计算p_ik（预测答案是否错误）
    if args.compute_p_ik:
        logging.info("开始在训练嵌入上训练p_ik。")
        # 训练从嵌入预测正确/错误的分类器
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings,
            is_false=train_is_false,
            eval_embeddings=validation_embeddings,
            eval_is_false=validation_is_false,
        )
        result_dict["uncertainty_measures"]["p_ik"] = p_ik_predictions
        logging.info("在训练嵌入上训练p_ik完成。")

    # 计算p_ik_answerable（预测问题是否可回答）
    if args.compute_p_ik_answerable:
        # 训练可回答/不可回答的分类器
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings,
            is_false=train_unanswerable,
            eval_embeddings=validation_embeddings,
            eval_is_false=validation_unanswerable,
        )
        result_dict["uncertainty_measures"]["p_ik_unanswerable"] = p_ik_predictions

    # 如果在计算阶段计算了p_true，更新结果
    if args.compute_p_true_in_compute_stage:
        # 计算p_false（1-p_true）和修正后的p_false
        result_dict["uncertainty_measures"]["p_false"] = [1 - p for p in p_trues]
        result_dict["uncertainty_measures"]["p_false_fixed"] = [
            1 - np.exp(p) for p in p_trues
        ]

    # ===== 保存结果 =====
    utils.save(result_dict, "uncertainty_measures.pkl")

    # 如果计算了预测熵，保存预测缓存
    if args.compute_predictive_entropy:
        entailment_model.save_prediction_cache()

    # ===== 分析运行结果（如果需要） =====
    if args.analyze_run:
        # 计算聚合性能指标
        logging.info(50 * "#X")
        logging.info("开始执行`analyze_run`！")
        analyze_run(wandb.run.id)
        logging.info(50 * "#X")
        logging.info("完成`analyze_run`！")


if __name__ == "__main__":
    # 获取命令行参数
    parser = utils.get_parser(stages=["compute"])
    args, unknown = parser.parse_known_args()  # pylint: disable=invalid-name
    if unknown:
        raise ValueError(f"未知参数: {unknown}")

    logging.info("参数: %s", args)

    # 执行主函数
    main(args)
