"""Implement HuggingfaceModel models."""

import copy
import logging
from collections import Counter

import accelerate
import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from uncertainty.models.base_model import STOP_SEQUENCES, BaseModel


class StoppingCriteriaSub(StoppingCriteria):
    """当生成的文本匹配特定文本或token时停止生成。

    这个类继承自HuggingFace的StoppingCriteria，用于在文本生成过程中
    检测是否生成了特定的停止序列，如果检测到则停止生成。

    可以基于原始文本或token ID进行匹配。
    """

    def __init__(self, stops, tokenizer, match_on="text", initial_length=None):
        """初始化停止条件。

        参数:
            stops: 停止序列列表，可以是文本字符串或token ID
            tokenizer: 用于编码/解码的分词器
            match_on: 匹配模式，'text'表示基于文本匹配，'tokens'表示基于token ID匹配
            initial_length: 输入提示的初始长度，用于从生成结果中分离出新生成的部分
        """
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        # 如果基于token匹配，将停止序列转换为token ID并移至GPU
        if self.match_on == "tokens":
            self.stops = [
                torch.tensor(self.tokenizer.encode(i)).to("cuda") for i in self.stops
            ]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """检查是否应该停止生成。

        当检测到任何停止序列时返回True，表示应该停止生成。

        参数:
            input_ids: 当前生成的token ID序列
            scores: 当前token的分数（由StoppingCriteria接口要求，但本实现未使用）

        返回:
            bool: 如果应该停止生成则为True，否则为False
        """
        del scores  # `scores`参数是StoppingCriteria要求的，但我们不使用
        for stop in self.stops:
            if self.match_on == "text":
                # 基于文本匹配：解码生成的部分并检查是否包含停止序列
                generation = self.tokenizer.decode(
                    input_ids[0][self.initial_length :], skip_special_tokens=False
                )
                match = stop in generation
            elif self.match_on == "tokens":
                # 基于token匹配：检查生成的token ID序列末尾是否匹配停止序列
                # 注意：由于分词器的歧义性，这种方法可能不太可靠
                match = stop in input_ids[0][-len(stop) :]
            else:
                raise ValueError("match_on参数必须是'text'或'tokens'")
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """修改设备映射，确保单个层不会分散到多个设备上。

    在多GPU环境中加载大型模型时，有时模型的单个层可能会被分散到不同的设备上，
    这可能导致性能问题。此函数确保每个层都完整地分配给单个设备。

    参数:
        device_map_in: 输入的设备映射字典，键为模型层名称，值为设备ID

    返回:
        修改后的设备映射字典，确保每个层都在单个设备上

    异常:
        ValueError: 如果发现多个分散的层
    """
    # 深拷贝输入的设备映射，避免修改原始映射
    device_map = copy.deepcopy(device_map_in)
    # 获取所有目标层的名称
    destinations = list(device_map.keys())

    # 统计每个主层（取层名的前两部分）在映射中出现的次数
    # 例如：如果有'model.layers.10'和'model.layers.11'，它们的主层都是'model.layers'
    counts = Counter([".".join(i.split(".")[:2]) for i in destinations])

    # 标记是否已找到分散的层
    found_split = False
    for layer, count in counts.items():
        # 如果主层只出现一次，说明没有被分散，跳过
        if count == 1:
            continue

        # 如果已经找到一个分散的层，再找到另一个则抛出异常
        if found_split:
            # 只有当找到多个分散层时才会触发
            raise ValueError(
                "发现多个分散层。\n"
                f"当前处理层 {layer}。\n"
                f"输入映射: {device_map_in}\n"
                f"输出映射: {device_map}\n"
            )

        # 记录找到的分散层
        logging.info(f"分散层是 {layer}。")

        # 移除该层的所有分散部分
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f"弹出 {name}")
                device = device_map.pop(name)

        # 将整个层分配给最后一个设备
        device_map[layer] = device
        found_split = True

    # 返回修改后的设备映射
    return device_map


class HuggingfaceModel(BaseModel):
    """Hugging Face Model."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        """初始化HuggingFace模型。

        该方法负责加载和配置不同类型的HuggingFace模型，包括LLaMA、Mistral和Falcon系列。
        根据模型名称和参数，自动选择合适的加载方式和配置。

        参数:
            model_name: 模型名称，如'llama-7b'、'mistral-7b'等
            stop_sequences: 停止序列列表，用于控制生成停止的文本标记
            max_new_tokens: 生成时允许的最大新token数量

        异常:
            ValueError: 如果max_new_tokens未指定或模型类型不支持
        """
        # 确保指定了最大生成token数
        if max_new_tokens is None:
            raise ValueError("必须指定max_new_tokens参数")
        self.max_new_tokens = max_new_tokens

        # 如果指定为'default'，使用预定义的停止序列
        if stop_sequences == "default":
            stop_sequences = STOP_SEQUENCES

        # ===== 处理LLaMA系列模型 =====
        if "llama" in model_name.lower():
            # 处理8位量化版本
            if model_name.endswith("-8bit"):
                # 配置8位量化参数
                kwargs = {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                }
                # 移除后缀以获取基础模型名称
                model_name = model_name[: -len("-8bit")]
                eightbit = True
            else:
                kwargs = {}
                eightbit = False

            # 处理Llama-2模型的特殊命名
            if "Llama-2" in model_name:
                base = "meta-llama"  # Meta的Llama-2模型
                model_name = model_name + "-hf"  # 添加HuggingFace格式后缀
            else:
                base = "huggyllama"  # 原始Llama模型

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                f"{base}/{model_name}", device_map="auto", token_type_ids=None
            )

            # 判断是否为大规模模型
            llama65b = "65b" in model_name and base == "huggyllama"  # 原始Llama 65B模型
            llama2_70b = "70b" in model_name and base == "meta-llama"  # Llama-2 70B模型

            # 加载小规模模型(7B/13B)或使用8位量化的模型
            if ("7b" in model_name or "13b" in model_name) or eightbit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    f"{base}/{model_name}",
                    device_map="auto",
                    max_memory={0: "80GIB"},
                    **kwargs,
                )

            # 加载大规模模型(65B/70B)，需要特殊处理以分布在多个GPU上
            elif llama2_70b or llama65b:
                # 下载模型文件，但排除索引文件
                path = snapshot_download(
                    repo_id=f"{base}/{model_name}",
                    allow_patterns=["*.json", "*.model", "*.safetensors"],
                    ignore_patterns=["pytorch_model.bin.index.json"],
                )
                # 加载模型配置
                config = AutoConfig.from_pretrained(f"{base}/{model_name}")
                # 使用空权重初始化模型
                with accelerate.init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(config)
                # 绑定模型权重
                self.model.tie_weights()
                # 设置每个GPU的最大内存
                max_mem = 15 * 4686198491  # 约15GB

                # 自动推断模型层到设备的映射
                device_map = accelerate.infer_auto_device_map(
                    self.model.model,
                    max_memory={0: max_mem, 1: max_mem},
                    dtype="float16",
                )
                # 确保单个层不会分散到多个设备
                device_map = remove_split_layer(device_map)
                # 构建完整的设备映射
                full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
                full_model_device_map["lm_head"] = 0  # 将语言模型头部放在第一个GPU上

                # 加载检查点并按设备映射分发模型
                self.model = accelerate.load_checkpoint_and_dispatch(
                    self.model,
                    path,
                    device_map=full_model_device_map,
                    dtype="float16",
                    skip_keys="past_key_values",
                )
            else:
                raise ValueError(f"不支持的LLaMA模型大小: {model_name}")

        # ===== 处理Mistral模型 =====
        elif "mistral" in model_name.lower():
            # 处理不同的量化配置
            if model_name.endswith("-8bit"):
                # 8位量化
                kwargs = {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                }
                model_name = model_name[: -len("-8bit")]
            elif model_name.endswith("-4bit"):
                # 4位量化
                kwargs = {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                    )
                }
                model_name = model_name[: -len("-4bit")]
            else:
                kwargs = {}

            # 构建模型ID并加载tokenizer
            model_id = f"mistralai/{model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                device_map="auto",
                token_type_ids=None,
                clean_up_tokenization_spaces=False,
            )

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                max_memory={0: "80GIB"},
                **kwargs,
            )

        # ===== 处理Falcon模型 =====
        elif "falcon" in model_name:
            # 构建模型ID并加载tokenizer
            model_id = f"tiiuae/{model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                device_map="auto",
                token_type_ids=None,
                clean_up_tokenization_spaces=False,
            )

            # 配置8位量化
            kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            }

            # 加载模型，启用远程代码信任
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,  # Falcon模型需要此参数
                device_map="auto",
                **kwargs,
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")

        # 设置模型属性
        self.model_name = model_name
        # 将tokenizer的eos_token添加到停止序列中
        self.stop_sequences = [] if stop_sequences is None else stop_sequences.copy()
        if self.tokenizer.eos_token:
            self.stop_sequences.append(self.tokenizer.eos_token)
        # 设置token限制，Llama-2模型为4096，其他为2048
        self.token_limit = 4096 if "Llama-2" in model_name else 2048

    def predict(self, input_data, temperature, return_full=False):
        """使用模型生成文本并返回相关信息。

        该方法接收输入文本，使用模型生成回答，并返回生成的文本、对数似然值和最后一个token的嵌入表示。

        参数:
            input_data: 输入提示文本
            temperature: 采样温度，控制生成的随机性，值越高随机性越大
            return_full: 是否返回完整回答（包括输入提示）

        返回:
            如果return_full为True，返回完整生成文本
            否则返回三元组：(生成的回答文本, 对数似然值列表, 最后一个token的嵌入向量)
        """
        # 将输入文本转换为模型可处理的token ids，并移至GPU
        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        # 针对不同模型系列进行特殊处理
        if (
            "llama" in self.model_name.lower()
            or "falcon" in self.model_name
            or "mistral" in self.model_name.lower()
        ):
            # 某些HF模型的输入格式有变化，需要移除token_type_ids
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            # 使用eos_token作为填充token
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        # 设置停止生成的条件
        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList(
                [
                    StoppingCriteriaSub(
                        stops=self.stop_sequences,
                        initial_length=len(inputs["input_ids"][0]),
                        tokenizer=self.tokenizer,
                    )
                ]
            )
        else:
            stopping_criteria = None

        # 记录使用的温度值
        logging.debug("temperature: %f", temperature)

        # 关闭梯度计算，因为只需要前向传播
        with torch.no_grad():
            # 使用模型生成文本
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,  # 最大生成的新token数量
                return_dict_in_generate=True,  # 以字典形式返回生成结果
                output_scores=True,  # 输出每个token的分数
                output_hidden_states=True,  # 输出隐藏状态
                temperature=temperature,  # 采样温度
                do_sample=True,  # 使用采样而非贪婪解码
                stopping_criteria=stopping_criteria,  # 停止生成的条件
                pad_token_id=pad_token_id,  # 填充token的ID
            )

        # 检查生成的token数量是否超过模型限制
        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                "Generation exceeding token limit %d > %d",
                len(outputs.sequences[0]),
                self.token_limit,
            )

        # 将生成的token ids解码为文本
        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )

        # 如果需要返回完整回答，直接返回
        if return_full:
            return full_answer

        # 从完整回答中移除输入提示部分
        # 对于某些模型，需要从生成的文本中移除输入提示
        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
        else:
            raise ValueError("Have not tested this in a while.")

        # 提取生成的回答部分（移除输入提示）
        answer = full_answer[input_data_offset:]

        # 从回答中移除停止词
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            # 检查是否成功移除了所有停止词
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                error_msg = "Error: Stop words not removed successfully!"
                error_msg += f"Answer: >{answer}< "
                error_msg += f"Sliced Answer: >{sliced_answer}<"
                if "falcon" not in self.model_name.lower():
                    raise ValueError(error_msg)
                else:
                    logging.error(error_msg)

        # 移除回答首尾的空白字符
        sliced_answer = sliced_answer.strip()

        # 计算停止词出现前的token数量
        # 注意：使用stop_at索引已经排除了停止词
        # 注意：使用完整回答进行tokenization很重要，因为输入提示和生成部分之间
        # 可能存在非平凡的交互（特别是在空白字符处理方面）
        token_stop_index = self.tokenizer(
            full_answer[: input_data_offset + stop_at], return_tensors="pt"
        )["input_ids"].shape[1]
        n_input_token = len(inputs["input_ids"][0])
        n_generated = token_stop_index - n_input_token

        # 处理边缘情况：如果只生成了停止词
        if n_generated == 0:
            logging.warning("只生成了停止词。对于似然值和嵌入，将使用停止词。")
            n_generated = 1

        # 获取最后一个隐藏状态（最后一层）和最后一个token的嵌入
        # 注意：我们不希望这是停止token

        # outputs.hidden_state是一个长度为n_generated_tokens的元组
        # 第一个隐藏状态是输入token的，形状为
        #     (n_layers) x (batch_size, input_size, hidden_size)
        # （注意这包括第一个生成的token！）
        # 剩余的隐藏状态是剩余生成token的，形状为
        #    (n_layers) x (batch_size, 1, hidden_size)

        # 注意：输出嵌入的形状为(batch_size, generated_length, hidden_size)
        # 我们不会获取input_data的嵌入！因此我们从token_stop_index中减去n_tokens_in_input
        # 以获得正确的输出

        # 根据模型输出的键获取隐藏状态
        if "decoder_hidden_states" in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        # 处理隐藏状态的边缘情况
        if len(hidden) == 1:
            logging.warning(
                "只有一个生成的隐藏状态！"
                "n_generated: %d, n_input_token: %d, token_stop_index %d, "
                "last_token: %s, generation was: %s",
                n_generated,
                n_input_token,
                token_stop_index,
                self.tokenizer.decode(outputs["sequences"][0][-1]),
                full_answer,
            )
            last_input = hidden[0]
        elif (n_generated - 1) >= len(hidden):
            # 如果访问索引大于等于隐藏状态长度
            logging.error(
                "由于n_generated太大，使用最后一个状态"
                "n_generated: %d, n_input_token: %d, token_stop_index %d, "
                "last_token: %s, generation was: %s, slice_answer: %s",
                n_generated,
                n_input_token,
                token_stop_index,
                self.tokenizer.decode(outputs["sequences"][0][-1]),
                full_answer,
                sliced_answer,
            )
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        # 获取输入的最后一层
        last_layer = last_input[-1]
        # 获取输入中最后一个token的嵌入
        last_token_embedding = last_layer[:, -1, :].cpu()

        # 获取对数似然值
        # outputs.scores是生成token的logits
        # outputs.scores是一个长度为n_generated_tokens的元组
        # 每个条目的形状为(bs, vocabulary_size)
        # outputs.sequences是所有token的序列：输入和生成的
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        # Transition_scores[0]只包含第一个生成token的分数

        # 提取对数似然值
        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning("只有一个生成的token，使用其对数似然值！")
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        # 检查是否因为达到最大token限制而中断生成
        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning("生成因达到max_token限制而中断。")

        # 确保至少有一个对数似然值
        if len(log_likelihoods) == 0:
            raise ValueError

        # 返回处理后的回答文本、对数似然值列表和最后一个token的嵌入
        return sliced_answer, log_likelihoods, last_token_embedding

    def get_p_true(self, input_data):
        """获取模型对输入回答为True(A)的概率。

        该方法通过计算模型生成"A"(True)的负对数似然来评估模型认为答案正确的概率。
        计算方法参考: https://huggingface.co/docs/transformers/perplexity

        参数:
            input_data: 输入提示文本,通常包含问题和可能的答案

        返回:
            float: 负的损失值,表示模型认为答案正确的对数概率
        """
        # 在输入末尾添加" A"作为目标标记
        # 这里的"A"表示"Answer is correct"(回答正确)
        input_data += " A"

        # 将输入文本转换为模型可处理的token ids,并移至GPU
        # 这些token ids将作为模型的输入
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors="pt").to(
            "cuda"
        )["input_ids"]

        # 复制token ids作为目标标签
        # 在计算损失时，我们只关心最后一个token("A")的预测概率
        target_ids_true = tokenized_prompt_true.clone()

        # 将除最后一个token外的所有标签设为-100
        # -100是HuggingFace模型中的特殊值，表示在计算损失时忽略这些位置
        # 这样模型只会计算最后一个token(即"A")的损失
        target_ids_true[0, :-1] = -100

        # 关闭梯度计算,因为只需要前向传播来获取损失值
        with torch.no_grad():
            # 获取模型输出,包含损失值
            # 这里使用标准的语言模型训练目标：预测下一个token
            model_output_true = self.model(
                tokenized_prompt_true, labels=target_ids_true
            )

        # 提取损失值
        # 由于我们只计算了最后一个token的损失，这个损失值代表模型预测"A"的负对数似然
        loss_true = model_output_true.loss

        # 返回负的损失值作为对数概率
        # 损失值越小表示模型越倾向于生成"A"，即认为答案正确的概率越高
        # 取负是因为损失是负对数似然，取负后变为对数似然
        return -loss_true.item()
