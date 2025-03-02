"""Compute p_true uncertainty metric."""
import logging


def construct_few_shot_prompt(
        *, model, dataset, indices, prompt, brief, brief_always, make_prompt,
        num_generations, metric):
    """构建p_true不确定性度量的few-shot提示。

    参数:
        model: 语言模型对象
        dataset: 数据集
        indices: 用于构建few-shot示例的数据索引列表
        prompt: 基础提示模板
        brief: 简短提示模板
        brief_always: 是否始终使用简短提示
        make_prompt: 构建提示的函数
        num_generations: 每个问题生成的答案数量
        metric: 评估答案正确性的指标函数

    返回:
        few_shot_prompt: 构建的few-shot提示字符串
        all_responses: 包含每个示例生成答案的字典
        it: 实际使用的few-shot示例数量
    """

    # 存储few-shot提示和所有生成的答案
    few_shot_prompt = []
    all_responses = dict()

    # 遍历每个示例构建few-shot提示
    for it, i in enumerate(indices):
        prompt_candidate = []  # 当前示例的提示内容
        example = dataset[i]
        question = example["question"]
        context = example["context"]

        # 添加换行分隔不同示例
        if it != 0:
            prompt_candidate += ['\n']

        # 添加问题和答案部分的提示
        prompt_candidate += ['Question: ' + question]
        prompt_candidate += ['\nBrainstormed Answers: ']

        # 使用make_prompt构建完整的提示
        current_question = make_prompt(context, question, None, brief, brief_always)
        local_prompt = prompt + current_question
        logging.info('P_TRUE >> Current Question: '.ljust(25) + current_question)

        # 存储当前问题的所有生成答案
        responses = []
        
        # 生成num_generations+1个答案,第一个使用低温度
        for j in range(num_generations + 1):
            # 第一次生成使用低温度(0.1)获得最可能的答案
            # 之后使用高温度(1.0)获得更多样的答案
            temperature = 0.1 if j == 0 else 1.0

            # 使用模型生成答案
            response, _, _ = model.predict(local_prompt, temperature)
            logging.info('P_TRUE >> Current Response: '.ljust(25) + response)

            # 保存生成的答案
            responses.append(response)
            prompt_candidate += [f'{response.strip()} \n']

            # 对第一个(最可能的)答案进行正确性评估
            if j == 0:
                most_likely_response = response
                is_correct = metric(response, example, model)
                answers = [answer for answer in example['answers']['text']]
                logging.info('P_TRUE >> LOW-T >> true answer: '.ljust(35) + str(answers))
                logging.info('P_TRUE >> LOW-T >> acc: '.ljust(35) + str(is_correct))

        # 保存当前示例的所有信息
        all_responses[i] = dict(
            responses=responses, 
            most_likely_response=most_likely_response,
            is_correct=is_correct)

        # 构建提示的后半部分,包含答案判断
        prompt_candidate += ['Possible answer: ' + most_likely_response + '\n']
        prompt_candidate += ['Is the possible answer:\n']
        prompt_candidate += ['A) True\n']
        prompt_candidate += ['B) False\n']
        prompt_candidate += ['The possible answer is:']
        prompt_candidate += [' A' if is_correct else ' B']

        # 计算当前提示的token长度
        prompt_len = len(model.tokenizer.encode(''.join(few_shot_prompt + prompt_candidate)))
        # 计算最大允许的输入长度:当前长度 + 生成时的最大token数 + 200个token的缓冲
        max_input_len = prompt_len + num_generations * model.max_new_tokens + 200

        # 如果总长度在模型限制内,添加当前示例
        # 否则停止添加新的示例
        if max_input_len < model.token_limit:
            few_shot_prompt.extend(prompt_candidate)
        else:
            logging.warning('Cutting of p_true prompt at length %d.', it)
            break

    # 返回构建的few-shot提示字符串、所有生成的答案和使用的示例数
    return ''.join(few_shot_prompt), all_responses, it

def calculate_p_true(
        model, question, most_probable_answer, brainstormed_answers,
        few_shot_prompt, hint=False):
    """计算p_true不确定性度量。
    
    参数:
        model: 语言模型对象
        question: 输入问题
        most_probable_answer: 最可能的答案(温度=0.1时生成的答案)
        brainstormed_answers: 头脑风暴生成的其他答案列表(温度=1.0时生成的答案)
        few_shot_prompt: few-shot提示示例
        hint: 是否使用提示模式
    
    返回:
        log_prob: 模型认为答案正确的对数概率
    """

    # 如果有few-shot提示,添加到prompt开头
    if few_shot_prompt:
        prompt = few_shot_prompt + '\n'
    else:
        prompt = ''

    # 构建提示文本
    prompt += 'Question: ' + question
    prompt += '\nBrainstormed Answers: '
    # 添加所有生成的答案,包括最可能答案
    for answer in brainstormed_answers + [most_probable_answer]:
        prompt += answer.strip() + '\n'
    prompt += 'Possible answer: ' + most_probable_answer + '\n'

    if not hint:
        # 标准模式:让模型判断答案是否正确
        prompt += 'Is the possible answer:\n'
        prompt += 'A) True\n'
        prompt += 'B) False\n'
        prompt += 'The possible answer is:'
    else:
        # 提示模式:让模型判断答案是否与头脑风暴答案一致
        prompt += 'Do the brainstormed answers match the possible answer? Respond with A if they do, if they do not respond with B. Answer:'

    # 获取模型对答案正确性的判断概率
    log_prob = model.get_p_true(prompt)
    # prompt示例如下
    """
    Question: 什么是深度学习？
    Brainstormed Answers: 深度学习是机器学习的一个子领域，使用多层神经网络进行学习。
    深度学习是一种基于人工神经网络的机器学习方法。
    Possible answer: 深度学习是机器学习的一个子领域，使用多层神经网络进行学习。
    Is the possible answer:
    A) True
    B) False
    The possible answer is: A

    Question: 什么是机器学习？
    Brainstormed Answers: 机器学习是计算机科学的一个领域，专注于开发能够从数据中学习的算法。
    机器学习是让计算机不需要明确编程就能学习的研究。
    机器学习是人工智能的一个分支，它使用数据和算法来模仿人类学习的方式，逐渐提高其准确性。
    Possible answer: 机器学习是人工智能的一个分支，它使用数据和算法来模仿人类学习的方式，逐渐提高其准确性。
    Is the possible answer:
    A) True
    B) False
    The possible answer is:
    
    """

    return log_prob