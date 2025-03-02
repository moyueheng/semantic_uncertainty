"""Compute p_true uncertainty metric."""
import logging


def construct_few_shot_prompt(
        *, model, dataset, indices, prompt, brief, brief_always, make_prompt,
        num_generations, metric):
    """Construct few shot prompt for p_true uncertainty metric."""

    # Call model n_shots many times.
    few_shot_prompt = []
    all_responses = dict()
    for it, i in enumerate(indices):
        prompt_candidate = []
        example = dataset[i]
        question = example["question"]
        context = example["context"]
        if it != 0:
            prompt_candidate += ['\n']
        prompt_candidate += ['Question: ' + question]
        prompt_candidate += ['\nBrainstormed Answers: ']
        current_question = make_prompt(context, question, None, brief, brief_always)
        local_prompt = prompt + current_question
        logging.info('P_TRUE >> Current Question: '.ljust(25) + current_question)

        responses = []
        for j in range(num_generations + 1):

            if j == 0:
                temperature = 0.1
            else:
                temperature = 1.0

            response, _, _ = model.predict(local_prompt, temperature)
            logging.info('P_TRUE >> Current Response: '.ljust(25) + response)

            responses.append(response)
            prompt_candidate += [f'{response.strip()} \n']
            if j == 0:
                # Save most likely response and compute correctness metric for it.
                most_likely_response = response
                is_correct = metric(response, example, model)
                answers = [answer for answer in example['answers']['text']]
                logging.info('P_TRUE >> LOW-T >> true answer: '.ljust(35) + str(answers))
                logging.info('P_TRUE >> LOW-T >> acc: '.ljust(35) + str(is_correct))

        all_responses[i] = dict(
            responses=responses, most_likely_response=most_likely_response,
            is_correct=is_correct)

        prompt_candidate += ['Possible answer: ' + most_likely_response + '\n']
        prompt_candidate += ['Is the possible answer:\n']
        prompt_candidate += ['A) True\n']
        prompt_candidate += ['B) False\n']
        prompt_candidate += ['The possible answer is:']
        prompt_candidate += [' A' if is_correct else ' B']

        prompt_len = len(model.tokenizer.encode(''.join(few_shot_prompt + prompt_candidate)))
        # At test time, get a maximum of `num_generations * model.token_limit` extra tokens
        # 200 buffer for question and 'Possible Answer'.
        max_input_len = prompt_len + num_generations * model.max_new_tokens + 200

        if max_input_len < model.token_limit:
            few_shot_prompt.extend(prompt_candidate)
        else:
            logging.warning('Cutting of p_true prompt at length %d.', it)
            break

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