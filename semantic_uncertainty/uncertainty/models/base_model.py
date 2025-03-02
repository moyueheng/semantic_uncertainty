from abc import ABC, abstractmethod
from typing import List, Text

STOP_SEQUENCES = ["\n\n\n\n", "\n\n\n", "\n\n", "\n", "Question:", "Context:"]


class BaseModel(ABC):
    stop_sequences: List[Text]

    @abstractmethod
    def predict(self, input_data, temperature):
        """预测方法。

        参数:
            input_data: 输入数据,可以是字符串或列表
            temperature: 温度参数,用于控制生成结果的随机性

        返回:
            str: 生成的文本
        """
        pass

    @abstractmethod
    def get_p_true(self, input_data):
        """获取模型对输入回答为True(A)的概率。

        参数:
            input_data: 输入提示文本,通常包含问题和可能的答案

        返回:
            float: 负的损失值,表示模型认为答案正确的对数概率
        """
        pass
