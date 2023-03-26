# PINYIN
*经12-计18 张诗颖 2021011056*

#### LOG
> 1. [verison1] （main分支）：正常的二元模型 \
>   正确率：（词）0.692589 
> + 模型问题：无法识别“分词”，如：“清华大学计算机系”中的“学计”难以识别 
> 2. [version2] （jieba分支）：增加了对于可能分词的处理选项 \
> 二元模型可变参数：ALTERNATIVE, RANK, sigmoid函数, alpha, beta
> + 目前实验最好的参数是：
>   Alternative = 35 \
>   Rank = 25 \
>   sigmoid = math.log() \
>   alpha = 0.9 \
>   beta = 0.2 \
>   正确率：（词）0.705328 