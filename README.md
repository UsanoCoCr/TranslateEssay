# TranslateEssay
基于gpt_academic的翻译版论文汇总

## 论文汇总

### 1. [Generative Agents: Interactive Simulacra of Human Behavior](./GenerativeAgent/Generative%20Agent.pdf)

- 相关领域：人机交互、多智能体
- 读者评价：没什么干货，整篇论文主要在讲AI小镇的构建，但是并没有很格式化的说明自己干了什么，只有记忆存储的系统和ablation study让我感觉耳目一新，其他部分零零散散，害得我还得做思维导图。可能是我没有读过这篇论文，所以看这篇论文的时候感觉又臭又长，更像工程介绍而不是学术论文，如果想要做相关的project或者想要增长这方面的见识的话，读读看倒是没有什么坏处。

### 2. [TubeDETR: Spatio-Temporal Video Grounding with Transformers](./目标检测/TubeDETR.pdf)

- 相关领域：目标检测
- 读者评价：这个work主要描述了一个目标检测的任务，通过一个编码器-解码器模型做grounding，并很好的完成了知识问答的任务。作者使用了video-text一个双流encoder的方式，整合了text和frame的信息编码，并且进行了一个类似于resnet的处理，完成了对信息的编码。并且经过了一个做好了mask的时间对齐的decoder，并使用复合的损失函数进行改进，达到了一个很好的效果，网络的复杂度比较高，这个没有办法本地复现，感觉对新手也不太友好，但是可以作为一个很好的参考，能够了解一下transformer变体的结构。