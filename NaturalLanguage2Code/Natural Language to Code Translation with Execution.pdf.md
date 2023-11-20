# GPT-Academic Report
## # Title:



Natural Language to Code Translation with Execution

## # Abstract:



shown great success in translating natural language to code (Chen et al., 2021;Austin et al., 2021; Li  et al., 2022, inter alia). While these models do not explicitly incorporate program semantics (i.e., execution results) during training, they are able to generate correct solutions for many problems. However, choosing a single correct program from a generated set for each problem remains challenging. In this work, we introduce execution resultbased minimum Bayes risk decoding (MBR-EXEC) for program selection and show that it improves the few-shot performance of pretrained code models on natural-language-tocode tasks. We select output programs from a generated candidate set by marginalizing over program implementations that share the same semantics. Because exact equivalence is intractable, we execute each program on a small number of test inputs to approximate semantic equivalence. Across datasets, execution or simulated execution significantly outperforms the methods that do not involve program semantics. We find that MBR-EXEC consistently improves over all execution-unaware selection methods, suggesting it as an effective approach for natural language to code translation.

## # Meta Translation

标题：带有执行的自然语言到代码翻译

作者：Freda Shi；Daniel Fried；Marjan Ghazvininejad；Luke Zettlemoyer；Sida I Wang；Meta Ai

摘要：已有研究（例如Chen等，2021；Austin等，2021；Li等，2022）在将自然语言转化为代码方面取得了巨大成功。尽管这些模型在训练过程中没有显式地纳入程序语义（即执行结果），但它们能够为许多问题生成正确的解决方案。然而，从生成的候选程序集中选择出单个正确的程序仍然具有挑战性。在本研究中，我们引入了基于执行结果的最小贝叶斯风险解码（MBR-EXEC），以进行程序选择，并证明它提高了预训练代码模型在自然语言到代码任务中的少样本性能。我们通过对共享相同语义的程序实现进行边缘化，从生成的候选程序集中选择输出程序。由于精确的等价性是难以计算的，我们在少量的测试输入上执行每个程序，以近似语义等价性。在各个数据集上，执行或模拟执行明显优于不涉及程序语义的方法。我们发现，MBR-EXEC在所有不关注执行的选择方法上始终有所改进，表明它是一种有效的自然语言到代码翻译方法。

## # Introduction

The recent success of large pretrained language models (Radford et al., 2019;Brown et al., 2020) has extended to translating natural language descriptions into executable code (Chen et al., 2021;Austin et al., 2021;Li et al., 2022, inter alia). After pretraining on large corpora of code with a simple language modeling objective, the models demonstrate the ability to follow few-shot prompts (Rad- Figure 1: Illustration of MBR-EXEC on translating natural language to Python code: we (1) sample programs from Codex (Chen et al., 2021), (2) execute each program on one test case, and (3) select the example with the minimal execution result-based Bayes risk. Numbers around dotted lines denote the 0/1 matching loss between execution results, while the Bayes risk of a program is defined by the sum of the loss between itself and other examples. In the figure, either Code #1 or Code #3 can be selected. Ground-truth program output is not needed for selection. ford et al., 2019;Brown et al., 2020) to translate natural language to various programming languages. While code sampled from such models obtains surprisingly good BLEU scores against ground-truth programs and relatively high execution accuracies, it often includes obvious mistakes, and is of much lower quality than the code written by intermediatelevel human programmers (Li et al., 2022). In addition, choosing a single correct one from a set of generated programs remains challenging.
In this work, we translate natural language to executable code with awareness of execution re-sults on a limited number of test case inputs, which we require only at inference time. Our approach is built on the hypothesis that a pretrained code model spreads probability mass over multiple semantically-equivalent code forms that implement the same functionality. Given a text description of a desired program function, we (1) sample a set of programs from a pretrained code model ( §3.1) and ( 2) select a single candidate program using execution-result-based minimum Bayes risk (MBR) decoding ( §3.2). Intuitively, we score each sampled program using its agreement to other samples in terms of execution results, and select a program with maximal overall agreement.
Our evaluation focuses on a challenging setting where only a single program can be submitted as the solution to a given problem. We show that the execution result-based selection method (i.e., MBR-EXEC) significantly outperforms all noexecution baselines across all considered datasets, despite having never executed any code during training and even when it has no access to groundtruth outputs. In addition, we show that MBR decoding with a BLEU-based risk function performs consistently well across datasets, and can be considered as a promising alternative when we are not able to execute.

近期大型预训练语言模型（Radford et al., 2019; Brown et al., 2020）的成功已经扩展到将自然语言描述转化为可执行代码（Chen et al., 2021; Austin et al., 2021; Li et al., 2022等）。在对大量代码进行预训练并采用简单的语言建模目标之后，这些模型展现出能够在少量样本提示下执行的能力（Radford et al., 2019; Brown et al., 2020），以将自然语言转化为各种编程语言。虽然从这些模型中采样得到的代码在与真实程序进行BLEU分数对比和较高的执行准确性方面表现出人们的惊讶，但它经常包含明显错误，并且质量远低于由中级人类程序员编写的代码（Li et al., 2022）。此外，从生成的代码集合中选择单个正确的代码仍然具有挑战性。

在这项工作中，我们借助在有限数量的测试样例输入上考虑执行结果的意识将自然语言转化为可执行代码，并且仅在推断时需要这些测试用例。我们的方法建立在这样一个假设上，即预训练的代码模型将概率质量分布到多个语义等价的代码形式上，这些代码形式实现了相同的功能。给定一个所需程序功能的文本描述，我们（1）从预训练的代码模型中采样一组程序（§3.1），并（2）使用基于执行结果的最小贝叶斯风险（MBR）解码（§3.2）选择一个候选程序。直观地说，我们根据每个样本程序与其它样本在执行结果上的一致性对其进行评分，并选择具有最大整体一致性的程序。

我们的评估专注于一个具有挑战性的设置，即在给定问题中只能提交一个程序作为解决方案。我们展示了基于执行结果的选择方法（即MBR-EXEC）在所有考虑的数据集上明显优于所有不执行的基准方法，即使在训练期间从未执行任何代码并且没有访问到真实输出时也是如此。此外，我们还展示了基于BLEU的风险函数的MBR解码在各个数据集上始终表现良好，可以被看作是一种有希望的替代方法，当我们无法执行代码时使用。

## # Language to Code with Neural Networks

With the progress of neural network-based language modeling and conditioned text generation, there has been much work exploring natural language to code generation with end-to-end neural model architectures (Xiao et al., 2016;Ling et al., 2016;Rabinovich et al., 2017;Dong and Lapata, 2018;Suhr et al., 2018;Xu et al., 2020;Lachaux et al., 2021, inter alia). Recently, large Transformer-based (Vaswani et al., 2017) pretrained code models have shown surprisingly strong generation performance across programming languages (Chen et al., 2021;Austin et al., 2021;Li et al., 2022, inter alia). In this work, we explore selection (i.e., inference) methods to apply to these pretrained models, showing that selecting programs using their execution results can greatly improve program generation.
Multiple benchmarks have been proposed to evaluate code model performance (Miceli Barone and Sennrich, 2017;Yin et al., 2018;Hendrycks et al., 2021;Lu et al., 2021, inter alia). In this work, we evaluate on three text-to-code datasets: MBPP (Python; Austin et al., 2021), Spider (SQL;Yu et al., 2018) and NL2Bash (Bash;Lin et al., 2018), covering a range of programming languages.

使用神经网络模型进行语言到代码的转换

随着基于神经网络的语言建模和条件文本生成的进展，已经有许多工作探索了使用端到端神经模型架构进行自然语言到代码生成的方法（Xiao等，2016；Ling等，2016；Rabinovich等，2017；Dong和Lapata，2018；Suhr等，2018；Xu等，2020；Lachaux等，2021，等）。最近，大规模的基于Transformer（Vaswani等，2017）的预训练代码模型展现出了惊人的跨编程语言的生成性能（Chen等，2021；Austin等，2021；Li等，2022，等）。在这项工作中，我们探索了适用于这些预训练模型的选择（即推理）方法，显示使用它们的执行结果来选择程序可以极大地改善程序生成。

已经提出了多个基准来评估代码模型的性能（Miceli Barone和Sennrich，2017；Yin等，2018；Hendrycks等，2021；Lu等，2021，等）。在这项工作中，我们在三个文本到代码的数据集上进行了评估：MBPP（Python；Austin等，2021），Spider（SQL；Yu等，2018）和NL2Bash（Bash；Lin等，2018），涵盖了各种编程语言。

## # Prompting Pretrained Language Models

The GPT-2 (Radford et al., 2019) and GPT-3 (Brown et al., 2020) models have shown strong prompting performance: after conditioning on a task-related prompt, the language models are often able to make accurate output predictions for unseen inputs. These results lead to prompt-based approaches for few-shot or zero-shot text classification (Shin et al., 2020;Gao et al., 2021;Min et al., 2021, inter alia), question answering (Khashabi et al., 2020), machine translation (Radford et al., 2019), and evaluation of generated text (Yuan et al., 2021), where no more than a few examples are used to construct the prompts. Few-shot examples are usually formatted into natural language prompts and continuations generated by the models for these prompts are then converted to taskspecific predictions. The prompt formatting can be either manually designed (Jiang et al., 2020) or automatically learned (Li and Liang, 2021;Lester et al., 2021). Recently, Wang et al. (2022) find that self-consistency based decoding improves chainof-thought prompting (Wei et al., 2022). We refer the readers to Liu et al. (2021) for a more comprehensive survey.
In this work, we prompt a pretrained code model (Codex; Chen et al., 2021) in a few-shot setting ( §3.1) and perform execution-based selection over the samples. We also find that the Codex model performs well with a fairly programming-languageagnostic prompt formatting (Table 1).

GPT-2 (Radford et al., 2019)和GPT-3 (Brown et al., 2020)模型在提示任务上表现出很强的性能：在以与任务相关的提示为条件下，这些语言模型经常能够对未见输入进行准确的输出预测。这些结果导致了基于提示的几次或零次样本文本分类（Shin等，2020; Gao等，2021; Min等，2021，etc.），问答（Khashabi等，2020），机器翻译（Radford等，2019）和生成文本评估（Yuan等，2021）的方法的出现，其中仅使用了少量示例来构建提示。少量示例通常以自然语言提示的形式进行格式化，并将模型为这些提示生成的延续转换为特定任务的预测。提示格式可以是手动设计的（Jiang等，2020）或自动学习的（Li和Liang，2021; Lester等，2021）。最近，Wang等（2022）发现基于自洽性的解码可以改进思维链提示（Wei等，2022）。更全面的调查请参阅Liu等（2021）。

在这项工作中，我们在少量示例的情况下通过提示预训练的代码模型（Codex; Chen等，2021）并对样本进行基于执行结果的选择。我们还发现Codex模型在相当语言无关的提示格式化上表现良好（表1）。

## # Minimum Bayes Risk Decoding

In structured prediction, Minimum Bayes risk (MBR) decoding (Bickel and Doksum, 1977) selects a structured output that minimizes the expected errors in the structure by introducing an explicit loss function to the decision criterion. This method has outperformed the maximum a posteriori (MAP) method on many tasks, including syntactic parsing (Titov and Henderson, 2006;Shi et al., 2019;Zhang et al., 2020), statistical machine translation (Kumar and Byrne, 2004;Zhang and Gildea, 2008), and neural machine translation (Eikema andAziz, 2020, 2021).
In machine translation, MBR decoding is usually implemented by reranking candidates (Goel and [CODE] with natural language descriptions and corresponding code snippets respectively. We also provide compatibility for an optional [INFO] section to provide the model extra information (e.g., the desired function identifier and example function calls) that helps code generation. In general, we expect the pretrained code models to generate a </code> token at the end of each code snippet given its pattern following ability (Brown et al., 2020;Chen et al., 2021), otherwise we truncate the generated code to a maximum of 1024 tokens.
and Byrne, 2000; Kumar and Byrne, 2004;Tromble et al., 2008, inter alia). Let F denote the input, and E denote the corresponding ground-truth translation. Given a loss function (•, •) between translations and a probability model P (E | F ), MBR decoding can be formulated as
Ê = arg min E ∈E h E∈Ee (E, E )P (E | F ),(1)
where E h is the hypothesis space, and E e is the evidence space: both are sets of possible translations. We define execution based MBR loss functions, and show that they are crucial in the sample selection processes for natural language to code with a pretrained large language model.

在结构化预测中，最小贝叶斯风险（MBR）解码（Bickel和Doksum，1977）通过引入显式损失函数到决策准则中，选择能够最小化结构中的预期错误的结构化输出。这种方法在许多任务上表现优于最大后验概率（MAP）方法，包括句法分析（Titov和Henderson，2006；Shi等，2019；Zhang等，2020）、统计机器翻译（Kumar和Byrne，2004；Zhang和Gildea，2008）以及神经机器翻译（Eikema和Aziz，2020，2021）。

在机器翻译中，MBR解码通常通过重新排序候选项进行实现（Goel和[CODE]）利用自然语言描述和相应的代码片段。我们还为可选的[INFO]部分提供了兼容性，以为模型提供额外的信息（例如所需的函数标识符和示例函数调用），有助于代码生成。总体而言，我们期望预训练代码模型在每个代码片段的最后生成一个</code>标记，以实现其遵循模式的能力（Brown等，2020；Chen等，2021），否则我们将截断生成的代码为最多1024个标记。

并且Byrne，2000；Kumar和Byrne，2004；Tromble等，2008），其中F表示输入，E表示相应的真实翻译。给定翻译之间的损失函数（•，•）和概率模型P（E | F），MBR解码可以被表述为

Ê = arg min E ∈E h E∈Ee（E，E）P（E | F），（1）

其中E h 是假设空间，E e 是证据空间：两者都是可能的翻译集合。我们定义了基于执行的MBR损失函数，并且展示了它们在使用预训练大型语言模型进行自然语言转代码的样本选择过程中的重要性。

## # Proposed Approach: MBR-EXEC

Our execution-based framework consists of two parts: (1) collecting samples from a pretrained code model ( §3.1) and (2) selecting the best candidate using minimum Bayes risk decoding ( §3.2).

我们的基于执行的框架包括两个部分：（1）从预训练的代码模型中收集样本（§3.1）和（2）使用最小贝叶斯风险解码选择最佳候选者（§3.2）。

## # Sample Collection

To obtain the corresponding code, we query the pretrained code model with few-shot prompts followed by the text description, using a unified mark-up style few-shot prompting template (Table 1). 2 In addition to the generated programs themselves, most existing models also allow us 2 While existing work on prompting language models usually requires a task-specific design of prompts (Shin et al., 2020;Zhong et al., 2021;Gao et al., 2021, inter alia), we find to have the associated probability of generating each generated token w i conditioned on the prompt tokens C = c 1 , . . . , c n and all the previously generated tokens w 1 , . . . , w i-1 , denoted by P (w i | C, w 1 , . . . w i-1 ).

为了获取相应的代码，我们使用了一个统一的标记样式的少样本提示模板（表1），通过预训练的代码模型以少量样本提示后跟文本描述进行查询。此外，除了生成的程序本身，大多数现有模型还允许我们获得每个生成的令牌wi在提示令牌C=c1,...,cn和先前生成的令牌w1,...,wi-1条件下生成的概率，表示为P(wi|C, w1,...,wi-1)（注：现有关于提示语言模型的研究通常需要针对任务进行提示的任务相关设计（Shin et al., 2020; Zhong et al., 2021; Gao et al., 2021,等），我们发现在生成的令牌wi条件下有相关的生成概率对实现此目的是必要的）。

## # Execution-Based MBR Decoding

Given a problem in its natural language description C, we sample a set of programs P = {p i } N i=1 using the method in §3.1. We formulate the executionbased MBR (MBR-EXEC) decoding by selecting
p = arg min p∈P L MBR (p; P) = arg min p∈P p ref ∈P (p, p ref )(2)
as the best candidate, where L MBR (•; •) denotes the MBR loss of a program conditioned on a set of references and is a predefined, execution-based loss function that examines the discrepancy between two programs. Intuitively, this finds a consensus candidate which has a low loss relative to all other candidates. The above implementation is an unbiased estimation of Eq (1). We introduce the following execution resultbased loss function:
(p i , p j ) = max t∈T 1 [p i (t) = p j (t)] ,
that a fairly general pattern (Table 1), which does not involve any programming language-specific information, works well across programming languages on Codex.

给定一个自然语言描述问题C，我们使用第3.1节中的方法采样一组程序P = {p i } N i=1。我们通过选择p = arg min p∈P L MBR (p; P) = arg min p∈P p ref ∈P (p, p ref )(2)，来制定基于执行结果的MBR（MBR-EXEC）解码。其中L MBR (•; •)表示程序在参考集合上条件概率的MBR损失函数，是一个预定义的基于执行结果的损失函数，用来衡量两个程序之间的差异。直观上，这个方法找到了一个一致的候选人，其损失相对于其他候选人来说很低。上述实现是Eq (1)的一个无偏估计。我们引入了以下基于执行结果的损失函数：
(p i , p j ) = max t∈T 1 [p i (t) = p j (t)]，
这是一个相对通用的模式（表1），不涉及任何特定编程语言的信息，在Codex的不同编程语言上都能很好地工作。

## # MBPP Spider NL2Bash

Greedy (3-shot) 47.3 ± 2.5 50.8 ± 2.6 52.8 ± 2.9 Sample (3-shot) 47.7 ± 1.5 48.5 ± 2.6 53.0 ± 2.9
MBR-EXEC 58.2 ± 0.3 63.6 ± 0.8 58.5 ± 0.3 There may be multiple programs receiving the same MBR loss L MBR (•; P), which are all minima. We break any ties by selecting the program with the largest likelihood among them.

贪婪法（3-shot）47.3 ± 2.5 50.8 ± 2.6 52.8 ± 2.9 采样法（3-shot）47.7 ± 1.5 48.5 ± 2.6 53.0 ± 2.9 MBR-EXEC 58.2 ± 0.3 63.6 ± 0.8 58.5 ± 0.3 可能会有多个程序接收相同的MBR损失L_MBR (•; P)，它们都是极小值。我们通过选择其中似然性最大的程序来打破平局。

## # Experiments

We evaluate ( §4.3) and analyze ( §4.4) the performance of MBR-EXEC, starting with introducing the datasets and evaluation metrics ( §4.1), as well as non-execution-based baselines ( §4.2) for MBR-EXEC. Finally, we show and discuss oracle performances on the considered tasks ( §4.5).

我们通过评估（§4.3）和分析（§4.4）MBR-EXEC的性能来进行实验，首先介绍数据集和评估指标（§4.1），以及MBR-EXEC的非执行性基线（§4.2）。最后，我们展示并讨论了所考虑任务的理想性能（§4.5）。

## # Datasets and Evaluation Metrics

We consider three datasets that cover a range of programming languages: MBPP (Python; Austin et al., 2021), Spider (SQL; Yu et al., 2018), and NL2Bash (Bash; Lin et al., 2018).
MBPP. The MBPP dataset (Austin et al., 2021) 4 consists of 974 basic Python programming problems, with 500 of them used for testing and the rest for training or few-shot prompting. There are ground-truth program and three assertions (i.e., test cases with input and ground-truth output) associated with the description of each problem. When collecting the samples, we use one assertion as the extra information ([INFO]; Table 1). 5 Programs are evaluated with execution accuracy, where a program is considered as passing if all three test cases are correct.
Spider. The Spider dataset (Yu et al., 2018) 6 is a text-to-SQL dataset, which requires a model to translate text descriptions into SQL commands. There are 7,000 examples for training and 1,034 for development. When prompting models to produce candidate commands, we concatenate the corresponding SQL table and column names as the [INFO]. Commands are evaluated with the execution accuracy, where a command is considered as passing if it returns the same result as the groundtruth command when being executed on the same database.
NL2Bash. The NL2Bash dataset (Lin et al., 2018) aims to translate natural language to bash commands. We do not include [INFO] in the sample collection process. Because it is difficult to execute bash commands in a sandbox, we split a bash command with bashlex,7 a rule-based bash parser, and use the token-level BLEU-4 score between commands as the estimation of execution result similarity. We consider a command to be unexecutable when bashlex fails to parse it. Following Lin et al. ( 2018), commands are evaluated with character-level BLEU-4 score.
Across datasets, we use 15 examples from the training set for few-shot prompting. A detailed example showing prompt formatting can be found in Appendix A. Unless otherwise specified, we collect samples by querying Codex with five different prompts, each containing 3 examples, using temperature 0.3. We combine the candidates sampled across the five prompts to get a set of candidate samples to use in our selection methods. For execution on MBPP and Spider, we apply a memory limit of 128GB and a time limit of 10 seconds on a single Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz CPU, and consider the programs that exceed these limits as inexecutable; unless otherwise specified, Figure 2: Primary evaluation results: performance of the evaluated selection criteria (best viewed in color). For each sample size, we evaluate the methods on 5 different groups of samples and report the average performance (lines) and the standard deviations (shaded regions). All samples are collected from Codex with temperature 0.3.
we only execute each program on the first test input provided for the example, and use the output for calculating the Bayes risk in the inference process.

我们考虑了三个覆盖不同编程语言的数据集：MBPP（Python；Austin等，2021年），Spider（SQL；Yu等，2018年）和NL2Bash（Bash；Lin等，2018年）。
MBPP数据集。MBPP数据集（Austin等，2021年）由974个基本Python编程问题组成，其中500个用于测试，其余用于训练或few-shot提示。每个问题的描述都与一个真实程序和三个断言（即带有输入和真实输出的测试用例）相关联。在收集样本时，我们将一个断言作为额外信息（[INFO]；表1）。根据执行准确性评估程序，只有当所有三个测试用例都正确时，程序才被认为通过。
Spider数据集。Spider数据集（Yu等，2018年）是一个文本到SQL的数据集，要求模型将文本描述转换为SQL命令。其中有7000个用于训练和1034个用于开发。在提示模型生成候选命令时，我们将相应的SQL表和列名连接在一起作为[INFO]。命令根据执行准确性进行评估，当命令在同一数据库上执行时返回与真实命令相同的结果时，命令被认为通过。
NL2Bash数据集。NL2Bash数据集（Lin等，2018年）旨在将自然语言翻译成Bash命令。在样本收集过程中，我们没有包含[INFO]。由于在沙箱中执行Bash命令很困难，我们使用基于规则的Bash解析器bashlex（7）分割Bash命令，并使用命令之间的标记级BLEU-4分数作为执行结果相似性的估计。当bashlex无法解析命令时，我们将其视为不可执行。根据Lin等人的方法（2018年），命令使用字符级BLEU-4分数进行评估。
在所有数据集中，我们从训练集中选择15个示例进行few-shot提示。附录A中提供了一个详细的示例，显示了提示格式化的方法。除非另有说明，我们通过使用包含3个示例的五个不同提示来查询Codex，并使用温度0.3来收集样本。我们将在这五个提示中取样的候选样本组合起来，形成一个候选样本集合，用于我们的选择方法。对于MBPP和Spider上的执行，我们将内存限制设为128GB，单个Intel（R）Xeon（R）CPU E5-2698 v4 @ 2.20GHz CPU的时间限制设为10秒，并将超过这些限制的程序视为不可执行；除非另有说明，否则我们仅在示例的第一个测试输入上执行每个程序，并使用输出来计算推理过程中的贝叶斯风险。

## # Baselines

We compare the most basic baselines with no selection, prompting Codex with three examples in Table 1 format: 8
• Greedy decoding. We perform token by token greedy decoding to generate the output.
• Sampling. We sample the output token by token with a fixed temperature, where we set the temperature as 0.3 in all of our experiments.
In addition, we consider the following baseline sample selection methods:
• Maximizing likelihood (ML). Given a set of sampled candidate programs, we select the one with the largest log likelihood. Formally, we select
p = arg max p∈P np i=1 P (w p,i | C, w p,1 , . . . , w p,i-1 ),
where n p denotes the number of tokens in a generated program p, and w p,i denotes its i-th token.
• Maximizing average log likelihood (MALL) across tokens. In order to address the practical issue that ML typically favors shorter sequences, we follow Chen et al. (2021) and propose another baseline that uses the average log likelihood across tokens as the selection criterion, where we 8 We use the code-davinci-001 engine throughout this work.  ,w p,).
• BLEU score based MBR (MBR-BLEU). To study the effect of execution based MBR in sample selection, we consider BLEU score based MBR, where the Bayes risk is calculated using the following risk function:
BLEU (p i , p j ) = -BLEU(p i , p j ),
where BLEU(p i , p j ) is the BLEU score of the two programs. We use character-level (MBR-charBLEU) or token-level (MBR-tokenBLEU) BLEU-4 in all of our experiments.

我们将最基本的基线模型与没有选择的方法进行比较，以Table 1中的三个示例提示Codex：
- 贪婪解码：我们采用逐词贪婪解码的方式生成输出。
- 采样：我们逐词进行采样，使用固定的温度。在我们的所有实验中，我们将温度设置为0.3。
此外，我们考虑以下基线样本选择方法：
- 最大似然 (ML)：给定一组采样的候选程序，我们选择具有最大对数似然的程序。形式上，我们选择：
   p = arg max p∈P np i=1 P(w p,i | C, w p,1 , . . . , w p,i-1 )，
其中 n p 表示生成程序 p 的标记数，w p,i 表示它的第 i 个标记。
- 平均对数似然最大化 (MALL)：为了解决 ML 倾向于较短序列的实际问题，我们按照 Chen 等人 (2021) 的方法，提出了另一个以标记的平均对数似然作为选择标准的基线模型。
- 基于 BLEU 分数的最小贝叶斯风险 (MBR-BLEU)：为了研究基于执行的 MBR 对样本选择的影响，我们考虑使用基于 BLEU 分数的 MBR，其中贝叶斯风险使用以下风险函数计算：
   BLEU(p i , p j ) = -BLEU(p i , p j )，
其中 BLEU(p i , p j ) 是两个程序的 BLEU 分数。在所有实验中，我们使用字符级别的 BLEU-4 (MBR-charBLEU) 或标记级别的 BLEU-4 (MBR-tokenBLEU)。

## # Primary Results

We evaluate MBR-EXEC on the three datasets ( §4.1) with dataset-specific metric, where we use one test case for each problem. MBR-EXEC outperforms all baselines without a selection process by a significant margin (Table 2). In addition, we find that MBR-EXEC outperforms all baseline selection methods (Figure 2), and is especially effective on the two datasets (MBPP and Spider) that use execution-based evaluation. In addition, the MBR-BLEU metrics are also strong and robust across datasets, suggesting the effectiveness of finding a consensus candidate that has generally low discrepancy with other samples. While more samples lead to better performance for most methods, MALL consistently performs worse with a larger sample size, as we find that MALL generally favors programs with unneces- Table 3: MBR-EXEC performance on greedily decoded and sampled programs: for each problem, we use 25 groups of 3-shot prompts, decode or sample one program with each prompt, and use MBR-EXEC to select the best program. For sampling with temperature 0.3, we repeat the process for 5 times and report the average performance and standard deviations. The dataset-specific metric can be found at §4.1. The best number in each row is in boldface. Note that the greedy performances are different from those reported in Table 2, as we perform MBR-EXEC here over greedy decoding outputs, while report the average performance in Table 2.
sary repetitions,9 and a larger sample size generally leads to a larger chance to have such a sample.

我们使用特定于数据集的指标在三个数据集上评估了MBR-EXEC（§4.1节）。与没有选择过程的所有基线相比，MBR-EXEC的性能大幅优于它们（表2）。此外，我们发现MBR-EXEC在基线选择方法上也表现优异（图2），尤其在使用基于执行的评估的两个数据集（MBPP和Spider）上效果显著。此外，MBR-BLEU指标在各个数据集上都较强且稳健，表明寻找与其他样本普遍差异较小的一致候选程序的有效性。虽然更多的样本对大多数方法来说可以带来更好的性能，但MALL在较大的样本量下始终表现较差，我们发现MALL通常偏好具有不必要重复的程序（表3）。较大的样本量通常意味着更大的机会有这样的样本。

## # Analysis

We analyze the performance of MBR-EXEC from the following perspectives: the effectiveness across different sample collection temperatures ( §4.4.1), the effectiveness of using groups of 3-shot prompts ( §4.4.2) and the contribution of using execution results instead of simply checking the executability of programs ( §4.4.3).

我们从以下几个方面分析了MBR-EXEC的性能：对于不同的样本收集温度的有效性(§4.4.1)，使用3个示例提示的有效性(§4.4.2)以及使用执行结果而不仅仅是检查程序的可执行性的贡献(§4.4.3)。

## # Effect of Sample Temperature

We first compare sampling with temperature 0.3 to greedy decoding (i.e., temperature τ = 0) from the Codex model (Table 3). When having the same number of examples, MBR-EXEC on sampled candidates with temperature 0.3 consistently reaches competitive or better performance than that on greedy decoded candidates. We plot the performance of MBR-EXEC for various sampling temperatures (Figure 3). Across datasets, we find that MBR-EXEC with a decoding temperature lower than 0.5 usually leads to reasonably good performance. When the temperature approaches 1.0, the results rapidly drop for all considered selection methods on MBPP and Spider; however, MALL generally achieves higher performance on NL2bash with a higher temperature.
According to the evidences discussed above, we recommend to use sampling with a low temperature (specifically, lower than 0.5) for candidate sample collection, and perform MBR-EXEC for final program selection for better results.

首先，我们将使用温度为0.3的采样方法与贪婪解码（即温度τ=0）从Codex模型进行比较（表3）。当具有相同数量的示例时，使用温度为0.3的采样候选者进行的MBR-EXEC在性能上一直保持竞争性或更好的表现，优于使用贪婪解码得到的候选者。我们绘制了MBR-EXEC在不同采样温度下的性能曲线（图3）。在数据集上，我们发现MBR-EXEC在解码温度低于0.5时通常表现良好。当温度接近1.0时，MBPP和Spider上考虑的所有选择方法的结果迅速下降；然而，在NL2bash上，MALL通常在温度较高时表现更好。

根据以上讨论的证据，我们建议使用低温（具体来说，低于0.5）的采样方法进行候选样本收集，并进行MBR-EXEC进行最终的程序选择，以获得更好的结果。

## # Effect of Different 3-shot Prompts

We analyze the necessity of choosing multiple groups of 3-shot instead of simply concatenating the available 15 examples as the prompt (Figure 4). 10 We allow different orders of the 15 examples when collecting samples. On both MBPP and NL2Bash datasets, we find that using different groups of 3-shot prompts clearly outperforms concatenating all 15 examples, suggesting that different groups of fewer-shot prompts followed by post-hoc decoding may be more effective than using all available examples for all time.

我们分析了选择多组3-shot提示的必要性，而不是简单地将可用的15个示例连接在一起作为提示（图4）。我们允许在收集样本时使用不同的15个示例的顺序。在MBPP和NL2Bash数据集上，我们发现使用不同组的3-shot提示明显优于将所有15个示例连接在一起，这表明使用较少示例的不同组来进行后续解码可能比同时使用所有可用示例更有效。

## # Executability vs. Execution Results

We perform an ablation study to identify the contribution of execution results vs. program executability (Figure 5) on the MBPP and Spider datasets. 11  We try to execute all candidates on the test cases, and perform baseline candidate methods only on the candidates that successfully execute within the time limit. On both datasets, we find that simply involving executability checking significantly helps improve the performance of all non-semantic feature-based selection methods; on Spider, applying ML over executable commands even outperforms MBR-EXEC across sample sizes.

我们对MBPP和Spider数据集进行了消融研究，以确定执行结果与程序可执行性对结果的贡献（图5）。我们尝试在测试用例上执行所有候选程序，并仅对在时间限制内成功执行的候选程序进行基线候选方法的处理。在这两个数据集上，我们发现仅涉及可执行性检查就明显有助于提高所有非语义特征为基础的选择方法的性能；在Spider数据集上，对可执行命令应用机器学习甚至超过了MBR-EXEC在样本大小上的表现。

## # Soft Loss as the Bayes Risk Function

While all the above evaluations are based on executing one test case per problem, more test cases can lead to more accurate judgments of semantic equivalence between programs (Zhong et al., 2020). Therefore, we introduce more test cases, and compare ( §3.2) with soft , a soft version of the loss function, as the Bayes risk function in MBR-EXEC. We define soft as follows:
soft (p i , p j ) = 1 |T | t∈T 1 [p i (t) = p j (t)] ,
10 We only include MBPP and NL2Bash results here as concatenating 15 Spider examples usually results in exceeding the token number limit of the pretrained models. 11 We did not include NL2bash since MBR-EXEC does not really execute the commands. However, the comparison between MBR-EXEC and MBR-tokenBLEU in Figure 3(c) shows that using an external bash parser as an executability estimator leads to more consistent and generally better performance.   which assesses equivalence based on the number of test cases that receive the same output. If there is only one test case available, and soft are equivalent.
We experiment with the MBPP dataset (Figure 6) as it provides three test cases per problem. While multiple test cases clearly outperforms MBR-EXEC with one test case across sample sizes, we did not find significant difference between hard and soft , nor between using two or three test cases.

在上述的所有评估中，都是基于每个问题执行一个测试用例，而更多的测试用例可以更准确地判断程序之间的语义等价性（Zhong等，2020）。因此，我们引入了更多的测试用例，并将其与soft进行比较（§3.2），作为MBR-EXEC中的Bayes风险函数。我们定义soft如下：
soft(p_i, p_j) = 1/|T| ∑（t∈T）1[p_i(t) = p_j(t)],
其中，1[p_i(t) = p_j(t)]根据接收到相同输出的测试用例数量来评估等价性。如果只有一个可用的测试用例，并且soft等价。
我们在MBPP数据集上进行了实验（图6），因为它为每个问题提供了三个测试用例。尽管多个测试用例在样本量上明显优于只有一个测试用例的MBR-EXEC，但我们没有发现硬和soft之间以及使用两个还是三个测试用例之间有显著差异。

## # Oracle Performance

We report the upper bound performance of all inference methods (Figure 7). Here, we define the expected Pass@K on one problem q by ExPass@K(q)
=E |P|=K max p∈P min t∈Tq 1 [p(t) = G(t)] ,
where G(t) denotes the ground-truth output for test case input t. Intuitively, to calculate the performance upper bound, a problem q is considered to be solved if there exists one program in the candidate sample set P that passes all associated test cases T q . The dataset-level expected Pass@K is defined as the average expected Pass@K over all problems.
In addition, we report the supervised performance on these datasets, where all available training data are used for model training or finetuning: for MBPP, the results are from Austin et al. (2021), where they use all 374 training examples to finetune their pretrained code model; for Spider, we compare to the current state-of-the-art result (Scholak et al., 2021); for NL2Bash, we finetune GPT-2 (Radford et al., 2019) with all training examples with the same prompting set up as Table 1.
However, it is worth noting that the upper bounds already outperform the state-of-the-art supervised performances on all datasets by a significant margin, when a reasonable amount of sample is given. This further demonstrates the effectiveness of the pretrained code models, and points out a potential next step in the direction: while such models are

我们报告了所有推理方法的性能上界（图7）。在这里，我们定义了问题q上的预期Pass@K为ExPass@K(q)
= E |P|=K max p∈P min t∈Tq 1 [p(t) = G(t)]，
其中G(t)表示测试用例输入t的真实输出。直观地说，为了计算性能上界，当候选样本集P中存在一个程序通过所有相关测试用例Tq时，问题q被认为已解决。数据集级别的预期Pass@K定义为所有问题的平均预期Pass@K。

此外，我们报告了在这些数据集上的有监督性能，其中所有可用的训练数据都用于模型训练或微调：对于MBPP，结果来自Austin等人（2021）的研究，他们使用了所有374个训练样例来微调他们的预训练代码模型；对于Spider，我们与当前最先进的结果（Scholak等人，2021）进行比较；对于NL2Bash，我们使用了与表1相同的提示设置，对GPT-2（Radford等人，2019）进行微调。

然而，值得注意的是，当给定一定数量的样本时，上界已经显著超过了所有数据集上的最先进有监督性能，这进一步证明了预训练代码模型的有效性，并指出了下一步的潜在方向：虽然这些模型已经取得了极大的成功，但如何进一步优化这些模型是一个潜在的研究方向。

## # Code #3

Code #2 Code #1 1

# 第三章节： 

Code #3

Code #2 Code #1 1

## # 

Figure 5: Comparison between applying methods to all possible candidates vs. applying methods to only executable candidates (best viewed in color), where executability-X denotes applying selection criteria X on executable candidates only. We did not include MBR-tokenBLEU and MALL and their combination with executability check in this figure for clarity -full analysis on execution vs. executability can be found in appendix B.
able to generate correct programs, designing effective inference algorithm may be a promising way towards translating natural language to code in real world applications.

图5：对比将方法应用于所有可能的候选程序和仅应用于可执行候选程序的方法（最佳观看效果为彩色），其中可执行性-X表示仅将选择准则X应用于可执行候选程序。为了清晰起见，我们没有在这个图中包含MBR-tokenBLEU和MALL以及它们与可执行性检查的组合-关于执行与可执行性的完整分析可以在附录B中找到。能够生成正确的程序，设计有效的推理算法可能是实现自然语言到代码转换的一种有希望的途径，这在实际应用中具有重要意义。

## # Discussion

We presented and systematically analyzed MBR-EXEC, an execution-based inference algorithm for pretrained language to code models, on datasets that cover three representative programming languages. Our results showed that doing execution, even with access only to inputs (not outputs) for test cases, or with only access to an executability checker, substantially helps improve the quality of generated programs especially in the settings that use execution accuracy as the evaluation metric (MBPP and Spider). Given the consistently strong 

我们在覆盖三种代表性编程语言的数据集上，提出并系统地分析了MBR-EXEC，这是一种基于执行的推理算法，用于预训练的语言到代码模型。我们的结果表明，进行执行操作，即使只有对测试用例的输入（而不是输出）的访问权限，或者只有对可执行性检查器的访问权限，也可以在生成的程序质量方面有很大的帮助，尤其是在以执行准确性为评估指标的设置中（MBPP和Spider）。考虑到一直保持强大的

## # Limitations

In this work, all selection methods are performed on top of a frozen pretrained code model (Codex; Chen et al., 2021). We note that incorporating execution information into the training or finetuning process of pretrained models may further help improve the performance. We leave the exploration of joint execution and training to future work. Figure 7: Sample size-oracle performance curves on the considered datasets. We calculate each expected Pass@K with 5 different sets of candidates for each sample size, while using the same sets to perform MBR-EXEC for fair comparison.
Harri 

在这项工作中，所有的选择方法都是在一个冻结的预训练代码模型（Codex; Chen et al., 2021）的基础上进行的。我们注意到将执行信息纳入预训练模型的训练或微调过程中可能进一步提高性能。我们将探索联合执行和训练的工作留待以后进行。图7：在考虑的数据集上的样本大小-预测准确率曲线。我们计算每个样本大小的预期Top-K准确率，对于每个样本大小有5组不同的候选程序集，同时也使用相同的程序集进行MBR-EXEC以进行公平比较。

## # Appendices A Example Prompts and Codex API Responses

We include example 3-shot prompts and corresponding Codex responses that we used in our experiments, on the three datasets (Tables 4,5,6), where we format the prompts following the patterns presented in Table 1. Data shown in the tables are collected with the greedy decoding strategy (i.e., temperature = 0), and can be found in the first line of seed 0 in our released data for each test dataset.

我们在实验中使用了三个数据集（表4、5、6）中的示例3-shot提示和相应的Codex响应，我们在这里展示了用于格式化提示的模式（表1）。表中显示的数据是使用贪婪解码策略（即temperature = 0）收集的，并且可以在每个测试数据集的seed 0的第一行中找到我们发布的数据。

## # B Full Analysis on Executability vs. Execution Result

We report the comparison between MBR-tokenBLEU and MALL vs. their combination with executability check (Figure 8; in complementary to Figure 5), where we observe that an executability checker is an effective filter to improve execution accuracies for both datasets (MBPP and Spider).
MBPP: Prompt <info>assert camel_to_snake( GoogleAssistant ) == google_assistant </info> <text>Write a function to convert camel case string to snake case string by using regex.</text> <code>import re def camel_to_snake(text):
).lower()</code> <info>assert sort_dict_item({(5, 6) : 3, (2, 3) : 9, (8, 4): 10, (6, 4): 12} ) == {(2, 3): 9, (6, 4): 12, (5, 6): 3, (8, 4): 10}</info> <text>Write a function to sort dictionary items by tuple product of keys for the given dictionary with tuple keys.</text> <code>def sort_dict_item(test_dict): res = {key: test_dict [key] for key in sorted(test_dict.keys(), key = lambda ele: ele[1] * ele [0])} return (res) </code> <info>assert reverse_list_lists ([[1, 2, 3, 4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
== [[4, 3, 2, 1], [8,7,6,5], [12,11,10,9], [16,15,14,13]]</info> <text>Write a function to reverse each list in a given list of lists.</text> <code>def reverse_list_lists(lists):
for l in lists: l.sort(reverse = True) return lists </code> <info>assert remove_Occ(\"hello\",\"l\") == \"heo\"</info> <text>Write a python function to remove first and last occurrence of a given character from the string.</text> <code>    1).. The content in the last <info>...</info> and <text>...</text> marks in the prompt corresponds to the test problem.   Figure 8: Comparison between applying methods to all possible candidates vs. applying methods to only executable candidates viewed in color), where executability-X denotes applying selection criteria X on executable candidates only. We also include the curves of MBR-EXEC for comparison.

我们报告了MBR-tokenBLEU和MALL与可执行性检查的组合的比较结果（见图8；与图5互补），我们观察到可执行性检查对两个数据集（MBPP和Spider）的执行准确性都有显著提高。以下是MBPP数据集的示例：Prompt <info>assert camel_to_snake(GoogleAssistant) == google_assistant </info> <text>编写一个函数，使用正则表达式将驼峰命名字符串转换为蛇形命名字符串。</text> <code>import re def camel_to_snake(text):
).lower()</code> <info>assert sort_dict_item({(5, 6) : 3, (2, 3) : 9, (8, 4): 10, (6, 4): 12} ) == {(2, 3): 9, (6, 4): 12, (5, 6): 3, (8, 4): 10}</info> <text>编写一个函数，通过元组键对给定字典的项目进行排序。</text> <code>def sort_dict_item(test_dict): res = {key: test_dict [key] for key in sorted(test_dict.keys(), key = lambda ele: ele[1] * ele [0])} return (res) </code> <info>assert reverse_list_lists ([[1, 2, 3, 4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
== [[4, 3, 2, 1], [8,7,6,5], [12,11,10,9], [16,15,14,13]]</info> <text>编写一个函数，颠倒给定列表中每个列表的顺序。</text> <code>def reverse_list_lists(lists):
for l in lists: l.sort(reverse = True) return lists </code> <info>assert remove_Occ(\"hello\",\"l\") == \"heo\"</info> <text>编写一个Python函数，从字符串中删除给定字符的第一个和最后一个出现的位置。</text> <code>    1)..  prompt中的最后一个<info>...</info>和<text>...</text>标记中的内容对应于测试问题。   图8：将方法应用于所有可能候选项与仅将方法应用于可执行候选项的比较（以颜色显示），其中可执行性-X表示仅对可执行候选项应用选择标准X。我们还包括MBR-EXEC曲线进行比较。

