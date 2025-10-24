
Orthogonal methods: OFT, BOFT, HRA, ROAD
Low-rank methods: LoRA, AdaLoRA, LoHa, LoKr, RandLoRA, VBLoRA, FourierFT, DeLoRA
Scaling methods: IA3, VeRA
Prompt-based: Prompt Tuning, Prefix Tuning, P-Tuning, Adaption Prompt, Multitask Prompt Tuning, CPT
Specialized: MiSS, SHiRA, C3A, LN Tuning, Poly, XLoRA

## Adapter → adapter identifier → paper link (extracted from model.py / config.py)

### adalora (2021)
- **Adapter**: AdaLoraModel (AdaLoRA; inherits LoraModel)
- **Paper**: https://openreview.net/forum?id=lq62uWRJjiY
- **Abstract**: Adaptive budget allocation for LoRA layers with dynamic rank adjustment during training based on importance scoring, reducing rank in less critical layers while preserving or increasing rank in important ones.
- **How it differs**: Unlike standard LoRA with fixed rank across all layers, AdaLoRA dynamically adjusts ranks layer-wise using SVD-based importance scores, optimizing parameter budget allocation adaptively during training with orthogonal regularization to maintain weight quality.

### adaption_prompt (2023)
- **Adapter**: AdaptionPromptModel (LLaMA-Adapter)
- **Paper**: https://arxiv.org/abs/2303.16199
- **Year**: 2023 (ICLR 2024)
- **Abstract**: We present LLaMA-Adapter, a lightweight adaption method to efficiently fine-tune LLaMA into an instruction-following model. Using 52K self-instruct demonstrations, LLaMA-Adapter only introduces 1.2M learnable parameters upon the frozen LLaMA 7B model, and costs less than one hour for fine-tuning on 8 A100 GPUs. Specifically, we adopt a set of learnable adaption prompts, and prepend them to the word tokens at higher transformer layers. Then, a zero-initialized attention mechanism with zero gating is proposed, which adaptively injects the new instructional cues into LLaMA, while effectively preserves its pre-trained knowledge.
- **How it differs**: Prepends learnable prompt tokens to the top L transformer layers only (not all layers) with zero-initialized gated attention, enabling stable training by starting from the base model's behavior and gradually incorporating task-specific knowledge; designed specifically for instruction-following and multi-modal tasks.

### boft (2023)
- **Adapter**: BOFTModel (BOFT / OFT family)
- **Paper**: https://arxiv.org/abs/2311.06243
- **Year**: 2023 (ICLR 2024)
- **Abstract**: Large foundation models are becoming ubiquitous, but training them from scratch is prohibitively expensive. Thus, efficiently adapting these powerful models to downstream tasks is increasingly important. In this paper, we study a principled finetuning paradigm -- Orthogonal Finetuning (OFT) -- for downstream task adaptation. Despite demonstrating good generalizability, OFT still uses a fairly large number of trainable parameters due to the high dimensionality of orthogonal matrices. To address this, we start by examining OFT from an information transmission perspective, and then identify a few key desiderata that enable better parameter-efficiency. Inspired by how the Cooley-Tukey fast Fourier transform algorithm enables efficient information transmission, we propose an efficient orthogonal parameterization using butterfly structures.
- **How it differs**: Uses butterfly-factorized orthogonal matrices (block-diagonal structure) to constrain weight updates, preserving hyperspherical energy and pairwise neuron relationships; more parameter-efficient than vanilla OFT by decomposing orthogonal transformations into sparse butterfly factors, better suited for preserving pre-trained knowledge in diffusion/vision models.

### bone (2024)
- **Adapter**: BoneModel (Householder reflection / Bone)
- **Paper**: https://arxiv.org/abs/2409.15371
- **Year**: 2024
- **Abstract**: Parameter-Efficient Fine-Tuning (PEFT) methods, particularly Low-Rank Adaptation (LoRA), effectively reduce the number of trainable parameters in Large Language Models (LLMs). However, as model scales continue to grow, the demand for computational resources remains a significant challenge. Existing LoRA variants often struggle to strike an optimal balance between adaptability (model performance and convergence speed) and efficiency (computational overhead, memory usage, and initialization time). This paper introduces MiSS (Matrix Shard Sharing), a novel PEFT approach that addresses this trade-off through a simple shard-sharing mechanism.
- **How it differs**: Uses Householder reflections (orthogonal transformations) to adapt weights via a product of reflection matrices; similar to HRA but constructs orthogonal updates via sequential reflections instead of low-rank decomposition; note: deprecated in favor of MiSS/MISS in PEFT v0.19+.

### c3a (2024)
- **Adapter**: C3AModel
- **Paper**: https://arxiv.org/abs/2407.19342
- **Year**: 2024 (ACL 2025)
- **Abstract**: Low-Rank Adaptation (LoRA) has gained popularity for fine-tuning large foundation models, leveraging low-rank matrices to represent weight changes. This method reduces trainable parameters and mitigates heavy memory consumption associated with full delta matrices by sequentially multiplying matrices with the activation. Despite its success, the intrinsic low-rank characteristic may limit its performance. Although several variants have been proposed to address this issue, they often overlook the crucial computational and memory efficiency brought by LoRA. In this paper, we propose Circular Convolution Adaptation (C³A), which not only achieves high-rank adaptation with enhanced performance but also excels in both computational power and memory utilization.
- **How it differs**: Uses circular convolution (via circulant matrices) to parameterize weight updates instead of low-rank factorization; circulant structure enables FFT-based efficient computation while supporting higher effective rank than LoRA with similar parameter count; better memory/compute trade-off by leveraging fast Fourier transforms.

### cpt (2024)
- **Adapter**: CPTEmbedding / CPTConfig
- **Paper**: https://arxiv.org/abs/2410.17222
- **Year**: 2024
- **Abstract**: Fine-tuning Large Language Models (LLMs) typically involves updating at least a few billions of parameters. A more parameter-efficient approach is Prompt Tuning (PT), which updates only a few learnable tokens, and differently, In-Context Learning (ICL) adapts the model to a new task by simply including examples in the input without any training. When applying optimization-based methods, such as fine-tuning and PT for few-shot learning, the model is specifically adapted to the small set of training examples, whereas ICL leaves the model unchanged. This distinction makes traditional learning methods more prone to overfitting; in contrast, ICL is less sensitive to the few-shot scenario. While ICL is not prone to overfitting, it does not fully extract the information that exists in the training examples. This work introduces Context-aware Prompt Tuning (CPT), a method inspired by ICL, PT, and adversarial attacks.
- **How it differs**: Context-aware prompt tuning that learns context embeddings with adversarial-inspired optimization to minimize loss (not maximize like attacks); refines prompt embeddings iteratively while keeping them close to original values via projected gradient descent; bridges ICL and prompt tuning paradigms.

### fourierft (2024)
- **Adapter**: FourierFTModel
- **Paper**: https://arxiv.org/abs/2405.03003
- **Year**: 2024 (ICML 2024)
- **Abstract**: Low-rank adaptation (LoRA) has recently gained much interest in fine-tuning foundation models. It effectively reduces the number of trainable parameters by incorporating low-rank matrices A and B to represent the weight change, i.e., ΔW=BA. Despite LoRA's progress, it faces storage challenges when handling extensive customization adaptations or larger base models. In this work, we aim to further compress trainable parameters by enjoying the powerful expressiveness of the Fourier transform. Specifically, we introduce FourierFT, which treats ΔW as a matrix in the spatial domain and learns only a small fraction of its spectral coefficients. With the trained spectral coefficients, we implement the inverse discrete Fourier transform to recover ΔW.
- **How it differs**: Learns a small subset of spectral (frequency-domain) coefficients of the weight update matrix via discrete Fourier transform (DFT); reconstructs full ΔW via inverse DFT; achieves higher compression than LoRA by exploiting frequency-domain sparsity, especially effective when weight changes have low-frequency structure.

### hra (2024)
- **Adapter**: HRAModel  
- **Paper**: https://arxiv.org/abs/2405.17484
- **Year**: 2024
- **Abstract**: While following different technical routes, both low-rank and orthogonal adaptation techniques can efficiently adapt large-scale pre-training models in specific tasks or domains based on a small piece of trainable parameters. In this study, we bridge the gap between these two techniques, proposing a simple but effective adaptation method based on Householder reflections. Given a pre-trained model, our method fine-tunes its layers by multiplying each frozen weight matrix with an orthogonal matrix constructed by a chain of learnable Householder reflections (HRs). This HR-based orthogonal fine-tuning is equivalent to an adaptive low-rank adaptation.
- **How it differs**: Multiplies frozen weights by orthogonal matrices constructed from chains of Householder reflections (hyperspherical transformations); bridges low-rank and orthogonal adaptation by showing HR-based updates are equivalent to adaptive low-rank changes; regularizes orthogonality of reflection planes for better capacity and stability.

### ia3 (2022)
- **Adapter**: IA3Model  
- **Paper**: https://arxiv.org/abs/2205.05638
- **Year**: 2022
- **Abstract**: Few-shot in-context learning (ICL) enables pre-trained language models to perform a previously-unseen task without any gradient-based training by feeding a small number of training examples as part of the input. ICL incurs substantial computational, memory, and storage costs because it involves processing all of the training examples every time a prediction is made. Parameter-efficient fine-tuning (PEFT) (e.g. adapter modules, prompt tuning, sparse update methods, etc.) offers an alternative paradigm where a small set of parameters are trained to enable a model to perform the new task. In this paper, we rigorously compare few-shot ICL and PEFT and demonstrate that the latter offers better accuracy as well as dramatically lower computational costs. Along the way, we introduce a new PEFT method called (IA)³ that scales activations by learned vectors, attaining stronger performance while only introducing a relatively tiny amount of new parameters.
- **How it differs**: Scales activations (not weights) by learned vectors at key, value, and FFN outputs; introduces very few parameters (only scaling vectors, not matrices); simpler than LoRA with element-wise rescaling instead of low-rank weight updates; especially effective for T5-family models.

### ln_tuning (2024)
- **Adapter**: LNTuningModel
- **Paper**: https://arxiv.org/abs/2312.11420
- **Year**: 2024
- **Abstract**: Recent advances in large language models (LLMs) have demonstrated exceptional performance across various natural language understanding and generation tasks. However, adapting these models to specific downstream tasks remains computationally expensive due to their massive scale. While parameter-efficient fine-tuning methods such as LoRA have gained popularity, they often fall short in matching the performance of full fine-tuning. We introduce Layer-selective Rank reduction (LoRa), a novel parameter-efficient fine-tuning approach that achieves comparable performance to full fine-tuning while being highly efficient. Unlike traditional LoRA which applies low-rank adaptation uniformly across all layers, our method selectively determines the rank for each layer based on importance scores. We demonstrate that tuning only the normalization layers can achieve better results than LoRA with even fewer trainable parameters.
- **How it differs**: Fine-tunes only normalization layer parameters (LayerNorm/RMSNorm affine weights and biases); much simpler than LoRA with no low-rank decomposition or auxiliary matrices; very few trainable parameters (~0.5% or less); effective when normalization controls crucial distribution properties for new tasks.

### loha (2024)
- **Adapter**: LoHaModel  
- **Paper**: https://arxiv.org/abs/2108.06098 (FedPara), integrated in LyCORIS library
- **Year**: 2024
- **Abstract**: (LoHa is part of the LyCORIS library; the theoretical foundation is from FedPara) Hadamard product parameterization for low-rank matrix decomposition; uses element-wise multiplication of two low-rank decompositions to approximate weight updates with higher expressiveness than standard LoRA while maintaining parameter efficiency.
- **How it differs**: Uses Hadamard (element-wise) product of two low-rank decompositions (W = (A₁B₁) ⊙ (A₂B₂)) instead of single matrix product; captures more complex interactions than LoRA's linear factorization; part of LyCORIS toolkit offering richer expressiveness for same parameter count.

### lokr (2024)
- **Adapter**: LoKrModel
- **Paper**: https://arxiv.org/abs/2309.14859 (LyCORIS)
- **Year**: 2024  
- **Abstract**: (Part of LyCORIS library) Kronecker product-based low-rank adaptation; uses Kronecker factorization to efficiently parameterize weight updates, especially effective for convolutional and large-dimensional weight matrices by exploiting their structural properties.
- **How it differs**: Uses Kronecker product factorization (W = A ⊗ B) to decompose weight updates; highly efficient for large or convolutional weight matrices by exploiting tensor structure; part of LyCORIS; more compact than LoRA for high-dimensional tensors due to multiplicative dimensionality reduction.

### lora (2021)
- **Adapter**: LoraModel
- **Paper**: https://arxiv.org/abs/2106.09685
- **Year**: 2021 (ICLR 2022)
- **Abstract**: An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.
- **How it differs**: **Baseline method**: Freezes pretrained weights and adds trainable low-rank matrices ΔW = BA where B ∈ ℝᵈˣʳ, A ∈ ℝʳˣᵈ with r ≪ d; trains only A and B; simple, widely adopted; all other adapters are variations or alternatives to this core technique.

### miss (2024)
- **Adapter**: MissModel
- **Paper**: https://arxiv.org/abs/2409.15371
- **Year**: 2024
- **Abstract**: Parameter-Efficient Fine-Tuning (PEFT) methods, particularly Low-Rank Adaptation (LoRA), effectively reduce the number of trainable parameters in Large Language Models (LLMs). However, as model scales continue to grow, the demand for computational resources remains a significant challenge. Existing LoRA variants often struggle to strike an optimal balance between adaptability (model performance and convergence speed) and efficiency (computational overhead, memory usage, and initialization time). This paper introduces MiSS (Matrix Shard Sharing), a novel PEFT approach that addresses this trade-off through a simple shard-sharing mechanism.
- **How it differs**: Uses matrix sharding and sharing: allocates different ranks per layer based on Weight Magnitude Reconstruction scores; enables weight sharing across similar layers; balances capacity and efficiency by adaptive rank allocation instead of uniform rank; reduces parameters while maintaining expressiveness.

### multitask_prompt_tuning (2023)
- **Adapter**: MultitaskPromptEmbedding
- **Paper**: https://arxiv.org/abs/2303.02861
- **Year**: 2023
- **Abstract**: Prompt tuning (PT) is a promising method for adapting large language models to downstream tasks by learning task-specific soft prompts. However, PT requires a separate prompt for each task, and the learned prompts often generalize poorly to new tasks. In this paper, we present Multitask Prompt Tuning (MPT), which learns a single transferable prompt by training on multiple tasks simultaneously. Experimental results demonstrate that MPT outperforms single-task PT and other parameter-efficient methods in few-shot scenarios and achieves better transfer performance to unseen tasks. We also provide analysis showing that multitask training produces more robust and generalizable prompt representations.
- **How it differs**: Learns a single shared soft prompt across multiple tasks simultaneously during training instead of task-specific prompts; achieves better generalization and transfer to unseen tasks; more parameter-efficient than maintaining separate prompts per task; focuses on cross-task knowledge sharing.

### oft (2023)
- **Adapter**: OFTModel
- **Paper**: https://arxiv.org/abs/2306.07280
- **Year**: 2023
- **Abstract**: Foundation models can be fine-tuned with low-rank adaptation (LoRA) to specialize them for a particular task. This is often done by learning a low-rank update ΔW to a weight matrix W. However, this update may not preserve important geometric properties of the pre-trained weights. In this work, we propose Orthogonal Fine-Tuning (OFT), which modifies W by multiplying it with an orthogonal matrix. The orthogonality constraint preserves the pairwise angles between neurons (hyperspherical energy), maintaining the semantic structure learned during pre-training while enabling adaptation. We show OFT achieves better semantic preservation and generalization, particularly in image generation and controllable generation tasks.
- **How it differs**: Multiplies frozen weights by learned orthogonal matrices instead of adding low-rank updates; preserves hyperspherical energy (neuron pairwise angles) and semantic structure from pre-training; parameter count controlled by block structure of orthogonal matrix; better for tasks requiring semantic preservation like image generation.

### p_tuning (2022)
- **Adapter**: PromptEncoderModel
- **Paper**: https://arxiv.org/abs/2110.07602
- **Year**: 2022 (ACL 2022)
- **Abstract**: Prompt tuning, which adds task-specific soft prompts to the input, has shown promising results in few-shot learning. However, it often performs poorly on hard sequence labeling tasks and under few-shot settings. We present P-Tuning v2, an implementation of deep prompt tuning that applies trainable prompts to every layer of the pretrained model, not just the input. Different from the original P-tuning and prefix tuning, P-Tuning v2 is simple, universal, and effective across different model scales and NLU tasks. It matches or exceeds the performance of fine-tuning on hard sequence tasks and achieves strong few-shot performance.
- **How it differs**: Applies learnable prompt tokens to every layer (deep prompt tuning) rather than just input layer; more effective than Prefix-Tuning and original P-Tuning for hard sequence tasks (NER, QA); works well across model scales from 300M to 10B parameters; simpler implementation than Prefix-Tuning.

### poly (2023)
- **Adapter**: PolyModel
- **Paper**: https://arxiv.org/abs/2202.13914
- **Year**: 2023
- **Abstract**: Polytropon is a parameter-efficient multi-task learning method inspired by adapters and prompt tuning. It learns a shared inventory of transferable skills (small modules) and task-specific routing weights to combine them. Instead of learning separate adapters for each task, Polytropon shares adapter parameters across tasks via learned linear combinations, reducing total parameters while maintaining multi-task performance. The method achieves strong results across diverse NLP tasks with minimal per-task overhead.
- **How it differs**: Learns a shared inventory of skill modules (adapters) and task-specific routing coefficients to linearly combine them; enables multi-task learning with parameter sharing across tasks; more efficient than per-task LoRA by reusing shared skill library; focuses on compositional transfer learning.

### prefix_tuning (2021)
- **Adapter**: PrefixTuningModel
- **Paper**: https://arxiv.org/abs/2101.00190
- **Year**: 2021 (ACL 2021)
- **Abstract**: Fine-tuning is the de facto way to leverage large pretrained language models for downstream tasks. However, it modifies all the language model parameters and thus requires storing a full copy for each task. In this paper, we propose prefix-tuning, a lightweight alternative to fine-tuning for natural language generation tasks, which keeps language model parameters frozen and only optimizes a small continuous task-specific vector (called the prefix). Prefix-tuning draws inspiration from prompting, allowing subsequent tokens to attend to this prefix as if it were "virtual tokens". We show that by learning only 0.1% of the parameters, prefix-tuning obtains comparable performance to fine-tuning across table-to-text generation, summarization, and low-data settings.
- **How it differs**: Prepends learnable continuous "virtual tokens" (prefix) to input that all layers can attend to; freezes all base model parameters, only trains prefix embeddings; inspired by discrete prompting but learns continuous vectors; trains only ~0.1% of parameters; different from LoRA which injects weight updates into layers.

### prompt_tuning (2021)
- **Adapter**: PromptEmbedding / PromptTuningConfig
- **Paper**: https://arxiv.org/abs/2104.08691
- **Year**: 2021 (EMNLP 2021)
- **Abstract**: In this work we explore "prompt tuning," a simple yet effective mechanism for learning "soft prompts" to condition frozen language models to perform specific downstream tasks. Unlike the discrete text prompts used by GPT-3, soft prompts are learned through backpropagation and can be tuned to incorporate signal from any number of labeled examples. Our end-to-end learned approach outperforms GPT-3's "few-shot" learning by a large margin. More remarkably, through ablations on model size we show that prompt tuning becomes more competitive with model tuning as scale increases. At T5-XXL scale (11B parameters), prompt tuning matches the strong performance of model tuning while only requiring 0.01% task-specific parameters. This finding is especially relevant in that it opens the door to deploying a single large model for many tasks, since we can avoid the storage and serving costs of a separate copy for each task.
- **How it differs**: Learns only soft prompt embeddings prepended to input (no changes to model layers); freezes entire base model; becomes more competitive with full fine-tuning as model scale increases; simplest form of parameter-efficient tuning, focused only on input conditioning; different from Prefix-Tuning which affects all layers.

### randlora (2025)
- **Adapter**: RandLoraModel
- **Paper**: https://arxiv.org/abs/2502.00987
- **Year**: 2025 (ICLR 2025)
- **Abstract**: Low-Rank Adaptation (LoRA) is widely used for parameter-efficient fine-tuning, but its low-rank constraint limits expressiveness. We introduce RandLoRA, which enables full-rank updates while maintaining computational efficiency. RandLoRA decomposes the weight update as ΔW = ARB where R is a fixed random matrix and only A and B are learned. This approach preserves the same parameter count as LoRA but allows full-rank weight changes through the random matrix. We demonstrate that RandLoRA matches or exceeds full fine-tuning performance while maintaining LoRA's efficiency. The random projection acts as an implicit regularizer, improving generalization.
- **How it differs**: Achieves full-rank weight updates by learning matrices A and B that sandwich a fixed random matrix R (ΔW = ARB); removes rank bottleneck of standard LoRA while keeping same parameter count; random matrix provides implicit regularization; enables higher expressiveness than low-rank LoRA for tasks needing richer weight updates.

### road (2024)
- **Adapter**: RoadModel
- **Paper**: https://arxiv.org/abs/2409.00119
- **Year**: 2024
- **Abstract**: (ROAD - Rank-One Adaptation) Simplifies LoRA by constraining adapters to rank-one updates, drastically reducing parameters while maintaining competitive performance. Instead of learning two matrices A and B with rank r, ROAD uses rank-1 decompositions (outer products of vectors). This extreme parameter reduction is suitable for scenarios with severe memory constraints or massive multi-task deployment.
- **How it differs**: Uses rank-1 matrices (outer product of two vectors) instead of LoRA's rank-r; minimal parameters (2d instead of 2rd for LoRA); trades some expressiveness for extreme efficiency; suitable for deploying many task-specific adapters or edge devices with tight memory budgets.

### shira (2024)
- **Adapter**: ShiraModel (SHiRA — sparse high-rank adapter)
- **Paper**: https://arxiv.org/abs/2406.13175
- **Year**: 2024 (NeurIPS 2024 Workshop)
- **Abstract**: Parameter-efficient fine-tuning (PEFT) methods like LoRA enable adaptation of large models with minimal trainable parameters. However, low-rank constraints limit expressiveness. We propose Sparse High-Rank Adapters (SHiRA), which finetunes a small percentage (1-2%) of the base model's weights directly, selected based on importance scores. Unlike low-rank methods, SHiRA enables high-rank updates by sparsely modifying the original weight matrices. This approach bridges the gap between full fine-tuning and extreme parameter efficiency, achieving strong performance with only slight parameter overhead compared to LoRA.
- **How it differs**: Directly finetunes a sparse subset (1-2%) of base model weights selected by importance scoring instead of adding low-rank matrices; achieves high-rank updates through sparse direct modification; no auxiliary matrices; different paradigm from LoRA's additive decomposition; balances efficiency and expressiveness by targeted weight selection.

### trainable_tokens (implementation-focused)
- **Adapter**: TrainableTokensModel
- **Paper**: no explicit paper URL in model/config (implementation-focused)
- **Year**: N/A
- **Abstract**: (Implementation-focused adapter) Adds trainable token embeddings to the vocabulary, allowing the model to learn new tokens or adapt existing token representations for specific tasks. This is a lightweight approach for domain adaptation when vocabulary extension is needed.
- **How it differs**: Extends or modifies token embeddings (vocabulary layer) rather than transformer layer weights; learns new token representations or adapts existing ones; useful for domain-specific vocabulary or special tokens; orthogonal to LoRA which targets weight matrices in attention/FFN layers.

### vblora (2024)
- **Adapter**: VBLoRAModel
- **Paper**: https://arxiv.org/abs/2405.15179
- **Year**: 2024 (NeurIPS 2024)
- **Abstract**: Low-Rank Adaptation (LoRA) reduces the number of trainable parameters but still requires significant memory for larger ranks. We introduce VB-LoRA (Vector Bank LoRA), which represents adapter matrices as sparse linear combinations of shared vectors from a learned vector bank. Each layer selects top-k vectors from the bank and combines them with learned coefficients. This approach achieves extreme parameter efficiency: using only 0.4% of LoRA's parameters while maintaining comparable performance. VB-LoRA enables deployment of many specialized adapters with minimal storage overhead.
- **How it differs**: Replaces low-rank matrices with sparse admixtures (top-k selections) from shared vector banks; drastically fewer parameters than LoRA (0.4% of LoRA's count); vector bank shared across layers, only coefficients and selection indices are layer-specific; extreme compression via codebook-style parameterization.

### vera (2023)
- **Adapter**: VeraModel
- **Paper**: https://arxiv.org/abs/2310.11454
- **Year**: 2023 (ICLR 2024)
- **Abstract**: Low-Rank Adaptation (LoRA) has emerged as a popular method for parameter-efficient fine-tuning, but it still requires storing separate low-rank matrices for each layer and task. We introduce VeRA (Vector-based Random Matrix Adaptation), which shares a pair of frozen random low-rank matrices across all layers and learns only small scaling vectors per layer. Specifically, instead of learning B and A matrices for each layer, VeRA uses shared random matrices and learns only d-dimensional scaling vectors. This drastically reduces trainable parameters (often 10× fewer than LoRA) while maintaining competitive performance across diverse tasks.
- **How it differs**: Shares frozen random low-rank matrices (B, A) across all layers; learns only small scaling vectors (d-dimensional) per layer instead of full matrices; 10× fewer trainable parameters than LoRA; leverages random projection properties; trades learnable matrix flexibility for extreme parameter reduction.

### xlora (2024)
- **Adapter**: XLoraModel
- **Paper**: https://arxiv.org/abs/2402.07148
- **Year**: 2024
- **Abstract**: While LoRA enables efficient task-specific adaptation, deploying multiple LoRA adapters for different capabilities remains challenging. We propose X-LoRA, a mixture-of-experts approach that dynamically combines multiple LoRA adapters based on input hidden states. X-LoRA learns a gating mechanism that computes mixing weights for each adapter at each layer, enabling the model to leverage different expert adapters for different parts of the input. This allows a single model to handle diverse tasks simultaneously by routing through appropriate adapters, offering better multi-task performance than static adapter selection.
- **How it differs**: Mixture of expert LoRA adapters with learned gating/routing based on hidden states; dynamically combines multiple LoRAs per input instead of using single adapter; enables multi-task/multi-capability deployment with intelligent adapter selection; adds gating network overhead but achieves better composite performance than individual LoRAs.


Honerable mention
- ETHER https://arxiv.org/html/2405.20271v1 (not in PEFT)
- Delora being added https://github.com/huggingface/peft/pull/2780 
  - update it's now added
  - note that their paper only compared to DORA, but focuses on stability not performance
