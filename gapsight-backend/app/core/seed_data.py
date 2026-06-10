from __future__ import annotations

from typing import Any, Dict, List


SEED_PATENTS: List[Dict[str, Any]] = [
    {
        "patent_id": "US10984207B2",
        "title": "Multi-Head Self-Attention Neural Machine Translation System",
        "abstract": (
            "A neural network architecture for sequence-to-sequence translation "
            "comprising a stack of encoder layers and decoder layers, each layer "
            "containing a multi-head self-attention sub-layer and a "
            "position-wise feed-forward sub-layer. The attention sub-layer "
            "computes scaled dot-product attention across a plurality of "
            "projection heads operating in parallel, enabling the model to "
            "jointly attend to information from different representation "
            "subspaces at different positions."
        ),
    },
    {
        "patent_id": "US11030521B1",
        "title": "Asynchronous Parameter Server for Distributed Deep Learning",
        "abstract": (
            "A distributed training system in which a plurality of worker "
            "processes compute gradients on disjoint mini-batches and push "
            "gradient updates asynchronously to a sharded parameter server. "
            "The parameter server maintains versioned weights and applies a "
            "staleness-aware learning rate correction to mitigate convergence "
            "degradation caused by delayed gradients from straggling workers."
        ),
    },
    {
        "patent_id": "US11176456B2",
        "title": "Post-Training Quantization of Neural Networks with Outlier Channels",
        "abstract": (
            "A method for converting a trained neural network from "
            "floating-point to low-bit integer representation by separating "
            "weight channels into outlier and non-outlier groups, quantizing "
            "non-outlier channels to 4-bit precision and outlier channels to "
            "8-bit precision, and applying a per-channel rescaling factor "
            "derived from a small calibration dataset to preserve task accuracy."
        ),
    },
    {
        "patent_id": "US11227210B1",
        "title": "Retrieval-Augmented Generation System for Large Language Models",
        "abstract": (
            "A question-answering system comprising a dense retriever that "
            "encodes documents and queries into a shared embedding space, a "
            "vector index supporting approximate nearest-neighbor search, and "
            "a generative language model that conditions on the top-k "
            "retrieved passages to produce grounded responses. The retriever "
            "and generator are jointly fine-tuned end-to-end using a "
            "marginal-likelihood objective."
        ),
    },
    {
        "patent_id": "US11342098B2",
        "title": "Federated Learning with Secure Aggregation of Client Updates",
        "abstract": (
            "A privacy-preserving machine learning system in which model "
            "updates computed locally on a plurality of client devices are "
            "combined at a central server using a cryptographic secure "
            "aggregation protocol. Individual client contributions remain "
            "hidden from the server while the aggregate gradient is "
            "recoverable, enabling collaborative training without exposing "
            "raw user data."
        ),
    },
    {
        "patent_id": "US11456789B2",
        "title": "Sparse Mixture-of-Experts Layer for Scalable Transformer Models",
        "abstract": (
            "A neural network layer comprising a plurality of expert "
            "sub-networks and a learned routing function that selects a small "
            "subset of experts to activate per input token. The routing "
            "function applies a top-k gating mechanism with a load-balancing "
            "auxiliary loss to ensure approximately uniform expert "
            "utilization, allowing model capacity to scale sub-linearly with "
            "compute cost."
        ),
    },
    {
        "patent_id": "US11523471B1",
        "title": "Differentiable Neural Architecture Search via Continuous Relaxation",
        "abstract": (
            "An automated method for designing neural network architectures "
            "by relaxing a discrete search space of candidate operations into "
            "a continuous mixture parameterized by learnable architecture "
            "weights. A bi-level optimization procedure alternates between "
            "updating model parameters on training data and updating "
            "architecture weights on validation data, producing a discrete "
            "architecture by selecting the highest-weight operation at each "
            "node."
        ),
    },
    {
        "patent_id": "US11604812B2",
        "title": "Activation Checkpointing for Memory-Efficient Training",
        "abstract": (
            "A GPU memory management technique for training deep neural "
            "networks in which intermediate activations are selectively "
            "discarded during the forward pass and recomputed on demand "
            "during the backward pass. A scheduler partitions the "
            "computation graph into segments and chooses checkpoint "
            "boundaries to minimize peak memory subject to a recomputation "
            "budget, enabling training of models larger than would otherwise "
            "fit in device memory."
        ),
    },
    {
        "patent_id": "US11698742B1",
        "title": "On-Device Inference of Compressed Vision Models for Mobile Hardware",
        "abstract": (
            "A system for executing convolutional neural networks on mobile "
            "and embedded devices comprising a depthwise-separable "
            "convolution kernel, an INT8 quantized weight format, and a "
            "hardware-aware compiler that fuses adjacent operations and "
            "schedules them on the device's neural processing unit. The "
            "system targets sub-100 ms latency for real-time image "
            "classification on resource-constrained edge hardware."
        ),
    },
    {
        "patent_id": "US11789456B2",
        "title": "Continual Learning with Elastic Weight Consolidation",
        "abstract": (
            "A method for training a neural network on a sequence of tasks "
            "without catastrophic forgetting by computing the Fisher "
            "information of model parameters with respect to each previous "
            "task and adding a quadratic penalty to the loss function that "
            "anchors parameters important for prior tasks. The penalty "
            "coefficient is scheduled adaptively based on a moving estimate "
            "of inter-task interference."
        ),
    },
]
