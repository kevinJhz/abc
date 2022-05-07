from omegaconf import OmegaConf

from libai.config import get_config
from libai.config import LazyCall
from libai.data.build import build_nlp_train_loader, build_nlp_test_loader

from modeling.model import BERT_Classifier
from dataset.dataset import ExtractionDataSet
from configs.graph import graph
from configs.optim import optim
from configs.train import train

# optim = get_config("./optim.py").optim
# graph = get_config("./graph.py").graph
# train = get_config("./train.py").train

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(ExtractionDataSet)(
            data_path="/workspace/re",
            vocab_path="bert-base-chinese",
            # vocab_path="/workspace/re/bert-base-chinese/vocab.txt",
            is_train=True,
            indc=1000,
        )
    ],
    num_workers=4,
)
dataloader.test = [LazyCall(build_nlp_test_loader)(
    dataset=LazyCall(ExtractionDataSet)(
        data_path="/workspace/re",
        vocab_path="bert-base-chinese",
        is_train=False,
        indc=200,
    ),
    num_workers=4,
)]

bert_cfg = dict(
    vocab_size=21128,
    hidden_size=512,
    hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=512,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=64,
    num_tokentypes=2,
    add_pooling_layer=True,
    initializer_range=0.02,
    layernorm_eps=1e-12,
    bias_gelu_fusion=True,
    bias_dropout_fusion=True,
    scale_mask_softmax_fusion=True,
    apply_query_key_layer_scaling=True,
    apply_residual_post_layernorm=False,
    amp_enabled=False,
)

model = LazyCall(BERT_Classifier)(cfg=bert_cfg)

train.update(
    dict(
        recompute_grad=dict(enabled=True),
        amp=dict(enabled=True),
        output_dir="output/couplet/",
        train_micro_batch_size=128,
        test_micro_batch_size=32,
        train_epoch=20,
        train_iter=0,
        eval_period=100,
        log_period=10,
        warmup_ratio=0.01,
        topk=(1,),
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        evaluation=dict(
            enabled=False,
        )
    )
)
