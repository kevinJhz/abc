from omegaconf import OmegaConf

from libai.config import get_config
from libai.config import LazyCall
from libai.data.build import build_image_train_loader, build_image_test_loader

from modeling.model import NeuralNetwork
from dataset.dataset import MnistDataSet
from configs.graph import graph
from configs.optim import optim
from configs.train import train

# optim = get_config("./optim.py").optim
# graph = get_config("./graph.py").graph
# train = get_config("./train.py").train

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_image_train_loader)(
    dataset=[
        LazyCall(MnistDataSet)(
            path="/workspace/quickstart/data/",
            is_train=True,
            # indc=1600,
        )
    ],
    num_workers=4,
)
dataloader.test = [LazyCall(build_image_test_loader)(
    dataset=LazyCall(MnistDataSet)(
        path="/workspace/quickstart/data/",
        is_train=False,
        # indc=160,
    ),
    num_workers=4,
)]

# transformer_cfg = dict(
#     vocab_size=9027,
#     max_position_embeddings=64,
#     hidden_size=512,
#     intermediate_size=512,
#     hidden_layers=6,
#     num_attention_heads=8,
#     embedding_dropout_prob=0.1,
#     hidden_dropout_prob=0.1,
#     attention_dropout_prob=0.1,
#     initializer_range=0.02,
#     layernorm_epsilon=1e-5,
#     bias_gelu_fusion=False,
#     bias_dropout_fusion=False,
#     scale_mask_softmax_fusion=False,
#     apply_query_key_layer_scaling=True,
# )
# model = LazyCall(Seq2Seq)(cfg=transformer_cfg)
model = LazyCall(NeuralNetwork)()

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
