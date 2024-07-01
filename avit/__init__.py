from .lightning_data import (
    ImageClassificationDataCollator,
    ImageClassificationDataModule,
    Caltech256DataModule
)
from .models import (
    DeiT,
    DeiTForImageClassification,
    AViT,
    AViTForImageClassification
)
from .lightning_models import (
    DeiTModel,
    AViTModel
)
from .metric_plots import (
    compute_agg_metrics,
    load_metrics,
    save_metrics,
    metrics_for_plot,
    plot
)

HF_MODEL_NAME = "facebook/deit-tiny-distilled-patch16-224"
AVIT_KWARGS = {
    "img_dim": 224,
    "patch_dim": 16,
    "in_channels": 3,
    "embed_dim": 192,
    "num_heads": 3,
    "attn_bias": True,
    "mlp_ratio": 4.,
    "depth": 12,
    "expected_depth": 8,
    "p_drop_path": 0.,
    "p_drop_path_max": 0.1,
    "num_classes": 257,
    "beta": -10.,
    "gamma": 5.,
    "eps": 0.01,
    "alpha_distr": 0.1,
    "alpha_ponder": 5.e-4
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 0xdeadbeef
