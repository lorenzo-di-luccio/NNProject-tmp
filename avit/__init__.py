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
    DeiTModel
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