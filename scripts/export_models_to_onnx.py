from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "Models"
PUBLIC_MODELS_DIR = ROOT / "public" / "models"


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            num_input_features,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.drop_rate = float(drop_rate)

    def forward(self, inputs: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        previous_features = inputs if isinstance(inputs, list) else [inputs]
        concated_features = torch.cat(previous_features, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
            )
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        for _, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__(
            OrderedDict(
                [
                    ("norm", nn.BatchNorm2d(num_input_features)),
                    ("relu", nn.ReLU(inplace=True)),
                    (
                        "conv",
                        nn.Conv2d(
                            num_input_features,
                            num_output_features,
                            kernel_size=1,
                            stride=1,
                            bias=False,
                        ),
                    ),
                    ("pool", nn.AvgPool2d(kernel_size=2, stride=2)),
                ]
            )
        )


class DenseNet121(nn.Module):
    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        growth_rate = 32
        block_config = (6, 12, 24, 16)
        num_init_features = 64
        bn_size = 4
        drop_rate = 0

        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            3,
                            num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.classifier = classifier

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)


def model1_classifier() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(256, 3),
    )


def model2_classifier() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 3),
    )


def export_checkpoint(checkpoint_name: str, classifier: nn.Module, onnx_name: str) -> None:
    checkpoint_path = MODELS_DIR / checkpoint_name
    output_path = PUBLIC_MODELS_DIR / onnx_name
    PUBLIC_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = DenseNet121(classifier)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
    )
    print(f"Exported {output_path.relative_to(ROOT)}")


def main() -> None:
    export_checkpoint("Model1.pth", model1_classifier(), "model1.onnx")
    export_checkpoint("Model2.pth", model2_classifier(), "model2.onnx")


if __name__ == "__main__":
    main()
