# Extra information

- An example of the setup file can be found here `settings.json`


## Prepare network.onnx model

```python
import torch
from collections import OrderedDict

# TODO: replace this with your real model constructor
def build_model():
    return SmallLeNet2()
    # raise NotImplementedError("Replace build_model() with your model constructor.")

def load_checkpoint_as_model(path: str):
    """Load a checkpoint that might be a full Module or a (wrapped) state_dict."""
    obj = torch.load(path, map_location="cpu", weights_only=True)

    # Case 1: saved entire module
    if isinstance(obj, torch.nn.Module):
        return obj

    # Case 2: saved dict (could contain state_dict or be a bare state_dict)
    if isinstance(obj, (dict, OrderedDict)):
        if "state_dict" in obj:
            state = obj["state_dict"]
        elif "model_state_dict" in obj:
            state = obj["model_state_dict"]
        elif "model" in obj and isinstance(obj["model"], torch.nn.Module):
            return obj["model"]
        else:
            # Assume bare state_dict
            state = obj
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    # Strip 'module.' prefix if present (from DataParallel/DistributedDataParallel)
    cleaned = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())

    model = build_model()
    res = model.load_state_dict(cleaned, strict=False)

    # Print any key mismatches to help you spot config mistakes
    missing = getattr(res, "missing_keys", [])
    unexpected = getattr(res, "unexpected_keys", [])
    if missing:
        print(f"Warning: missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"Warning: unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected)>10 else ''}")

    return model

def export_model_to_onnx(
    model: torch.nn.Module,
    filename="network.onnx",
    in_channels=3,  # change to 10 if your model expects 10 bands
    height=64,
    width=64,
    opset=10
):
    model.eval()
    model.cpu()

    dummy_input = torch.randn(1, in_channels, height, width)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            filename,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
    print(f"Model successfully exported to: {filename}")


model = load_checkpoint_as_model("wildfire_classifier.pth")
export_model_to_onnx(model, "network.onnx", in_channels=3, height=64, width=64, opset=10)
```