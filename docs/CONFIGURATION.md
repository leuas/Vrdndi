## Base Productive Model

The Basic configuration

| Parameter   | Type    | Default     | Description                                                 |
| --------------- | ----------- | --------------- | --------------------------------------------------------------- |
| `model_name`    | `str`       | `'BAAI/bge-m3'` | The base pre-trained model. Usually you won't change it |
| `seed`          | `int`       | `42`            | Random seed  |
| `compile_model` | `bool`      | `True`          | Whether to use `torch.compile` for faster inference/training.   |
| `batch_size`    | `int`       | `4`             | Number of samples processed per training step.                  |
| `total_epoch`   | `int`       | `10`            | Total number of training epochs.                                |
| `g`             | `Generator` | `None`          | Placeholder, it would be updated in the pipelines        |
| `lr`           | `float`  | `5e-5`      | Learning rate for the optimizer.                                    |
| `weight_decay` | `float`  | `0.01`      | L2 penalty applied to weights to prevent overfitting.               |
| `ignore_index` | `int`    | `-100`      | Label index to ignore when calculating loss and weight. |
| `wandb_config` | `dict`   | _{lr}_      | Dictionary configuration for Weights & Biases logging.              |
| `productive_out_feature`          | `int`    | `2`         | Output dimension for the productive classification head. |
| `interest_out_feature`            | `int`    | `2`         | Output dimension for the interest classification head.   |
| `productive_output_layer_dropout` | `float`  | `0.1`       | Dropout rate applied to the productive head.               |
| `interest_output_layer_dropout`   | `float`  | `0.1`       | Dropout rate applied to the interest head.                 |
|`use_lora`|`bool`|`True`|Enable parameter-efficient fine-tuning using LoRA.|
|`lora_rank`|`int`|`8`|The dimension (r) of the low-rank matrices.|
|`lora_alpha`|`int`|`16`|Scaling factor for LoRA weights.|
|`lora_target_modules`|`str`|`'all-linear'`|Which modules to apply LoRA to (e.g., `q_proj`, `v_proj`).|
|`productive_loss_weight`|`float`|`1`|Weight multiplier for the productive head loss.|
|`interest_loss_weight`|`float`|`1`|Weight multiplier for the interest head loss.|
|`ema_alpha`|`float`|`0.6`|Smoothing factor for Exponential Moving Average.|
|`ema_productive_weight`|`float`|`0.65`|Specific weight factor applied during EMA calculations.|
|||||

## Hybrid Productive Model

>**Note**: It's the child class of ``ProductiveModel``, so it inherit all the configuration (above) from base model.


| Parameter     | Type | Default | Description                                                     |
| ----------------- | -------- | ----------- | ------------------------------------------------------------------- |
| `num_in_feature`  | `int`    | `3`         | Number of additional input features (Duration, Time Sin, Time Cos). |
| `num_out_feature` | `int`    | `384`       | Number of dimension in projected output tensor. In Default, it's same with encoded textual tensor by Sentence Transformer                        |
| `cond_dim`        | `int`    | `1`         | Dimension size for the conditional input (Duration for now).                |
| `max_length`      | `int`    | `8094`      | Maximum sequence length (tokens) allowed for input. I set it to the maximum of BGE-M3. In most case, it won't hit that                
|`accumulation_steps`|`int`|`4`|Number of steps to accumulate gradients before updating weights.|
|`sampler_interest_ratio`|`float`|`0.5`|Ratio of "Interest" samples in a training batch.|
|`sampler_productive_ratio`|`float`|`0.5`|Ratio of "Productive" samples (Calculated as `1 - interest_ratio`).|
|`interest_label_smooth`|`float`|`0`|Label smoothing factor for the Interest loss function.|
|`productive_label_smooth`|`float`|`0`|Label smoothing factor for the Productive loss function.|
