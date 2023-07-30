# Stable-Diffusion-Regularization-Images

## Tool
https://github.com/bmaltais/kohya_ss

## Training 
images: 14
Reg Images: 200 from here https://github.com/hack-mans/Stable-Diffusion-Regularization-Images

## Command
``` bash
accelerate launch --num_cpu_threads_per_process=2 "./sdxl_train_network.py" \
  --enable_bucket \
  --min_bucket_reso=256 \
  --max_bucket_reso=2048 \
  --pretrained_model_name_or_path="/checkpoints/sd_xl_base_1.0.safetensors" \
  --train_data_dir="/training/sakamoto/train_man" \
  --reg_data_dir="/training/sakamoto/reg_man" \
  --resolution="1024,1024" \
  --output_dir="/training/sakamoto/output" \
  --logging_dir="/training/sakamoto/logging" \
  --network_alpha="1" \
  --save_model_as=safetensors \
  --network_module=networks.lora \
  --text_encoder_lr=0.0004 \
  --unet_lr=0.0004 \
  --network_dim=256 \
  --output_name="djsakamotolora" \
  --lr_scheduler_num_cycles="10" \
  --no_half_vae --learning_rate="0.0004" \
  --lr_scheduler="cosine" \
  --train_batch_size="1" \
  --max_train_steps="3000" \
  --save_every_n_epochs="1" \
  --mixed_precision="bf16" \
  --save_precision="bf16" \
  --cache_latents \
  --cache_latents_to_disk \
  --optimizer_type="Adafactor" \
  --gradient_checkpointing \
  --optimizer_args scale_parameter=False relative_step=False warmup_init=False \
  --max_data_loader_n_workers="0" \
  --bucket_reso_steps=64 \
  --xformers \
  --bucket_no_upscale
```
