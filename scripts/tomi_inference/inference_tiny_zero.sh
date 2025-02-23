python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=$DATA_DIR/test.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$DATA_OUTPUT_PATH \
    model.path=$ACTOR_MODEL \
    +model.trust_remote_code=True \
    rollout.temperature=0.6 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=512 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.4