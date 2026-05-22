rjob submit --name=flux2-klein-rl-embeddings-expand-long-prompt \
--gpu=8 --memory=800000 \
--cpu=120 \
--charged-group=mllmexp_gpu \
--namespace=ailab-mllmexp \
--private-machine=group \
--mount=gpfs://gpfs1/mllm:/mnt/shared-storage-user/mllm \
--mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
--image=registry.h.pjlab.org.cn/ailab/pytorch:2.7.0-cuda12.8.1-py3.12-ubuntu24.04 \
-P 1 \
--host-network=true \
-e DISTRIBUTED_JOB=true \
-- bash -ex /mnt/shared-storage-user/mllm/xinglong/Pref-GRPO/scripts/preprocess/flux2_klein_expand_long_prompt/preprocess_flux2_klein_rl_embeddings.sh

