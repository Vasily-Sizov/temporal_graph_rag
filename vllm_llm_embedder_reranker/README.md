# Components
- **LLM**: https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ (16384 context)
- **Embedder**: https://huggingface.co/deepvk/USER-bge-m3
- **Reranker**: https://huggingface.co/BAAI/bge-reranker-v2-m3

# Endpoints:
- **LLM**: http://{IP}:8001/v1
- **Embedder**: http://{IP}:8006/v1
- **Reranker**: http://{IP}:8010/v1


- `df -h`

/dev/mapper/ubuntu--vg-ubuntu--lv  194G   18G  168G  10% /

# llm-qwen32
- `docker compose up -d llm-qwen32`
- `docker logs -f llm-qwen32`
- `nvidia-smi`
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 PCIe               On  |   00000000:02:03.0 Off |                    0 |
| N/A   42C    P0             87W /  350W |   47809MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          334426      C   VLLM::EngineCore                      47800MiB |
+-----------------------------------------------------------------------------------------+
- `df -h`
/dev/mapper/ubuntu--vg-ubuntu--lv  194G   63G  123G  34% /


# embed-ru
- `docker compose up -d embed-ru`
- `docker logs -f embed-ru`
- `nvidia-smi`
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 PCIe               On  |   00000000:02:03.0 Off |                    0 |
| N/A   40C    P0             85W /  350W |   49385MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          334426      C   VLLM::EngineCore                      47800MiB |
|    0   N/A  N/A          335104      C   VLLM::EngineCore                       1570MiB |
+-----------------------------------------------------------------------------------------+
- `df -h`
/dev/mapper/ubuntu--vg-ubuntu--lv  194G   65G  122G  35% /


# reranker-ru
- `docker compose up -d reranker-ru`
- `docker logs -f reranker-ru`
- `nvidia-smi`
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 PCIe               On  |   00000000:02:03.0 Off |                    0 |
| N/A   60C    P0            326W /  350W |   51944MiB /  81559MiB |     96%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          334426      C   VLLM::EngineCore                      47988MiB |
|    0   N/A  N/A          335104      C   VLLM::EngineCore                       1570MiB |
|    0   N/A  N/A          335580      C   VLLM::EngineCore                       2366MiB |
+-----------------------------------------------------------------------------------------+
- `df -h`
/dev/mapper/ubuntu--vg-ubuntu--lv  194G   67G  119G  36% /