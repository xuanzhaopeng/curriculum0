import os
import hydra
import ray
from omegaconf import OmegaConf


from curriculum.trainer.cc_ray_trainer import CCRayGRPOTrainer
from curriculum.trainer.data_loader import create_dataloader
from curriculum.utils.tokenizer import get_processor, get_tokenizer
from curriculum.workers.reward_manager import SequentialFunctionRewardManager, BatchFunctionRewardManager

from verl.single_controller.ray.base import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker


@ray.remote(num_cpus=1)
class Runner:
    """
    Main entrypoint for Curriculum Agent training
    """
    def run(self, config):
        tokenizer = get_tokenizer(
            model_path=config.actor_rollout_ref.model.path,
        )
        processor = get_processor(
            model_path=config.actor_rollout_ref.model.path,
        )

        # =======================
        # Init the dataset
        # =======================
        train_dataloader = create_dataloader(config=config.data, tokenizer=tokenizer, processor=processor)


        # =======================
        # Set up Ray roles, pools and mappings
        # =======================
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
            Role.RefPolicy: ray.remote(AsyncActorRolloutRefWorker)
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        }
        role_pool_mapping = {
            Role.ActorRollout: global_pool_id,
            Role.RefPolicy: global_pool_id
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=role_pool_mapping)


        # =======================
        # Init Reward functions, the reward functions are also Ray actors
        # =======================
        if config.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.reward.reward_type}")
        RemoteRewardManager = ray.remote(RewardManager).options(num_cpus=config.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config, tokenizer)


        cc_trainer = CCRayGRPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            role_worker_mapping=role_worker_mapping, # type: ignore
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn, # type: ignore
        )
        cc_trainer.init_workers()
        cc_trainer.fit()

# use Hydra to load the configs
@hydra.main(config_path='../configs', config_name='curriculum_config')
def main(config):
    """
    Main entry point to start the Ray
        * read configurations
        * initialise the ray
        * start the the Ray Actor
    """
    #========= Config parsing and Update ==============
    assert os.path.exists(config.reward.reward_function) is True, f"the reward_function file is not exist"
    assert config.reward.reward_function_name is not None, f"the reward_function_name must be defined"

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                "VLLM_USE_V1": "1",
            }
        }
        # https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html
        ray.init(
            address="auto",
            runtime_env=runtime_env,
        )

        try:
            from pprint import pprint
            print("=" * 60)
            print("Trainer Configration:")
            pprint(OmegaConf.to_container(config, resolve=True))
            print("=" * 60)

            runner = Runner.remote()
            ray.get(runner.run.remote(config)) # type: ignore
        finally:
            if ray.is_initialized():
                ray.shutdown()

if __name__ == '__main__':
    main()