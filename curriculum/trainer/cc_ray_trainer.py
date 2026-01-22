from pprint import pprint
from typing import Dict
import uuid
import numpy as np
from omegaconf import OmegaConf, open_dict
import ray
import torch
from transformers import PreTrainedTokenizer, ProcessorMixin
from torchdata.stateful_dataloader import StatefulDataLoader

from curriculum.utils.tracking import ValidationGenerationsLogger
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import (RayPPOTrainer, Role, WorkerType, ResourcePoolManager, AdvantageEstimator, compute_response_mask, apply_kl_penalty, compute_advantage)
from verl.single_controller.ray.base import RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.utils import need_reference_policy
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_timing_metrics, compute_throughout_metrics
from verl.utils.profiler.performance import marked_timer

from curriculum.workers.reward_manager.function import FunctionRewardManager
from tqdm import tqdm



class CCRayGRPOTrainer(RayPPOTrainer):
    """
    This is the distributed tainer runs on a single GPU node
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self, 
                 config, 
                 tokenizer: PreTrainedTokenizer,
                 processor: ProcessorMixin,
                 train_dataloader: StatefulDataLoader,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 reward_fn:FunctionRewardManager,
                 ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = None

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )
        
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager

        self.use_rm = False
        self.use_reward_loop = False

        self.use_critic = False
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger()

        self.train_dataloader = train_dataloader


        # ============================
        # Ref in Actor setup
        # ============================
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        
        # ============================
        # Define KL
        # ============================
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)
        else :
            raise ValueError("To have better Curriculum agent, you must enable KL in Reward")
        
        # ============================
        # Define additional config
        # ============================       
        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # ============================
        # Customised Validation
        # ============================ 
        if config.data.train_batch_size % config.actor_rollout_ref.actor.ppo_mini_batch_size != 0:
            raise ValueError("Train batch size must be divisible by actor mini batch size(global across gpu).")

        if (config.data.train_batch_size * config.actor_rollout_ref.rollout.n) % config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu != 0:
            raise ValueError("Train batch size * Rollout.n must be devisible by actor micro batch size per gpu")
        
        if config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO) and config.actor_rollout_ref.rollout.n == 1:
            raise ValueError("GRPO and RLOO algorithm need `config.actor_rollout_ref.rollout.n > 1`")

        # ============================
        # Total training step
        # ============================ 
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total Epoch: {self.config.trainer.total_epochs}")
        print(f"Total lenght of batches: {len(self.train_dataloader)}")
        print(f"Total training steps: {self.total_training_steps}")

        # Update optim.total_training_steps
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = self.total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")            

        # ============================
        # Dynamic Ray Prometheus Metrics
        # ============================ 
        self._ray_gauges = None
        if config.actor_rollout_ref.rollout.prometheus is not None and config.actor_rollout_ref.rollout.prometheus.enable:
            from ray.util import metrics
            self._ray_gauges = {}


    def init_workers(self):
        super().init_workers()

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch
    
    def fit(self):
        from omegaconf import OmegaConf
        from curriculum.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        
        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # Loop each epoch
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            # Loop each batch
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                # =====================================
                # Init Profiler
                # =====================================
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                
                # =====================================
                # Prepare batch, and gen_batch will be repated config.actor_rollout_ref.rollout.n times
                # =====================================
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch=batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # gen
                    with marked_timer("gen", timing_raw, color="red"):
                         # Note: mode is always "async" since sync mode is deprecated
                        if not self.async_rollout_mode:
                            # never reach
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # !!! Here we skip REMAX which is not used by us
                    
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] =  compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    
                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    
                    # !!! Here we skipped image images_seqlens

                    # =========================================
                    # Customised Compute Reward by reward_fn, without using Reward Model
                    # =========================================
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        # This will call RewardManager.compute_reward function
                        reward_ref = self.reward_fn.compute_reward.remote(batch) # type: ignore
                    
                    # =========================================
                    # Customised: Always calculate old_log_prob
                    # =========================================      
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)             
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = core_algos.agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item(),
                            "perf/mfu/actor_infer": old_log_prob_mfu,
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:

                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # !!! skip compute values from critic

                    with marked_timer("adv", timing_raw, color="brown"):
                        reward_tensor, reward_extra_infos_dict = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            # Log curriculum sub-scores (means) to metrics
                            for score_key, score_values in reward_extra_infos_dict.items():
                                metrics[f"reward/{score_key}"] = np.mean(score_values)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            assert isinstance(self.kl_ctrl_in_reward, core_algos.AdaptiveKLController), f"Expected AdaptiveKLController, got {type(self.kl_ctrl_in_reward)}"
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor
                        
                        # !!! Specifically for GRPO
                        batch = compute_advantage(
                            data=batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )
                        print(f"[😊 Check me 😊]  Get Current Batch info  {batch.get_data_info()}")


                    # !!!! Skip Critic update and Critic warmup
                    
                     # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # !!! skip validation

                # =========================================
                # Customised: Updated save checkpoint
                # =========================================
                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                # Profiling
                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                # Dynamically export ALL metrics to Ray Prometheus
                if  self._ray_gauges is not None:
                    from ray.util import metrics as ray_metrics
                    for k, v in metrics.items():
                        if isinstance(v, (int, float, np.floating, np.integer)):
                            # Clean key for Prometheus compatibility (only alphanumeric and underscores)
                            prom_key = k.replace("/", "_").replace(".", "_").replace("-", "_").replace("@", "_")
                            if prom_key not in self._ray_gauges:
                                self._ray_gauges[prom_key] = ray_metrics.Gauge(prom_key, description=f"Metric: {k}")
                            self._ray_gauges[prom_key].set(float(v))

                # Concise console summary
                curriculum_summary = (
                    f"Step {self.global_steps} | "
                    f"Reward: {metrics.get('reward/mean', 0):.4f} | "
                    f"SC: {metrics.get('reward/sc_score', 0):.4f} | "
                    f"Fmt: {metrics.get('reward/format_score', 0):.2f} | "
                    f"Uncert: {metrics.get('reward/uncertainty_score', 0):.4f} | "
                    f"Time: {steps_duration:.2f}s"
                )
                print(curriculum_summary)

                progress_bar.update(1)
                self.global_steps += 1


                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
