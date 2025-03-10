from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PhantomXRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 96#24
    max_iterations = 1500*2 #1500
    save_interval = 50
    experiment_name = "phantom_x_rough_blind"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=6,
        learning_rate=0.5e-3,#3.4218345324875584e-05,#1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class PhantomXMLPRoughPPORunnerCfg(PhantomXRoughPPORunnerCfg):
    experiment_name = "phantom_x_rough_blind_mlp"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
@configclass
class PhantomXHeightScanRoughPPORunnerCfg(PhantomXRoughPPORunnerCfg):
    experiment_name = "phantom_x_rough_height_scan"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

@configclass
class PhantomXHIMLocomotionRoughPPORunnerCfg(PhantomXRoughPPORunnerCfg):
    experiment_name = "phantom_x_rough_him_locomotion"
    policy = RslRlPpoActorCriticCfg(
        class_name="HIMActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

@configclass
class PhantomXDreamWaQRoughPPORunnerCfg(PhantomXRoughPPORunnerCfg):
    experiment_name = "phantom_x_rough_dreamwaq"
    policy = RslRlPpoActorCriticCfg(
        class_name="DreamWaQActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

@configclass
class PhantomXOursRoughPPORunnerCfg(PhantomXRoughPPORunnerCfg):
    experiment_name = "phantom_x_rough_ours"
    policy = RslRlPpoActorCriticCfg(
        class_name="OursActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

@configclass
class PhantomXOurs2RoughPPORunnerCfg(PhantomXRoughPPORunnerCfg):
    experiment_name = "phantom_x_rough_ours2"
    policy = RslRlPpoActorCriticCfg(
        class_name="Ours2ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

@configclass
class PhantomXOurs3RoughPPORunnerCfg(PhantomXRoughPPORunnerCfg):
    experiment_name = "phantom_x_rough_ours3"
    policy = RslRlPpoActorCriticCfg(
        class_name="Ours3ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

@configclass
class PhantomXOurs4RoughPPORunnerCfg(PhantomXRoughPPORunnerCfg):
    experiment_name = "phantom_x_rough_ours4"
    policy = RslRlPpoActorCriticCfg(
        class_name="Ours4ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

@configclass
class PhantomXOracleRoughPPORunnerCfg(PhantomXRoughPPORunnerCfg):
    experiment_name = "phantom_x_rough_oracle"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
@configclass
class PhantomXFlatPPORunnerCfg(PhantomXRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 2000 #1000
        self.experiment_name = "phantom_x_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
