from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name="internvla_n1",
        model_settings={
            "mode": "dual_system",  # inference mode: dual_system or system2
            "model_path": "checkpoints/InternVLA-N1",  # path to model checkpoint
            "num_future_steps": 4,  # number of future steps for prediction
            "num_frames": 32,  # number of frames used in evaluation
            "num_history": 8,
            "resize_w": 384,  # image resize width
            "resize_h": 384,  # image resize height
            "predict_step_nums": 32,  # number of steps to predict
            "continuous_traj": True,  # whether to use continuous trajectory
            "max_new_tokens": 1024,  # maximum number of tokens for generation
        },
    ),
    env=EnvCfg(
        env_type="habitat",
        env_settings={
            # habitat sim specifications - agent, sensors, tasks, measures etc. are defined in the habitat config file
            "config_path": "scripts/eval/configs/vln_r2r_test_scene8_instr3.yaml",
        },
    ),
    eval_type="habitat_vln",
    eval_settings={
        # curated test split: 8 scenes × 3 instruction variants
        "output_path": "./logs/habitat/test_dual_system_scene8_instr3",
        "save_video": True,
        "epoch": 0,
        "max_steps_per_episode": 500,
        # distributed settings
        "port": "2336",
        "dist_url": "env://",
    },
)


