from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='system3',
        model_settings={
            "mode": "system3",  # inference mode
            "infer_mode": "sync", # Used by InternVLAN1Agent
            "policy_name": "InternVLAN1_Policy", # Required by InternVLAN1Agent
            
            # System 3 specific settings
            "vlm_api_key": "EMPTY",
            "vlm_api_url": "http://localhost:8080/v1",
            "vlm_model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            
            # Legacy settings needed for inheritance
            "width": 640,
            "height": 480,
            "hfov": 79,
            "model_path": "checkpoints/InternVLA-N1",
            "num_future_steps": 4,  
            "num_frames": 32, 
            "num_history": 8,
            "resize_w": 384, 
            "resize_h": 384,
            "predict_step_nums": 32,
            "continuous_traj": True,
            "max_new_tokens": 1024,
            "vis_debug": False,
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            # habitat sim specifications - agent, sensors, tasks, measures etc. are defined in the habitat config file
            'config_path': 'scripts/eval/configs/vln_r2r.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        # all current parse args
        "output_path": "./logs/habitat/test_system3",  # output directory for logs/results
        "save_video": True,  # whether to save videos
        "epoch": 0,  # epoch number for logging
        "max_steps_per_episode": 100,  # maximum steps per episode
        # distributed settings
        "port": "2334",  # communication port
        "dist_url": "env://",  # url for distributed setup
    },
)
