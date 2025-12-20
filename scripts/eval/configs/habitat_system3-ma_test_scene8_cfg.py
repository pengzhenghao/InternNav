from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name="system3",
        model_settings={
            "mode": "system3",  # Used by Evaluator to determine flow
            "infer_mode": "sync",  # Used by InternVLAN1Agent
            "policy_name": "InternVLAN1_Policy",  # Required by InternVLAN1Agent
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
            "vis_debug": False,  # InternVLAN1Agent needs this
            "sys3_arch": "multi",
        },
    ),
    env=EnvCfg(
        env_type="habitat",
        env_settings={
            "config_path": "scripts/eval/configs/vln_r2r_test_scene8_instr3.yaml",
        },
    ),
    eval_type="habitat_vln",
    eval_settings={
        "output_path": "./logs/habitat/test_system3-ma_scene8_instr3",
        "save_video": True,
        "epoch": 0,
        "max_steps_per_episode": 80,
        "port": "2335",
        "dist_url": "env://",
    },
)


